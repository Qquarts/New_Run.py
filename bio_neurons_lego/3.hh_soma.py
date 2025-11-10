# =============================================================
# hh_soma.py — Hodgkin–Huxley 막전위 모델 (ATP 펌프 + ATP 소비율 포함)
# =============================================================
# 목적:
#   • 뉴런 소마(Soma)의 막전위를 계산하는 기본 전기생리 모델
#   • 나트륨(Na⁺), 칼륨(K⁺), 누설(Leak) 채널 포함
#   • ATP 의존 Na⁺/K⁺ 펌프 및 ATP 소비율(J_use) 계산 포함
#
# -------------------------------------------------------------
# [핵심 수식 요약]
# -------------------------------------------------------------
# (1) 막전위 방정식
#   C_m·dV/dt = g_Na·m³h·(E_Na−V)
#              + g_K·n⁴·(E_K−V)
#              + g_L·(E_L−V)
#              + I_ext − I_pump
#
# (2) 게이트 동역학
#   dm/dt = α_m(V)(1−m) − β_m(V)m
#   dh/dt = α_h(V)(1−h) − β_h(V)h
#   dn/dt = α_n(V)(1−n) − β_n(V)n
#
# (3) ATP 의존 펌프 전류
#   I_pump = g_pump·(1−e^{−ATP/ATP₀})·(V−E_pump)
#
# (4) ATP 소비율 (플럭스)
#   J_use = g_pump_consume·|I_pump|
#       → 펌프 전류의 절댓값(활동량)에 비례하여 ATP가 소모됨
#
# 생리학적 의미:
#   Na⁺/K⁺ 펌프는 3Na⁺를 배출하고 2K⁺를 유입시키는 데 ATP 1분자를 사용함.
#   따라서 전류 세기 |I_pump|는 ATP 소비 속도(J_use)에 비례함.
#
# =============================================================

import numpy as np


class HHSoma:
    r"""
    Hodgkin–Huxley Soma Model with ATP-dependent Na⁺/K⁺ Pump
    --------------------------------------------------------
    dV/dt = g_Na·m³h·(E_Na−V) + g_K·n⁴·(E_K−V) + g_L·(E_L−V) + I_ext − I_pump
    I_pump = g_pump·(1−e^{−ATP/ATP₀})·(V−E_pump)
    J_use  = g_pump_consume·|I_pump|
    """

    def __init__(self, cfg: dict):
        # ------------------ 막전위 / 채널 파라미터 ------------------
        self.V = float(cfg["V0"])
        self.gNa, self.gK, self.gL = cfg["gNa"], cfg["gK"], cfg["gL"]
        self.ENa, self.EK, self.EL = cfg["ENa"], cfg["EK"], cfg["EL"]
        self.Cm = cfg.get("Cm", 1.0)
        self.spike_thresh = cfg.get("spike_thresh", 0.0)
        self.spike_reset = cfg.get("spike_reset", -60.0)  # 재분극 임계값
        self.refrac_time = cfg.get("refrac_time", 2.0)    # 불응기 (ms)
        self.last_spike_t = -1e9  # 마지막 스파이크 시간

        # ------------------ ATP 펌프 파라미터 ------------------
        self.use_pump = cfg.get("use_pump", True)
        self.g_pump = cfg.get("g_pump", 0.5)
        self.E_pump = cfg.get("E_pump", -70.0)
        self.ATP0_ref = cfg.get("ATP0_ref", 100.0)

        # ATP 소비율 변환 계수 (µA → ATP/ms)
        self.g_pump_consume = cfg.get("g_pump_consume", 0.005)

        # ------------------ 게이트 시간 상수 스케일링 ------------------
        self.tau_h_Na_scale = cfg.get("tau_h_Na_scale", 1.0)  # tau_h_Na 스케일링 팩터

        # ------------------ 게이트 초기값 ------------------
        self.m, self.h, self.n = 0.05, 0.60, 0.32

    # =========================================================
    # α(V), β(V) — 게이트 개폐 속도 상수
    # =========================================================
    @staticmethod
    def _am(V):
        """Na⁺ 활성화 (m 게이트) α(V)"""
        x = V + 40.0
        return 0.1*x/(1.0 - np.exp(-x/10.0)) if abs(x) > 1e-6 else 1.0

    @staticmethod
    def _bm(V):
        """Na⁺ 활성화 (m 게이트) β(V)"""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _ah(V):
        """Na⁺ 비활성화 (h 게이트) α(V)"""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _bh(V):
        """Na⁺ 비활성화 (h 게이트) β(V)"""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def _an(V):
        """K⁺ 활성화 (n 게이트) α(V)"""
        x = V + 55.0
        return 0.01*x/(1.0 - np.exp(-x/10.0)) if abs(x) > 1e-6 else 0.1

    @staticmethod
    def _bn(V):
        """K⁺ 활성화 (n 게이트) β(V)"""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    # =========================================================
    # Step 함수 — 시간 적분
    # =========================================================
    def step(self, dt: float, I_ext: float = 0.0, ATP: float = 100.0, t_ms: float = 0.0):
        """
        한 스텝(dt[ms]) 적분 수행:
        - 게이트 갱신
        - 이온 전류 계산
        - ATP 펌프 전류 및 ATP 소비율 계산
        - 스파이크 감지 및 재분극 처리
        """
        V = self.V

        # ------------------ 0) 불응기 체크 ------------------
        in_refractory = (t_ms - self.last_spike_t) < self.refrac_time
        if in_refractory:
            # 불응기 중에는 외부 자극 무시
            I_ext = 0.0

        # ------------------ 1) 게이트 업데이트 ------------------
        am, bm = self._am(V), self._bm(V)
        ah, bh = self._ah(V), self._bh(V)
        an, bn = self._an(V), self._bn(V)

        self.m += dt * (am * (1.0 - self.m) - bm * self.m)
        # tau_h_Na 스케일링 적용: tau_h를 줄이면 h 게이트가 더 빠르게 회복
        self.h += dt * self.tau_h_Na_scale * (ah * (1.0 - self.h) - bh * self.h)
        self.n += dt * (an * (1.0 - self.n) - bn * self.n)

        # [0,1] 범위로 제한
        self.m, self.h, self.n = np.clip([self.m, self.h, self.n], 0.0, 1.0)

        # ------------------ 2) 채널 전류 계산 ------------------
        INa = self.gNa * (self.m ** 3) * self.h * (self.ENa - V)
        IK  = self.gK  * (self.n ** 4) * (self.EK - V)
        IL  = self.gL  * (self.EL - V)

        # ------------------ 3) ATP 펌프 전류 계산 ------------------
        I_pump = 0.0
        if self.use_pump:
            # ATP 농도에 따라 포화되는 비선형 함수
            factor = (1.0 - np.exp(-ATP / self.ATP0_ref))
            I_pump = self.g_pump * factor * (V - self.E_pump)

        # ------------------ 4) 막전위 갱신 ------------------
        dV = (INa + IK + IL + I_ext - I_pump) / self.Cm
        self.V = np.nan_to_num(V + dt * dV, nan=-70.0, posinf=120.0, neginf=-120.0)

        # ------------------ 5) 스파이크 감지 및 재분극 처리 ------------------
        if self.V > self.spike_thresh and not in_refractory:
            # 스파이크 발생: 재분극 임계값으로 리셋
            self.V = self.spike_reset
            self.last_spike_t = t_ms

        # ------------------ 6) ATP 소비율 계산 ------------------
        #   I_pump의 절댓값(=펌프 작동 세기)에 비례
        J_use = self.g_pump_consume * abs(I_pump)

        #   (대안식) 이온 플럭스 기반 모델:
        #   J_use = self.g_pump_consume * abs(INa + IK)

        return self.V, INa, IK, IL, I_pump, J_use

    # =========================================================
    # Spike 감지 함수
    # =========================================================
    def spiking(self) -> bool:
        """막전위가 임계값을 초과하면 스파이크로 간주"""
        return self.V > self.spike_thresh


# =============================================================
# 단독 테스트 (시각적 검증용)
# =============================================================
if __name__ == "__main__":
    cfg = {
        "V0": -70.0,
        "gNa": 120.0, "gK": 36.0, "gL": 0.3,
        "ENa": 50.0, "EK": -77.0, "EL": -54.4,
        "spike_thresh": 0.0,
        "use_pump": True,
        "g_pump": 0.5, "E_pump": -70.0, "ATP0_ref": 100.0,
        "g_pump_consume": 0.005
    }

    soma = HHSoma(cfg)
    ATP = 100.0
    print("[HH Soma + ATP Pump + J_use Test]")
    print(" t(ms) |    Vm(mV) |     INa |      IK |   I_pump |   J_use")
    print("-----------------------------------------------------------")
    for t in range(20):
        Vm, INa, IK, IL, Ip, J = soma.step(dt=0.01, I_ext=10.0, ATP=ATP)
        print(f"{t*0.01:6.2f} | {Vm:10.3f} | {INa:8.3f} | {IK:8.3f} | {Ip:8.3f} | {J:8.5f}")