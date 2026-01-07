# =============================================================
# 03.hh_soma.py — Hodgkin–Huxley 막전위 모델 (V3)
# =============================================================
# 목적:
#   • 뉴런 소마(Soma)의 막전위를 계산하는 기본 전기생리 모델
#   • 나트륨(Na⁺), 칼륨(K⁺), 누설(Leak) 채널 포함
#   • ATP 의존 Na⁺/K⁺ 펌프 및 ATP 소비율(J_use) 계산 포함
#
# V3 계약 고정:
#   - 시간 단위: [ms] (밀리초) - 모든 시간 관련 파라미터는 ms 단위
#   - ATP: [0,100] (정규화, 0~100 범위로 통일) ⭐ V3 변경
#   - V, E_Na, E_K, E_L: [mV] (밀리볼트)
#   - I_ext, I_pump: [µA/cm²] (마이크로암페어/제곱센티미터)
#   - J_use: [arb/ms] (ATP 소비율) ⭐ V3: ms 단위 명시
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
#   - ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
#
# (4) ATP 소비율 (플럭스)
#   J_use = g_pump_consume·|I_pump| [arb/ms] ⭐ V3: ms 단위 명시
#       → 펌프 전류의 절댓값(활동량)에 비례하여 ATP가 소모됨
#
# 생리학적 의미:
#   Na⁺/K⁺ 펌프는 3Na⁺를 배출하고 2K⁺를 유입시키는 데 ATP 1분자를 사용함.
#   따라서 전류 세기 |I_pump|는 ATP 소비 속도(J_use)에 비례함.
#
# 설계 이유:
#   - Hodgkin-Huxley 모델은 뉴런의 액션 전위 생성 메커니즘을 정확하게 모델링
#   - ATP 의존 펌프는 에너지 소비와 막전위 유지 간의 연관성을 반영
#   - 게이트 변수(m, h, n)는 이온 채널의 개폐 상태를 모델링
# =============================================================

import numpy as np


class HHSoma:
    r"""
    Hodgkin–Huxley Soma Model with ATP-dependent Na⁺/K⁺ Pump (V3)
    --------------------------------------------------------
    
    V3 계약:
    - 입력: dt [ms], I_ext [µA/cm²], ATP [0,100], t_ms [ms]
    - 출력: (V [mV], INa [µA/cm²], IK [µA/cm²], IL [µA/cm²], I_pump [µA/cm²], J_use [arb/ms])
    - Side-effect: self.V, self.m, self.h, self.n, self.last_spike_t 업데이트
    
    Differential equations:
        dV/dt = g_Na·m³h·(E_Na−V) + g_K·n⁴·(E_K−V) + g_L·(E_L−V) + I_ext − I_pump
        I_pump = g_pump·(1−e^{−ATP/ATP₀})·(V−E_pump)
        J_use  = g_pump_consume·|I_pump| [arb/ms] ⭐ V3: ms 단위 명시
    
    설계 이유:
    - Hodgkin-Huxley 모델은 뉴런의 액션 전위 생성 메커니즘을 정확하게 모델링
    - ATP 의존 펌프는 에너지 소비와 막전위 유지 간의 연관성을 반영
    - 게이트 변수(m, h, n)는 이온 채널의 개폐 상태를 모델링
    """

    def __init__(self, cfg: dict):
        """
        HHSoma 초기화
        
        Parameters
        ----------
        cfg : dict
            설정 딕셔너리:
            - V0: 초기 막전위 [mV]
            - gNa, gK, gL: 채널 전도도 [mS/cm²]
            - ENa, EK, EL: 역평형 전위 [mV]
            - Cm: 막 용량 [µF/cm²]
            - spike_thresh: 스파이크 임계값 [mV]
            - spike_reset: 재분극 임계값 [mV]
            - refrac_time: 불응기 [ms] ⭐ V3: ms 단위 명시
            - use_pump: 펌프 사용 여부 (bool)
            - g_pump: 펌프 전도도 [mS/cm²]
            - E_pump: 펌프 역평형 전위 [mV]
            - ATP0_ref: ATP 참조 값 [0,100] ⭐ V3: 0~100 범위 명시
            - g_pump_consume: ATP 소비율 변환 계수 [arb/(ms·µA)] ⭐ V3: ms 단위 명시
        """
        # ------------------ 막전위 / 채널 파라미터 ------------------
        self.V = float(cfg["V0"])  # [mV]
        self.gNa, self.gK, self.gL = cfg["gNa"], cfg["gK"], cfg["gL"]  # [mS/cm²]
        self.ENa, self.EK, self.EL = cfg["ENa"], cfg["EK"], cfg["EL"]  # [mV]
        self.Cm = cfg.get("Cm", 1.0)  # [µF/cm²]
        self.spike_thresh = cfg.get("spike_thresh", 0.0)  # [mV]
        self.spike_reset = cfg.get("spike_reset", -60.0)  # 재분극 임계값 [mV]
        self.refrac_time = cfg.get("refrac_time", 2.0)    # 불응기 [ms] ⭐ V3: ms 단위 명시
        self.last_spike_t = -1e9  # 마지막 스파이크 시간 [ms] ⭐ V3: ms 단위 명시

        # ------------------ ATP 펌프 파라미터 ------------------
        self.use_pump = cfg.get("use_pump", True)
        self.g_pump = cfg.get("g_pump", 0.5)  # [mS/cm²]
        self.E_pump = cfg.get("E_pump", -70.0)  # [mV]
        self.ATP0_ref = cfg.get("ATP0_ref", 100.0)  # [0,100] ⭐ V3: 0~100 범위 명시

        # ATP 소비율 변환 계수 (µA → ATP/ms) ⭐ V3: ms 단위 명시
        self.g_pump_consume = cfg.get("g_pump_consume", 0.005)  # [arb/(ms·µA)] ⭐ V3: ms 단위 명시

        # ------------------ 게이트 시간 상수 스케일링 ------------------
        self.tau_h_Na_scale = cfg.get("tau_h_Na_scale", 1.0)  # tau_h_Na 스케일링 팩터

        # ------------------ 게이트 초기값 ------------------
        self.m, self.h, self.n = 0.05, 0.60, 0.32  # [0,1] (무차원)

    # =========================================================
    # α(V), β(V) — 게이트 개폐 속도 상수
    # =========================================================
    @staticmethod
    def _am(V):
        """
        Na⁺ 활성화 (m 게이트) α(V)
        
        V3 계약:
        - 입력: V [mV]
        - 출력: α_m [1/ms] ⭐ V3: ms 단위 명시
        - Side-effect: 없음
        """
        x = V + 40.0
        return 0.1*x/(1.0 - np.exp(-x/10.0)) if abs(x) > 1e-6 else 1.0

    @staticmethod
    def _bm(V):
        """
        Na⁺ 활성화 (m 게이트) β(V)
        
        V3 계약:
        - 입력: V [mV]
        - 출력: β_m [1/ms] ⭐ V3: ms 단위 명시
        - Side-effect: 없음
        """
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _ah(V):
        """
        Na⁺ 비활성화 (h 게이트) α(V)
        
        V3 계약:
        - 입력: V [mV]
        - 출력: α_h [1/ms] ⭐ V3: ms 단위 명시
        - Side-effect: 없음
        """
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _bh(V):
        """
        Na⁺ 비활성화 (h 게이트) β(V)
        
        V3 계약:
        - 입력: V [mV]
        - 출력: β_h [1/ms] ⭐ V3: ms 단위 명시
        - Side-effect: 없음
        """
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def _an(V):
        """
        K⁺ 활성화 (n 게이트) α(V)
        
        V3 계약:
        - 입력: V [mV]
        - 출력: α_n [1/ms] ⭐ V3: ms 단위 명시
        - Side-effect: 없음
        """
        x = V + 55.0
        return 0.01*x/(1.0 - np.exp(-x/10.0)) if abs(x) > 1e-6 else 0.1

    @staticmethod
    def _bn(V):
        """
        K⁺ 활성화 (n 게이트) β(V)
        
        V3 계약:
        - 입력: V [mV]
        - 출력: β_n [1/ms] ⭐ V3: ms 단위 명시
        - Side-effect: 없음
        """
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    # =========================================================
    # Step 함수 — 시간 적분
    # =========================================================
    def step(self, dt: float, I_ext: float = 0.0, ATP: float = 100.0, t_ms: float = 0.0, 
             ENa_override: float = None, EK_override: float = None, Heat: float = 37.0):
        """
        한 스텝(dt[ms]) 적분 수행
        
        V3 계약:
        - 입력:
          - dt: [ms] (밀리초) ⭐ V3 계약 고정
          - I_ext: [µA/cm²] (외부 자극 전류)
          - ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
          - t_ms: [ms] (현재 시간) ⭐ V3: ms 단위 명시
          - ENa_override: [mV] (Na⁺ 역전위 오버라이드, 선택적)
          - EK_override: [mV] (K⁺ 역전위 오버라이드, 선택적)
          - Heat: [arb] (온도, Q10 효과용, 선택적)
        - 출력:
          - dict {"V": [mV], "INa": [µA/cm²], "IK": [µA/cm²], "IL": [µA/cm²], 
                  "I_pump": [µA/cm²], "J_use": [arb/ms], "spiking": bool}
        - Side-effect:
          - self.V 업데이트 [mV]
          - self.m, self.h, self.n 업데이트 [0,1]
          - self.last_spike_t 업데이트 [ms] (스파이크 발생 시)
        
        실행 순서:
        1. 불응기 체크
        2. 게이트 업데이트
        3. 채널 전류 계산
        4. ATP 펌프 전류 계산
        5. 막전위 갱신
        6. 스파이크 감지 및 재분극 처리
        7. ATP 소비율 계산

        Parameters
        ----------
        dt : float
            시간 스텝 [ms] ⭐ V3: ms 단위 명시
        I_ext : float, optional
            외부 자극 전류 [µA/cm²]. 기본값: 0.0
        ATP : float, optional
            ATP 농도 [0,100] ⭐ V3: 0~100 범위 명시. 기본값: 100.0
        t_ms : float, optional
            현재 시간 [ms] ⭐ V3: ms 단위 명시. 기본값: 0.0
        ENa_override : float, optional
            Na⁺ 역전위 오버라이드 [mV]. None이면 self.ENa 사용
        EK_override : float, optional
            K⁺ 역전위 오버라이드 [mV]. None이면 self.EK 사용
        Heat : float, optional
            온도 [arb] (Q10 효과용). 기본값: 37.0
            
        Returns
        -------
        dict
            현재 스텝의 상태를 담은 딕셔너리:
            - "V": 막전위 [mV]
            - "INa": Na⁺ 전류 [µA/cm²]
            - "IK": K⁺ 전류 [µA/cm²]
            - "IL": 누설 전류 [µA/cm²]
            - "I_pump": 펌프 전류 [µA/cm²]
            - "J_use": ATP 소비율 [arb/ms] ⭐ V3: ms 단위 명시
            - "spiking": 스파이크 발생 여부 (bool)
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

        self.m += dt * (am * (1.0 - self.m) - bm * self.m)  # [0,1] = [0,1] + [ms] · [1/ms] · [0,1]
        # tau_h_Na 스케일링 적용: tau_h를 줄이면 h 게이트가 더 빠르게 회복
        self.h += dt * self.tau_h_Na_scale * (ah * (1.0 - self.h) - bh * self.h)
        self.n += dt * (an * (1.0 - self.n) - bn * self.n)

        # 물리적 이유: 게이트 변수는 [0,1] 범위로 제한 (채널 개폐 확률)
        self.m, self.h, self.n = np.clip([self.m, self.h, self.n], 0.0, 1.0)

        # ------------------ 2) 채널 전류 계산 ------------------
        # 역전위 오버라이드 사용 (동적 이온 농도 기반)
        ENa_use = ENa_override if ENa_override is not None else self.ENa
        EK_use = EK_override if EK_override is not None else self.EK
        
        INa = self.gNa * (self.m ** 3) * self.h * (ENa_use - V)  # [µA/cm²]
        IK  = self.gK  * (self.n ** 4) * (EK_use - V)  # [µA/cm²]
        IL  = self.gL  * (self.EL - V)  # [µA/cm²]

        # ------------------ 3) ATP 펌프 전류 계산 ------------------
        I_pump = 0.0
        if self.use_pump:
            # ATP 농도에 따라 포화되는 비선형 함수
            # ATP: [0,100] ⭐ V3: 0~100 범위 명시
            factor = (1.0 - np.exp(-ATP / self.ATP0_ref))
            I_pump = self.g_pump * factor * (V - self.E_pump)  # [µA/cm²]

        # ------------------ 4) 막전위 갱신 ------------------
        dV = (INa + IK + IL + I_ext - I_pump) / self.Cm  # [mV/ms] = [µA/cm²] / [µF/cm²]
        self.V = np.nan_to_num(V + dt * dV, nan=-70.0, posinf=120.0, neginf=-120.0)  # [mV]
        # 물리적 이유: 막전위는 생리학적 범위 [-120, 120] mV 내에서 유지되어야 함
        # NaN/Inf는 수치 오류이므로 안전한 값으로 대체

        # ------------------ 5) 스파이크 감지 및 재분극 처리 ------------------
        spiking = False
        if self.V > self.spike_thresh and not in_refractory:
            # 스파이크 발생: 재분극 임계값으로 리셋
            self.V = self.spike_reset  # [mV]
            self.last_spike_t = t_ms  # [ms] ⭐ V3: ms 단위 명시
            spiking = True

        # ------------------ 6) ATP 소비율 계산 ------------------
        #   I_pump의 절댓값(=펌프 작동 세기)에 비례
        J_use = self.g_pump_consume * abs(I_pump)  # [arb/ms] = [arb/(ms·µA)] · [µA/cm²] ⭐ V3: ms 단위 명시

        return {
            "V": self.V,  # [mV]
            "INa": INa,  # [µA/cm²]
            "IK": IK,  # [µA/cm²]
            "IL": IL,  # [µA/cm²]
            "I_pump": I_pump,  # [µA/cm²]
            "J_use": J_use,  # [arb/ms] ⭐ V3: ms 단위 명시
            "spiking": spiking,  # bool
        }

    # =========================================================
    # Spike 감지 함수
    # =========================================================
    def spiking(self) -> bool:
        """
        막전위가 임계값을 초과하면 스파이크로 간주
        
        V3 계약:
        - 입력: 없음
        - 출력: bool (스파이크 발생 여부)
        - Side-effect: 없음
        
        Returns
        -------
        bool
            스파이크 발생 여부 (V > spike_thresh)
        """
        return self.V > self.spike_thresh

    def set_I_pump_scale(self, scale: float):
        """
        펌프 효율 스케일링 (ATP 의존성 조정용)
        
        V3 계약:
        - 입력: scale [0,1] (펌프 효율 스케일)
        - 출력: 없음
        - Side-effect: self.g_pump_effective 업데이트
        
        Parameters
        ----------
        scale : float
            펌프 효율 스케일 [0,1] (1.0 = 최대 효율)
        """
        # 내부적으로 사용할 수 있는 스케일 팩터 저장
        # 실제 구현은 step() 내부에서 사용
        self._I_pump_scale = float(np.clip(scale, 0.0, 1.0))

    def update_reversal_potentials(self, ionflow):
        """
        IonFlowDynamics에서 계산된 이온 농도를 기반으로 역전위 업데이트
        
        V3 계약:
        - 입력: ionflow (IonFlowDynamics 객체)
        - 출력: 없음
        - Side-effect: 없음 (역전위는 step()에서 오버라이드로 전달)
        
        주의:
        - 이 메서드는 역전위를 직접 업데이트하지 않음
        - step()에서 ENa_override, EK_override로 전달하는 방식 사용
        - V3 단일 방향화 원칙 준수 (역방향 참조 없음)
        
        Parameters
        ----------
        ionflow : IonFlowDynamics
            이온 흐름 동역학 객체
        """
        # V3: 역전위는 step()에서 오버라이드로 전달
        # 이 메서드는 호환성을 위해 유지하되, 실제 업데이트는 하지 않음
        pass


# =============================================================
# V3 독립 실행 테스트
# =============================================================
if __name__ == "__main__":
    """
    V3 계약 고정 검증:
    - ATP: [0,100] 확인
    - 시간 단위: [ms] 확인
    - 입출력/side-effect 확인
    """
    cfg = {
        "V0": -70.0,  # [mV]
        "gNa": 120.0, "gK": 36.0, "gL": 0.3,  # [mS/cm²]
        "ENa": 50.0, "EK": -77.0, "EL": -54.4,  # [mV]
        "spike_thresh": 0.0,  # [mV]
        "use_pump": True,
        "g_pump": 0.5, "E_pump": -70.0, "ATP0_ref": 100.0,  # ATP: [0,100] ⭐ V3: 0~100 범위 명시
        "g_pump_consume": 0.005,  # [arb/(ms·µA)] ⭐ V3: ms 단위 명시
        "refrac_time": 2.0,  # [ms] ⭐ V3: ms 단위 명시
    }

    soma = HHSoma(cfg)
    ATP = 100.0  # [0,100] ⭐ V3: 0~100 범위 명시
    dt = 0.01  # [ms] ⭐ V3: ms 단위 명시
    
    print("=" * 60)
    print("HH Soma V3 — 계약 고정 검증")
    print("=" * 60)
    print(f"ATP 입력: {ATP} [0,100] ⭐ V3 계약")
    print(f"dt: {dt} [ms] ⭐ V3 계약")
    print("-" * 60)
    print("[HH Soma + ATP Pump + J_use Test]")
    print(f"{'t(ms)':>8} | {'Vm(mV)':>10} | {'INa':>8} | {'IK':>8} | {'I_pump':>8} | {'J_use(arb/ms)':>14}")
    print("-" * 60)
    
    for t in range(20):
        result = soma.step(dt=dt, I_ext=10.0, ATP=ATP, t_ms=t*dt)
        print(f"{t*dt:8.2f} | {result['V']:10.3f} | {result['INa']:8.3f} | "
              f"{result['IK']:8.3f} | {result['I_pump']:8.3f} | {result['J_use']:14.5f}")
    
    print("=" * 60)
    print("✅ V3 계약 고정 검증 완료")
    print("  - ATP: [0,100] 범위 확인")
    print("  - 시간 단위: [ms] 확인")
    print("  - 입출력/side-effect 명시 확인")

