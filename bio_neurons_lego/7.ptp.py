# =============================================================
# ptp.py — Post-Tetanic Potentiation (PTP) only
# =============================================================
# 개념/수식 요약
# -------------------------------------------------------------
# PTP는 고빈도 자극(tetanus) 후 수 초~수십 초 동안
#  방출확률 p 또는 시냅스 이득 w를 일시적으로 증가시키는
#  Ca²⁺-의존 단기 가소성.
#
# 상태변수:
#   R(t) : PTP “잔여 강화량”(무차원, 0 이상)
#
# 동역학:
#   dR/dt = -R / τ_ptp  +  A(Ca_res) · Σ_k δ(t - t_k)
#
#   • τ_ptp : PTP 감쇠 시정수 (초 단위; 실험적으로 10~60 s)
#   • A(Ca_res) : 스파이크 직후 잔여 Ca²⁺(또는 Ca 이벤트 S)에 비례하는 증분
#       A(Ca) = g_ptp · (Ca_norm)^n / ( (Ca_norm)^n + K^n )
#     여기서 Ca_norm = clamp( (Ca - C0)/(Cmax - C0), 0, 1 )
#
# 적용:
#   p_eff = clamp( p0 * (1 + R), 0, 1 )
#   또는
#   w_eff = w0 * (1 + R)
#
# 주의:
#   • 본 클래스는 “PTP 항만” 제공. 단기 facilitation/depression(τ_f, τ_d)은 포함하지 않음.
#   • Ca 입력은 네가 가진 CaVesicle의 (Ca, S) 중 하나를 전달하면 됨.
# =============================================================

from dataclasses import dataclass

@dataclass
class PTPConfig:
    tau_ptp_s: float = 20.0     # PTP 감쇠 시정수 [s]
    g_ptp: float    = 1.2       # PTP 첨가 이득(스파이크당 최대 증분 스케일)
    K_half: float   = 0.25      # Ca_norm의 반포화점 (0~1 범위)
    hill_n: int     = 3         # Hill 계수 (비선형 민감도)
    R_clip: tuple   = (0.0, 3.0)# R(t) 안전 범위 (증폭 한계)
    Ca_thr: float   = 0.0       # Ca 임계값 (μM)
    gain: float     = 1.0       # PTP 이득
    decay: float    = 0.001     # PTP 감쇠 계수
    # p0, w0는 외부 시냅스가 갖고 있고, 여기서는 효과만 계산해 제공

class PTPPlasticity:
    r"""
    Post-Tetanic Potentiation (PTP) — Ca²⁺-dependent short-term potentiation

    State:
        R(t) ≥ 0  : residual potentiation, decays with decay rate

    Dynamics:
        dR/dt = -decay * R + gain * (S - Ca_thr) * δ(t - t_k)

    Effective modulation:
        p_eff = clamp(p0*(1 + R), 0, 1)
        w_eff = w0*(1 + R)
    """
    def __init__(self, cfg: PTPConfig):
        self.R = 0.0
        self.Ca_thr = cfg.Ca_thr
        self.gain = cfg.gain
        self.decay = cfg.decay

    def on_spike(self, S):
        """스파이크 직후 호출. S는 μM 단위로 입력됨"""
        S = min(S, 200.0)          # μM 상한선
        if S > self.Ca_thr:
            self.R += self.gain * (S - self.Ca_thr)

    def step(self, dt_ms):
        """dt_ms 만큼 시간 전진 (선형 감쇠)"""
        self.R *= (1.0 - self.decay * dt_ms)
        return self.R

    # --- 외부에 줄 모듈레이션 팩터 ---
    def p_eff(self, p0: float) -> float:
        """방출확률 p의 PTP 적용값"""
        return max(0.0, min(1.0, p0 * (1.0 + self.R)))

    def w_eff(self, w0: float) -> float:
        """가중치/시냅스 이득의 PTP 적용값(상한은 외부에서 관리)"""
        return w0 * (1.0 + self.R)