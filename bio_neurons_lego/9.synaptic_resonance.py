# =============================================================
# synaptic_resonance.py — Ca²⁺ 기반 커플링 게인 공명 모델
# =============================================================
# 목적:
#   시냅스의 내부 위상 θ(t)가 상위 발진자(DTG System)의 위상 φ(t)에
#   동기화(phase locking)되는 과정을 모델링한다.
#
#   이때 결합 강도(coupling gain, K)가 칼슘 신호(S)에 의해
#   동적으로 조절되는 구조를 포함한다.
#
#   ┌──────────────────────────────────────┐
#   │ dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)   │
#   └──────────────────────────────────────┘
#
#   • θ : 시냅스 고유 위상 (synaptic phase)
#   • φ : 상위 발진 위상 (DTG phase)
#   • ω : 고유 위상속도 (intrinsic angular frequency)
#   • K : 기본 커플링 게인 (baseline coupling strength)
#   • λ : Ca²⁺ 민감도 (coupling modulation coefficient)
#   • S : Ca²⁺ 정규화 농도 (0~1)
#
#   ⇒ Ca²⁺가 높을수록 결합이 강해지고(동기화↑),
#      Ca²⁺가 낮을수록 각자 독립 진동(비동기화).
#
#   물리적 의미:
#       “Ca²⁺는 시냅스 공명의 커플링 게인으로 작용한다.”
#
# =============================================================

import numpy as np

class SynapticResonance:
    r"""
    SynapticResonance — Ca²⁺-modulated Phase Coupling Resonator
    ------------------------------------------------------------
    Differential equation:
        dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)

    where:
        • θ : synaptic phase (local oscillator)
        • φ : global DTG phase (energy-phase driver)
        • ω : intrinsic angular frequency [rad/ms]
        • K : baseline coupling gain (0 ≤ K ≤ 1)
        • λ : Ca²⁺ sensitivity (dimensionless)
        • S : normalized Ca²⁺ activity (0~1)

    Effective coupling:
        K_eff = K·(1 + λ·S)

    Integration (discrete time):
        θ_{t+Δt} = (θ_t + (ω + K_eff·sin(φ−θ_t))·Δt) mod 2π

    Biophysical interpretation:
        - High Ca²⁺ (S↑) → coupling gain ↑ → stronger phase-lock to φ
        - Low Ca²⁺ (S↓) → coupling gain ↓ → weak synchronization
    """

    def __init__(self, omega: float = 1.0, K: float = 0.05, lambda_ca: float = 1.0):
        """
        Parameters
        ----------
        omega : float
            Intrinsic oscillation frequency [rad/ms].
        K : float
            Baseline coupling strength (0 ≤ K ≤ 1).
        lambda_ca : float
            Calcium modulation coefficient λ (coupling sensitivity).
        """
        self.theta = 0.0          # Current synaptic phase θ [rad]
        self.omega = omega        # Intrinsic angular velocity
        self.K = K                # Base coupling gain
        self.lambda_ca = lambda_ca  # Ca²⁺-dependent modulation factor

    # -------------------------------------------------------------
    # Step Integration
    # -------------------------------------------------------------
    def step(self, dt: float, phi: float, S: float):
        r"""
        Integrate phase θ over dt [ms].

        Equation:
            dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)
            θ(t+Δt) = (θ + dθ·Δt) mod 2π

        Parameters
        ----------
        dt : float
            Integration step [ms].
        phi : float
            DTG (driver) phase [rad].
        S : float
            Normalized calcium activity (0~1).

        Returns
        -------
        tuple(float, float)
            (θ, Δθ) → (synaptic phase, phase difference φ−θ)
        """

        # 1) 유효 커플링 게인 계산 (Ca²⁺ 영향 반영)
        #    K_eff = K * (1 + λ·S)
        K_eff = self.K * (1.0 + self.lambda_ca * S)

        # 2) 위상 변화율 계산
        #    dθ/dt = ω + K_eff·sin(φ−θ)
        dtheta_dt = self.omega + K_eff * np.sin(phi - self.theta)

        # 3) 이산 적분 (Euler)
        #    θ ← θ + dθ·dt
        self.theta += dtheta_dt * dt

        # 4) 위상 wrap (0~2π)
        self.theta = self.theta % (2 * np.pi)

        # 5) 위상차 Δθ 계산
        delta_phi = (phi - self.theta)

        return self.theta, delta_phi

    # -------------------------------------------------------------
    # Optional helper: instantaneous coupling gain
    # -------------------------------------------------------------
    def coupling_gain(self, S: float) -> float:
        """현재 Ca²⁺ 값(S)에 따른 실시간 유효 커플링 게인 반환"""
        return self.K * (1.0 + self.lambda_ca * S)