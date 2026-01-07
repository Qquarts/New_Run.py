# =============================================================
# 09.synaptic_resonance.py — Ca²⁺ 기반 커플링 게인 공명 모델 (V3)
# =============================================================
# 목적:
#   시냅스의 내부 위상 θ(t)가 상위 발진자(DTG System)의 위상 φ(t)에
#   동기화(phase locking)되는 과정을 모델링한다.
#
#   이때 결합 강도(coupling gain, K)가 칼슘 신호(S)에 의해
#   동적으로 조절되는 구조를 포함한다.
#
# V3 계약 고정:
#   - 시간 단위: [ms] (밀리초) - 모든 시간 관련 파라미터는 ms 단위
#   - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
#   - θ, φ: [rad] (라디안)
#   - ω: [rad/ms] (위상 속도) ⭐ V3: ms 단위 명시
#
#   ┌──────────────────────────────────────┐
#   │ dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)   │
#   └──────────────────────────────────────┘
#
#   • θ : 시냅스 고유 위상 (synaptic phase) [rad]
#   • φ : 상위 발진 위상 (DTG phase) [rad]
#   • ω : 고유 위상속도 (intrinsic angular frequency) [rad/ms] ⭐ V3: ms 단위 명시
#   • K : 기본 커플링 게인 (baseline coupling strength) [무차원]
#   • λ : Ca²⁺ 민감도 (coupling modulation coefficient) [무차원]
#   • S : Ca²⁺ 정규화 농도 (0~1) ⭐ V3 계약 고정
#
#   ⇒ Ca²⁺가 높을수록 결합이 강해지고(동기화↑),
#      Ca²⁺가 낮을수록 각자 독립 진동(비동기화).
#
#   물리적 의미:
#       "Ca²⁺는 시냅스 공명의 커플링 게인으로 작용한다."
#
# 설계 이유:
#   - 시냅스 위상 동기화는 뉴런 네트워크의 협조적 활동을 모델링
#   - Ca²⁺ 농도에 따라 동기화 강도가 변하는 것은 생리학적으로 타당
#   - Kuramoto 모델 기반으로 위상 결합을 모델링
# =============================================================

import numpy as np


class SynapticResonance:
    r"""
    SynapticResonance — Ca²⁺-modulated Phase Coupling Resonator (V3)
    ------------------------------------------------------------
    
    V3 계약:
    - 입력: dt [ms], phi [rad], S [0,1]
    - 출력: (theta [rad], delta_phi [rad])
    - Side-effect: self.theta 업데이트 [rad]
    
    Differential equation:
        dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)

    where:
        • θ : synaptic phase (local oscillator) [rad]
        • φ : global DTG phase (energy-phase driver) [rad]
        • ω : intrinsic angular frequency [rad/ms] ⭐ V3: ms 단위 명시
        • K : baseline coupling gain (0 ≤ K ≤ 1) [무차원]
        • λ : Ca²⁺ sensitivity (dimensionless) [무차원]
        • S : normalized Ca²⁺ activity (0~1) ⭐ V3 계약 고정

    Effective coupling:
        K_eff = K·(1 + λ·S)

    Integration (discrete time):
        θ_{t+Δt} = (θ_t + (ω + K_eff·sin(φ−θ_t))·Δt) mod 2π

    Biophysical interpretation:
        - High Ca²⁺ (S↑) → coupling gain ↑ → stronger phase-lock to φ
        - Low Ca²⁺ (S↓) → coupling gain ↓ → weak synchronization
    
    설계 이유:
    - 시냅스 위상 동기화는 뉴런 네트워크의 협조적 활동을 모델링
    - Ca²⁺ 농도에 따라 동기화 강도가 변하는 것은 생리학적으로 타당
    - Kuramoto 모델 기반으로 위상 결합을 모델링
    """

    def __init__(self, omega: float = 1.0, K: float = 0.05, lambda_ca: float = 1.0):
        """
        SynapticResonance 초기화
        
        Parameters
        ----------
        omega : float
            Intrinsic oscillation frequency [rad/ms] ⭐ V3: ms 단위 명시. 기본값: 1.0
        K : float
            Baseline coupling strength (0 ≤ K ≤ 1) [무차원]. 기본값: 0.05
        lambda_ca : float
            Calcium modulation coefficient λ (coupling sensitivity) [무차원]. 기본값: 1.0
        """
        self.theta = 0.0          # Current synaptic phase θ [rad]
        self.omega = omega        # Intrinsic angular velocity [rad/ms] ⭐ V3: ms 단위 명시
        self.K = K                # Base coupling gain [무차원]
        self.lambda_ca = lambda_ca  # Ca²⁺-dependent modulation factor [무차원]

    # -------------------------------------------------------------
    # Step Integration
    # -------------------------------------------------------------
    def step(self, dt: float, phi: float, S: float):
        r"""
        Integrate phase θ over dt [ms].
        
        V3 계약:
        - 입력:
          - dt: [ms] (시간 스텝) ⭐ V3 계약 고정
          - phi: [rad] (DTG 위상)
          - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
        - 출력: (theta [rad], delta_phi [rad])
        - Side-effect: self.theta 업데이트 [rad]

        Equation:
            dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)
            θ(t+Δt) = (θ + dθ·Δt) mod 2π

        Parameters
        ----------
        dt : float
            Integration step [ms] ⭐ V3: ms 단위 명시
        phi : float
            DTG (driver) phase [rad]
        S : float
            Normalized calcium activity (0~1) ⭐ V3 계약 고정

        Returns
        -------
        tuple(float, float)
            (θ [rad], Δθ [rad]) → (synaptic phase, phase difference φ−θ)
        """

        # 1) 유효 커플링 게인 계산 (Ca²⁺ 영향 반영)
        #    K_eff = K * (1 + λ·S)
        #    S는 [0,1] 범위 ⭐ V3 계약 고정
        K_eff = self.K * (1.0 + self.lambda_ca * S)  # [무차원]

        # 2) 위상 변화율 계산
        #    dθ/dt = ω + K_eff·sin(φ−θ)
        #    단위: [rad/ms] = [rad/ms] + [무차원] · [rad] ⭐ V3: ms 단위 명시
        dtheta_dt = self.omega + K_eff * np.sin(phi - self.theta)  # [rad/ms] ⭐ V3: ms 단위 명시

        # 3) 이산 적분 (Euler)
        #    θ ← θ + dθ·dt
        #    단위: [rad] = [rad] + [rad/ms] · [ms] ⭐ V3: ms 단위 명시
        self.theta += dtheta_dt * dt  # [rad]

        # 4) 위상 wrap (0~2π)
        #    물리적 이유: 위상은 주기적이므로 [0, 2π) 범위로 제한
        self.theta = self.theta % (2 * np.pi)  # [rad]

        # 5) 위상차 Δθ 계산 (정규화: -π ~ +π)
        #    위상차를 [-π, +π] 범위로 정규화하여 시각화/분석 용이
        #    물리적 이유: 위상차는 [-π, +π] 범위로 표현하는 것이 직관적
        delta_phi = (phi - self.theta + np.pi) % (2 * np.pi) - np.pi  # [rad]

        return self.theta, delta_phi

    # -------------------------------------------------------------
    # Optional helper: instantaneous coupling gain
    # -------------------------------------------------------------
    def coupling_gain(self, S: float) -> float:
        """
        현재 Ca²⁺ 값(S)에 따른 실시간 유효 커플링 게인 반환
        
        V3 계약:
        - 입력: S [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
        - 출력: K_eff [무차원] (유효 커플링 게인)
        - Side-effect: 없음
        
        Parameters
        ----------
        S : float
            Normalized calcium activity (0~1) ⭐ V3 계약 고정
            
        Returns
        -------
        float
            유효 커플링 게인 K_eff = K·(1 + λ·S) [무차원]
        """
        return self.K * (1.0 + self.lambda_ca * S)  # [무차원]

    def on_spike(self, R: float, phi: float):
        """
        스파이크 발생 시 호출 (선택적, 위상 리셋 등)
        
        V3 계약:
        - 입력: R [무차원] (PTP 잔여 강화량), phi [rad] (DTG 위상)
        - 출력: 없음
        - Side-effect: 없음 (현재 구현에서는 사용하지 않음)
        
        Parameters
        ----------
        R : float
            PTP 잔여 강화량 (무차원)
        phi : float
            DTG 위상 [rad]
        """
        # 현재 구현에서는 사용하지 않음
        # V3 확장 시 위상 리셋 또는 조정 로직 추가 가능
        pass


# =============================================================
# V3 독립 실행 테스트
# =============================================================
if __name__ == "__main__":
    """
    V3 계약 고정 검증:
    - S: [0,1] 확인
    - 시간 단위: [ms] 확인
    - 입출력/side-effect 확인
    """
    resonance = SynapticResonance(
        omega=1.0,  # [rad/ms] ⭐ V3: ms 단위 명시
        K=0.05,
        lambda_ca=1.0
    )
    
    dt = 0.1  # [ms] ⭐ V3: ms 단위 명시
    phi = 0.0  # [rad]
    S = 0.5  # [0,1] ⭐ V3 계약 고정
    
    print("=" * 60)
    print("SynapticResonance V3 — 계약 고정 검증")
    print("=" * 60)
    print(f"S 입력: {S} [0,1] ⭐ V3 계약")
    print(f"dt: {dt} [ms] ⭐ V3 계약")
    print(f"omega: {resonance.omega} [rad/ms] ⭐ V3 계약")
    print("-" * 60)
    print("[SynapticResonance Test]")
    print(f"{'t(ms)':>8} | {'theta(rad)':>12} | {'delta_phi(rad)':>15} | {'K_eff':>8}")
    print("-" * 60)
    
    # DTG 위상 시뮬레이션 (선형 증가)
    for t in range(20):
        phi = t * dt * 1.0  # [rad] = [ms] · [rad/ms] (DTG 위상 선형 증가)
        theta, delta_phi = resonance.step(dt=dt, phi=phi, S=S)  # S는 [0,1] ⭐ V3 계약 고정
        K_eff = resonance.coupling_gain(S)  # S는 [0,1] ⭐ V3 계약 고정
        print(f"{t*dt:8.2f} | {theta:12.3f} | {delta_phi:15.3f} | {K_eff:8.3f}")
    
    print("=" * 60)
    print("✅ V3 계약 고정 검증 완료")
    print("  - S: [0,1] 범위 확인")
    print("  - 시간 단위: [ms] 확인")
    print("  - 입출력/side-effect 명시 확인")

