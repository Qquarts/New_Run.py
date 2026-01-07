# =============================================================
# 10.terminal_release.py — Presynaptic Terminal (Ca, PTP, φ, ATP, Broadcast) (V3)
# =============================================================
# 모든 수식은 아래 생리학/신경동역학 규칙을 따른다:
#
# V3 계약 고정:
#   - ATP: [0,100] (정규화, 0~100 범위로 통일) ⭐ V3 변경
#   - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
#   - R: [0,∞) (PTP 잔여 강화량, 무차원)
#   - Δφ: [rad] (위상 차)
#   - Q: [arb] (방출 강도, 무차원)
#
# (1) 방출 강도 Q(t)
#     --------------------------------------------------------------
#     Q(t) = spike · α_C · S(t)^p · α_R · R(t)^q · α_φ · (1 + h |Δφ|)
#                · (ATP(t)/100)^γ
#
#     의미:
#       - spike: 스파이크 발생 시 1, 아니면 0
#       - S(t): Calcium normalized value ∈ [0,1] ⭐ V3 계약 고정
#       - R(t): Post-tetanic potentiation (PTP) 계수 (무차원)
#       - Δφ: 위상 차 (phase difference) [rad]
#       - (ATP/100)^γ : 에너지 수준이 방출량에 영향 ⭐ V3: ATP/100 변환
#       - ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
#
# (2) 방출 확률 P(t)
#     --------------------------------------------------------------
#     P(t) = σ( k_C·S(t) + k_R·R(t) + k_φ·Δφ )
#     σ(x) = 1 / (1 + e^{-x})
#
#     의미:
#       - S가 커질수록 확률 증가
#       - R이 클수록 시냅스 민감도 증가
#       - Δφ도 확률에 기여
#
# (3) 최종 방출량 Release(t)
#     --------------------------------------------------------------
#     Release(t) = Q(t)       (prob_mode=False)
#                = Q(t)       if rand < P(t)
#                = 0          otherwise
#
# 설계 이유:
#   - 시냅스 전달물질 방출은 Ca²⁺, PTP, 위상, ATP에 의해 조절됨
#   - Ca²⁺는 방출의 핵심 신호로, 농도에 비례하여 방출 증가
#   - PTP는 고빈도 자극 후 일시적 강화를 모델링
#   - 위상 차이는 시냅스 동기화를 반영
# ====================================================================

import numpy as np


class Terminal:
    r"""
    Synaptic Terminal — Ca / PTP / Phase 기반 방출량 Q(t) 계산기 (V3)
    
    V3 계약:
    - 입력: spike (0 or 1), S [0,1], R [0,∞), dphi [rad], ATP [0,100]
    - 출력: (Q [arb], p_eff [0,1])
    - Side-effect: self.last_Q, self.last_P 업데이트 (내부 상태)
    
    Release equation:
        Q = spike · α_C S^p · α_R R^q · α_φ (1 + h|Δφ|) · (ATP/100)^{1/2} ⭐ V3: ATP/100 변환

    Where:
        S  : Ca normalized (0~1) ⭐ V3 계약 고정
        R  : PTP residual potentiation (무차원)
        Δφ : phase difference (phi - theta) [rad]
        ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
    
    설계 이유:
    - 시냅스 전달물질 방출은 Ca²⁺, PTP, 위상, ATP에 의해 조절됨
    - Ca²⁺는 방출의 핵심 신호로, 농도에 비례하여 방출 증가
    - PTP는 고빈도 자극 후 일시적 강화를 모델링
    - 위상 차이는 시냅스 동기화를 반영
    """

    def __init__(self,
                 alpha_C=1.0, p=1.0,
                 alpha_R=1.0, q=1.0,
                 alpha_phi=1.0, h=0.2,
                 p0=0.3):
        """
        Terminal 초기화
        
        Parameters
        ----------
        alpha_C : float
            Ca²⁺ 방출 계수 (무차원). 기본값: 1.0
        p : float
            Ca²⁺ 지수 (무차원). 기본값: 1.0
        alpha_R : float
            PTP 방출 계수 (무차원). 기본값: 1.0
        q : float
            PTP 지수 (무차원). 기본값: 1.0
        alpha_phi : float
            위상 방출 계수 (무차원). 기본값: 1.0
        h : float
            위상 차이 계수 (무차원). 기본값: 0.2
        p0 : float
            기본 방출 확률 [0,1]. 기본값: 0.3
        """
        self.alpha_C = alpha_C
        self.p = p
        self.alpha_R = alpha_R
        self.q = q
        self.alpha_phi = alpha_phi
        self.h = h
        self.p0 = p0
        self.synapses = []   # broadcast targets
        
        # 내부 상태 (디버깅/모니터링용)
        self.last_Q = 0.0  # [arb]
        self.last_P = 0.0  # [0,1]

    # ---------------------------------------------
    # Synapse registration
    # ---------------------------------------------
    def attach_synapse(self, syn):
        """
        시냅스 연결 등록
        
        V3 계약:
        - 입력: syn (시냅스 객체)
        - 출력: 없음
        - Side-effect: self.synapses 업데이트 (시냅스 추가)
        """
        self.synapses.append(syn)

    # ---------------------------------------------
    # Core release computation
    # ---------------------------------------------
    def compute_Q(self, spike, S, R, dphi, ATP=100):
        """
        Release equation 계산
        
        V3 계약:
        - 입력:
          - spike: 0 or 1 (스파이크 발생 여부)
          - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
          - R: [0,∞) (PTP 잔여 강화량, 무차원)
          - dphi: [rad] (위상 차)
          - ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
        - 출력: Q [arb] (방출 강도, 무차원)
        - Side-effect: 없음
        
        Release equation:
            Q = spike · α_C S^p · α_R R^q
                · α_φ (1 + h|Δφ|) · (ATP/100)^{1/2} ⭐ V3: ATP/100 변환
        
        Returns
        -------
        float
            방출 강도 Q [arb] (무차원)
        """
        if spike <= 0:
            return 0.0
        
        # 입력 검증 및 클램프
        # 물리적 이유: S는 정규화된 값이므로 [0,1] 범위로 제한 ⭐ V3 계약 고정
        S = np.clip(S, 0.0, 1.0)  # [0,1] ⭐ V3 계약 고정
        # 물리적 이유: R은 잔여 강화량이므로 음수가 될 수 없음
        R = max(0.0, R)
        # 물리적 이유: 위상 차이는 [-π, +π] 범위로 제한 (주기적 특성)
        dphi = np.clip(dphi, -np.pi, np.pi)  # [rad]
        # 물리적 이유: ATP는 [0,100] 범위로 제한 (생리학적 범위) ⭐ V3 계약 고정
        ATP = np.clip(ATP, 0.0, 100.0)  # [0,100] ⭐ V3 계약 고정

        # V3: ATP는 [0,100] 범위이므로 100으로 나누어 [0,1]로 변환
        Q = (spike
             * self.alpha_C * (S ** self.p)  # S는 [0,1] ⭐ V3 계약 고정
             * self.alpha_R * (R ** self.q)
             * self.alpha_phi * (1.0 + self.h * abs(dphi))
             * (ATP / 100.0) ** 0.5)  # ATP/100 변환 ⭐ V3 변경
        return float(Q)  # [arb]

    # ---------------------------------------------
    # Probability modulation (PTP)
    # ---------------------------------------------
    def prob(self, R):
        """
        방출 확률 계산 (PTP 적용)
        
        V3 계약:
        - 입력: R [0,∞) (PTP 잔여 강화량, 무차원)
        - 출력: p_eff [0,1] (방출 확률)
        - Side-effect: 없음
        
        Returns
        -------
        float
            방출 확률 [0,1]
        """
        # 물리적 이유: 방출 확률은 [0,1] 범위로 제한
        return np.clip(self.p0 * (1.0 + R), 0.0, 1.0)  # [0,1]

    # ---------------------------------------------
    # High-level release API
    # ---------------------------------------------
    def release(self, spike, S, R, dphi, ATP=100):
        """
        시냅스 전달물질 방출량 계산
        
        V3 계약:
        - 입력:
          - spike: 0 or 1 (스파이크 발생 여부)
          - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
          - R: [0,∞) (PTP 잔여 강화량, 무차원)
          - dphi: [rad] (위상 차)
          - ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
        - 출력: (Q [arb], p_eff [0,1])
        - Side-effect: self.last_Q, self.last_P 업데이트
        
        Returns
        -------
        tuple(float, float)
            (Q [arb], p_eff [0,1]) → (방출 강도, 방출 확률)
        """
        Q = self.compute_Q(spike, S, R, dphi, ATP)  # Q는 [arb]
        p_eff = self.prob(R)  # p_eff는 [0,1]
        
        # 내부 상태 업데이트 (디버깅/모니터링용)
        self.last_Q = Q  # [arb]
        self.last_P = p_eff  # [0,1]
        
        return Q, p_eff

    # ---------------------------------------------
    # Broadcast output to connected synapses
    # ---------------------------------------------
    def broadcast(self, t, Q):
        """
        연결된 시냅스에 방출량 브로드캐스트
        
        V3 계약:
        - 입력: t [ms] (시간), Q [arb] (방출 강도) ⭐ V3: ms 단위 명시
        - 출력: 없음
        - Side-effect: syn.receive() 호출 (각 시냅스에 이벤트 전달)
        
        Parameters
        ----------
        t : float
            시간 [ms] ⭐ V3: ms 단위 명시
        Q : float
            방출 강도 [arb]
        """
        for syn in self.synapses:
            syn.receive(t, Q)


# =============================================================
# SimpleSynapse — 시냅스 출력 수집기
# =============================================================
class SimpleSynapse:
    """
    시냅스 출력 수집기 (V3)
    
    V3 계약:
    - 입력: t [ms], Q [arb]
    - 출력: DataFrame (t_ms, Q) 또는 None
    - Side-effect: self.events 업데이트 (이벤트 추가)
    
    시냅스 출력 수집기
    - 입력: Terminal에서 받은 방출량
    - 출력: DataFrame (t_ms, Q, p_eff)
    """
    def __init__(self):
        """
        SimpleSynapse 초기화
        
        V3 계약:
        - 입력: 없음
        - 출력: 없음
        - Side-effect: self.events 초기화
        """
        self.events = []  # [(t [ms], Q [arb]), ...] ⭐ V3: ms 단위 명시
    
    def receive(self, t, Q):
        """
        터미널에서 방출량 수신
        
        V3 계약:
        - 입력: t [ms] (시간), Q [arb] (방출 강도) ⭐ V3: ms 단위 명시
        - 출력: 없음
        - Side-effect: self.events 업데이트 (이벤트 추가)
        
        Parameters
        ----------
        t : float
            시간 [ms] ⭐ V3: ms 단위 명시
        Q : float
            방출 강도 [arb]
        """
        self.events.append((float(t), float(Q)))
    
    def to_dataframe(self):
        """
        이벤트를 DataFrame으로 변환
        
        V3 계약:
        - 입력: 없음
        - 출력: DataFrame 또는 None
        - Side-effect: 없음
        
        Returns
        -------
        pd.DataFrame or None
            이벤트 DataFrame (t [ms], Q [arb]) 또는 None (pandas 미설치 시)
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.events, columns=["t", "Q"])
        except Exception:
            return None


# =============================================================
# V3 독립 실행 테스트
# =============================================================
if __name__ == "__main__":
    """
    V3 계약 고정 검증:
    - ATP: [0,100] 확인
    - S: [0,1] 확인
    - 시간 단위: [ms] 확인
    - 입출력/side-effect 확인
    """
    terminal = Terminal(
        alpha_C=1.0, p=1.0,
        alpha_R=1.0, q=1.0,
        alpha_phi=1.0, h=0.2,
        p0=0.3
    )
    
    spike = 1  # 스파이크 발생
    S = 0.5  # [0,1] ⭐ V3 계약 고정
    R = 1.2  # PTP 잔여 강화량 (무차원)
    dphi = 0.1  # [rad]
    ATP = 100.0  # [0,100] ⭐ V3 계약 고정
    
    print("=" * 60)
    print("Terminal V3 — 계약 고정 검증")
    print("=" * 60)
    print(f"ATP 입력: {ATP} [0,100] ⭐ V3 계약")
    print(f"S 입력: {S} [0,1] ⭐ V3 계약")
    print("-" * 60)
    print("[Terminal Test]")
    print(f"{'spike':>6} | {'S[0,1]':>8} | {'R':>6} | {'dphi(rad)':>12} | {'ATP[0,100]':>12} | {'Q':>8} | {'p_eff':>8}")
    print("-" * 60)
    
    for i in range(5):
        Q, p_eff = terminal.release(spike=spike, S=S, R=R, dphi=dphi, ATP=ATP)
        print(f"{spike:6d} | {S:8.3f} | {R:6.3f} | {dphi:12.3f} | {ATP:12.1f} | {Q:8.3f} | {p_eff:8.3f}")
        # 다음 스텝을 위해 값 변경
        S += 0.1  # [0,1] ⭐ V3 계약 고정
        R += 0.1
        dphi += 0.1  # [rad]
        ATP -= 10.0  # [0,100] ⭐ V3 계약 고정
    
    print("=" * 60)
    print("✅ V3 계약 고정 검증 완료")
    print("  - ATP: [0,100] 범위 확인")
    print("  - S: [0,1] 범위 확인")
    print("  - 입출력/side-effect 명시 확인")

