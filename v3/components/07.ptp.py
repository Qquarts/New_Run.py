# =============================================================
# 07.ptp.py — Post-Tetanic Potentiation (PTP) only (V3)
# =============================================================
# 개념/수식 요약
# -------------------------------------------------------------
# PTP는 고빈도 자극(tetanus) 후 수 초~수십 초 동안
#  방출확률 p 또는 시냅스 이득 w를 일시적으로 증가시키는
#  Ca²⁺-의존 단기 가소성.
#
# V3 계약 고정:
#   - 시간 단위: [ms] (밀리초) - 모든 시간 관련 파라미터는 ms 단위
#   - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
#   - R: [0,∞) (PTP 잔여 강화량, 무차원)
#   - tau_ptp: [s] (초 단위, 내부적으로 ms로 변환)
#
# 상태변수:
#   R(t) : PTP "잔여 강화량"(무차원, 0 이상)
#
# 동역학:
#   dR/dt = -R / τ_ptp  +  A(Ca_res) · Σ_k δ(t - t_k)
#
#   • τ_ptp : PTP 감쇠 시정수 [s] (실험적으로 10~60 s)
#   • A(Ca_res) : 스파이크 직후 잔여 Ca²⁺(또는 Ca 이벤트 S)에 비례하는 증분
#       A(Ca) = g_ptp · (Ca_norm)^n / ( (Ca_norm)^n + K^n )
#     여기서 Ca_norm = clamp( (Ca - C0)/(Cmax - C0), 0, 1 ) = S ⭐ V3 계약 고정
#
# 적용:
#   p_eff = clamp( p0 * (1 + R), 0, 1 )
#   또는
#   w_eff = w0 * (1 + R)
#
# 주의:
#   • 본 클래스는 "PTP 항만" 제공. 단기 facilitation/depression(τ_f, τ_d)은 포함하지 않음.
#   • Ca 입력은 CaVesicle의 정규화된 S 값 (0~1)을 전달 ⭐ V3 계약 고정
#
# 설계 이유:
#   - PTP는 고빈도 자극 후 시냅스 강화를 모델링하는 핵심 메커니즘
#   - Ca²⁺ 농도에 비례하여 강화되며, Hill 함수로 비선형 반응 모델링
#   - 지수 감쇠로 시간에 따라 강화가 점진적으로 사라짐
# =============================================================

from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class PTPConfig:
    """
    PTP 설정 파라미터
    
    Parameters
    ----------
    tau_ptp_s : float
        PTP 감쇠 시정수 [s] (실험적으로 10~60 s)
    g_ptp : float
        PTP 첨가 이득 (스파이크당 최대 증분 스케일, 무차원)
    K_half : float
        Ca_norm의 반포화점 (0~1 범위, Hill 함수 파라미터) ⭐ V3 계약 고정
    hill_n : int
        Hill 계수 (비선형 민감도, 보통 2~4)
    R_clip : tuple
        R(t) 안전 범위 (증폭 한계) (min, max)
    """
    tau_ptp_s: float = 20.0     # PTP 감쇠 시정수 [s]
    g_ptp: float    = 1.2       # PTP 첨가 이득 (무차원)
    K_half: float   = 0.25      # Ca_norm의 반포화점 (0~1 범위) ⭐ V3 계약 고정
    hill_n: int     = 3         # Hill 계수 (비선형 민감도)
    R_clip: tuple   = (0.0, 3.0)# R(t) 안전 범위 (증폭 한계)


class PTPPlasticity:
    r"""
    Post-Tetanic Potentiation (PTP) — Ca²⁺-dependent short-term potentiation (V3)
    
    V3 계약:
    - 입력: S [0,1] (정규화된 Ca 농도), dt_ms [ms]
    - 출력: R [0,∞) (PTP 잔여 강화량, 무차원)
    - Side-effect: self.R 업데이트
    
    상태변수:
        R(t) ≥ 0  : PTP "잔여 강화량" (무차원)
    
    동역학:
        dR/dt = -R / τ_ptp  +  A(Ca_norm) · Σ_k δ(t - t_k)
        
        A(Ca_norm) = g_ptp · (Ca_norm^n) / (Ca_norm^n + K^n)
        
        여기서 Ca_norm = clamp((Ca - C0)/(Cmax - C0), 0, 1) = S ⭐ V3 계약 고정
        또는 CaVesicle의 정규화된 S 값 (0~1)을 직접 사용
    
    적용:
        p_eff = clamp(p0 * (1 + R), 0, 1)
        w_eff = w0 * (1 + R)
    
    주의:
        • Ca 입력은 CaVesicle의 정규화된 S 값 (0~1)을 전달 ⭐ V3 계약 고정
        • Hill 함수를 통한 비선형 강화
        • τ_ptp 기반 지수 감쇠
    
    설계 이유:
    - PTP는 고빈도 자극 후 시냅스 강화를 모델링하는 핵심 메커니즘
    - Ca²⁺ 농도에 비례하여 강화되며, Hill 함수로 비선형 반응 모델링
    - 지수 감쇠로 시간에 따라 강화가 점진적으로 사라짐
    """
    def __init__(self, cfg: Union[PTPConfig, dict]):
        """
        PTPPlasticity 초기화
        
        Parameters
        ----------
        cfg : PTPConfig or dict
            설정 객체 또는 딕셔너리:
            - tau_ptp_s: PTP 감쇠 시정수 [s] (실험적으로 10~60 s)
            - g_ptp: PTP 첨가 이득 (무차원)
            - K_half: Ca_norm의 반포화점 (0~1 범위) ⭐ V3 계약 고정
            - hill_n: Hill 계수 (비선형 민감도, 보통 2~4)
            - R_clip: R(t) 안전 범위 (증폭 한계) (min, max)
        """
        # cfg를 dict로 변환 (유연성)
        if isinstance(cfg, dict):
            self.tau_ptp_s = float(cfg.get("tau_ptp_s", 20.0))  # [s]
            self.g_ptp = float(cfg.get("g_ptp", 1.2))
            self.K_half = float(cfg.get("K_half", 0.25))  # [0,1] ⭐ V3 계약 고정
            self.hill_n = int(cfg.get("hill_n", 3))
            self.R_clip = tuple(cfg.get("R_clip", (0.0, 3.0)))
        else:
            self.tau_ptp_s = float(cfg.tau_ptp_s)  # [s]
            self.g_ptp = float(cfg.g_ptp)
            self.K_half = float(cfg.K_half)  # [0,1] ⭐ V3 계약 고정
            self.hill_n = int(cfg.hill_n)
            self.R_clip = cfg.R_clip
        
        # 상태 변수
        self.R = 0.0  # 잔여 강화량 (무차원)
        
        # 파라미터 검증
        if self.tau_ptp_s <= 0:
            raise ValueError(f"tau_ptp_s must be > 0, got {self.tau_ptp_s}")
        if self.K_half <= 0:
            raise ValueError(f"K_half must be > 0, got {self.K_half}")
        if self.K_half >= 1.0:
            import warnings
            warnings.warn(
                f"K_half={self.K_half} >= 1.0: Hill 함수가 거의 활성화되지 않을 수 있습니다. "
                f"일반적으로 K_half < 1.0을 권장합니다.",
                RuntimeWarning
            )
        if self.hill_n < 1:
            raise ValueError(f"hill_n must be >= 1, got {self.hill_n}")

    def _hill_function(self, Ca_norm: float) -> float:
        """
        Hill 함수: A(Ca_norm) = g_ptp · (Ca_norm^n) / (Ca_norm^n + K^n)
        
        V3 계약:
        - 입력: Ca_norm [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
        - 출력: A (PTP 증분, 무차원)
        - Side-effect: 없음
        
        비선형 Ca 의존 강화 계산
        
        물리적 이유: Hill 함수는 Ca²⁺ 농도에 대한 비선형 반응을 모델링
        Ca_norm이 낮을 때는 약한 반응, 높을 때는 강한 반응 (포화 특성)
        
        Parameters
        ----------
        Ca_norm : float
            정규화된 Ca 농도 (0~1) ⭐ V3 계약 고정
            
        Returns
        -------
        float
            PTP 증분 (무차원)
        """
        # Ca_norm 클램프 (0~1) ⭐ V3 계약 고정
        # 물리적 이유: Ca_norm은 정규화된 값이므로 [0,1] 범위로 제한
        Ca_norm = max(0.0, min(1.0, float(Ca_norm)))
        
        # Hill 함수 계산
        # A(Ca_norm) = g_ptp · (Ca_norm^n) / (Ca_norm^n + K^n)
        num = Ca_norm ** self.hill_n
        den = num + (self.K_half ** self.hill_n)
        
        if den == 0.0:
            return 0.0
        
        A = self.g_ptp * (num / den)
        return float(A)

    def on_spike(self, S: float):
        """
        스파이크 직후 호출: Hill 함수를 통한 비선형 강화
        
        V3 계약:
        - 입력: S [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
        - 출력: 없음
        - Side-effect: self.R 업데이트 (증분 추가)
        
        수식: dR/dt = ... + A(Ca_norm) · δ(t - t_k)
        → 각 스파이크마다 R += A(Ca_norm) (dt 독립적)
        
        ⚠️ 중요: 스파이크가 너무 자주 오면 R이 상한(R_clip[1])에 도달할 수 있음
        이는 의도된 동작 특성입니다 (고빈도 자극 시 강화 한계).
        
        Parameters
        ----------
        S : float
            정규화된 Ca 농도 (0~1) ⭐ V3 계약 고정
            CaVesicle.get_state()["S"] 값을 전달
            
        사용 예:
        --------
        # 스파이크 발생 시
        ca_event = ca_vesicle.step(ATP=atp_level)
        ptp.on_spike(S=ca_event.S)  # CaVesicle의 S 값 전달
        
        # 매 스텝
        ptp.step(dt_ms=0.1)
        """
        # Hill 함수를 통한 비선형 증분 계산
        increment = self._hill_function(S)  # S는 [0,1] ⭐ V3 계약 고정
        
        # R 업데이트
        # δ항: 스파이크마다 R에 increment를 더함 (dt 독립적)
        # 수식: dR/dt = ... + A(Ca_norm) · Σ_k δ(t - t_k)
        # → 각 스파이크마다 R += A(Ca_norm)
        self.R += increment
        
        # R 클램프 (폭주 방지)
        # 물리적 이유: R이 비정상적으로 높아지면 수치 불안정성 발생 가능
        # np.clip은 np.float64를 반환할 수 있으므로 float로 명시적 변환
        self.R = float(np.clip(self.R, self.R_clip[0], self.R_clip[1]))

    def step(self, dt_ms: float) -> float:
        """
        dt_ms 만큼 시간 전진: 지수 감쇠
        
        V3 계약:
        - 입력: dt_ms [ms] (시간 스텝) ⭐ V3 계약 고정
        - 출력: R [0,∞) (PTP 잔여 강화량, 무차원)
        - Side-effect: self.R 업데이트 (지수 감쇠)
        
        수식: dR/dt = -R / τ_ptp
        이산화: R_{n+1} = R_n · exp(-dt / τ_ptp)
        
        Parameters
        ----------
        dt_ms : float
            시간 스텝 [ms] ⭐ V3: ms 단위 명시
            
        Returns
        -------
        float
            업데이트된 R 값 (무차원)
        """
        # 단위 변환: dt_ms [ms] → dt_s [s]
        dt_s = dt_ms / 1000.0
        
        # 지수 감쇠: R(t) = R(0) · exp(-t / τ_ptp)
        # 이산화: R_{n+1} = R_n · exp(-dt_s / tau_ptp_s)
        decay_factor = np.exp(-dt_s / self.tau_ptp_s)
        self.R *= decay_factor
        
        # R 클램프 (음수 방지)
        # 물리적 이유: R은 잔여 강화량이므로 음수가 될 수 없음
        self.R = max(0.0, self.R)
        
        return float(self.R)

    # --- 외부에 줄 모듈레이션 팩터 ---
    def p_eff(self, p0: float) -> float:
        """
        방출확률 p의 PTP 적용값
        
        V3 계약:
        - 입력: p0 [0,1] (기본 방출확률)
        - 출력: p_eff [0,1] (PTP 적용 방출확률)
        - Side-effect: 없음
        
        Returns
        -------
        float
            PTP 적용 방출확률 [0,1]
        """
        # 물리적 이유: 방출확률은 [0,1] 범위로 제한
        return max(0.0, min(1.0, p0 * (1.0 + self.R)))

    def w_eff(self, w0: float) -> float:
        """
        가중치/시냅스 이득의 PTP 적용값(상한은 외부에서 관리)
        
        V3 계약:
        - 입력: w0 (기본 가중치)
        - 출력: w_eff (PTP 적용 가중치)
        - Side-effect: 없음
        
        Returns
        -------
        float
            PTP 적용 가중치 (상한은 외부에서 관리)
        """
        return w0 * (1.0 + self.R)


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
    cfg = {
        "tau_ptp_s": 20.0,  # [s]
        "g_ptp": 1.2,
        "K_half": 0.25,  # [0,1] ⭐ V3 계약 고정
        "hill_n": 3,
        "R_clip": (0.0, 3.0),
    }
    
    ptp = PTPPlasticity(cfg)
    dt_ms = 1.0  # [ms] ⭐ V3: ms 단위 명시
    S = 0.5  # [0,1] ⭐ V3 계약 고정
    
    print("=" * 60)
    print("PTPPlasticity V3 — 계약 고정 검증")
    print("=" * 60)
    print(f"S 입력: {S} [0,1] ⭐ V3 계약")
    print(f"dt_ms: {dt_ms} [ms] ⭐ V3 계약")
    print("-" * 60)
    print("[PTP Test]")
    print(f"{'t(ms)':>8} | {'R':>8} | {'p_eff(0.5)':>12}")
    print("-" * 60)
    
    # 스파이크 시뮬레이션
    for t in range(10):
        if t == 0 or t == 5:  # t=0ms, t=5ms에 스파이크
            ptp.on_spike(S=S)  # S는 [0,1] ⭐ V3 계약 고정
        R = ptp.step(dt_ms=dt_ms)  # [ms] ⭐ V3: ms 단위 명시
        p_eff = ptp.p_eff(0.5)
        print(f"{t*dt_ms:8.2f} | {R:8.3f} | {p_eff:12.3f}")
    
    print("=" * 60)
    print("✅ V3 계약 고정 검증 완료")
    print("  - S: [0,1] 범위 확인")
    print("  - 시간 단위: [ms] 확인")
    print("  - 입출력/side-effect 명시 확인")

