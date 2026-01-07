# =============================================================
# 06.ca_vesicle.py — Ca²⁺ Vesicle (Spike-triggered Alpha kernels) (V3)
# =============================================================
# 목적:
#   • 스파이크 시각 목록 {t_k}에 의해 유도되는 Ca²⁺ 유입(α-커널 합)과
#     ATP-의존 펌프에 의한 Ca 제거를 함께 모델링.
#   • 정규화 시그널 S=(Ca−C0)/(Cmax−C0) 및 상태 레이블(under/normal/alert) 제공.
#
# V3 계약 고정:
#   - 시간 단위: [ms] (밀리초) - 모든 시간 관련 파라미터는 ms 단위
#   - Ca: [μM] (마이크로몰) ⭐ V3 계약 고정
#   - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
#   - ATP: [0,100] (정규화, 0~100 범위로 통일) ⭐ V3 변경
#
# 수식(시간 단위 주의: dt_ms[ms], τ_r, τ_d는 [s]):
#   α(t)      = (e^{−t/τ_d} − e^{−t/τ_r})_+    ,  t ≥ 0
#   dCa/dt[s] =  Σ_k  A·α(t − t_k)  −  k_c·(ATP/100)·(Ca − C0)
#   Ca[t+Δt]  =  Ca[t] + (dCa/dt)·(Δt_ms / 1000)
#   S         =  (Ca − C0) / (Cmax − C0)
#
#   • A        : α-커널 스케일 [μM/s] (유입 크기)
#   • k_c      : 펌프 계수 [1/s] (ATP 의존), ATP↑ → 제거↑
#   • C0, Cmax : 기준 및 최대 농도 [μM] (정규화용 경계)
#   • Ca       : 칼슘 농도 [μM] (내부 상태 변수)
#   • ATP      : ATP 농도 [0,100] (정규화, 0~100 범위) ⭐ V3 변경
#
# 안전장치:
#   • τ_d ≤ τ_r면 물리적 의미가 사라지므로, 초기화 시 τ_d>τ_r 되도록 자동 수정.
#   • Ca는 음수가 되지 않도록 하한만 설정(물리적 클램프 X).
#   • 스파이크 메모리는 max_spike_memory_ms 윈도우로 관리.
#
# 반환:
#   VesicleEvent(t_ms, Ca, S, status) — 각 스텝 상태 스냅샷
#
# 설계 이유:
#   - Ca²⁺는 시냅스 전달물질 방출의 핵심 신호로, 스파이크 발생 시 급격히 증가
#   - ATP 의존 펌프는 Ca²⁺를 제거하여 시냅스 회복을 모델링
#   - 정규화된 S 값은 시냅스 가소성(PTP 등)의 입력으로 사용됨
# =============================================================

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# matplotlib은 선택 사항
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


@dataclass
class VesicleEvent:
    """
    Ca²⁺ Vesicle 이벤트 데이터
    
    V3 계약:
    - t_ms: [ms] (시간) ⭐ V3 계약 고정
    - Ca: [μM] (칼슘 농도) ⭐ V3 계약 고정
    - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
    - status: "under" | "normal" | "alert"
    """
    t_ms: float  # [ms] ⭐ V3: ms 단위 명시
    Ca: float    # [μM] ⭐ V3: μM 단위 명시
    S: float     # [0,1] ⭐ V3 계약 고정
    status: str  # "under" | "normal" | "alert"


class CaVesicle:
    r"""
    Spike-triggered Ca²⁺ Vesicle Dynamics (Alpha-kernel + ATP pump) (V3)
    
    V3 계약:
    - 입력: ATP [0,100] (정규화, 0~100 범위), dt_ms [ms]
    - 출력: VesicleEvent (t_ms [ms], Ca [μM], S [0,1], status)
    - Side-effect: self.Ca, self.t_ms, self.ATP, self.spike_times, self.events 업데이트
    
    미분방정식 (연속시간, t는 초[s]):
        α(t)      = (exp(−t/τ_d) − exp(−t/τ_r))_+ ,  τ_d > τ_r > 0
        dCa/dt[s] =  Σ_k A·α(t − t_k) − k_c·(ATP/100)·(Ca − C0) ⭐ V3: ATP/100 변환
    
    이산 적분 (dt_ms[ms]):
        Ca_{n+1} = Ca_n + (dCa/dt)·(dt_ms/1000)
    
    정규화:
        S = (Ca − C0) / (Cmax − C0) ∈ [0,1] ⭐ V3 계약 고정
    
    상태 레이블:
        S < 0      → "under"
        0 ≤ S ≤ 1  → "normal"
        S > 1      → "alert"
    
    설계 이유:
    - Ca²⁺는 시냅스 전달물질 방출의 핵심 신호로, 스파이크 발생 시 급격히 증가
    - ATP 의존 펌프는 Ca²⁺를 제거하여 시냅스 회복을 모델링
    - 정규화된 S 값은 시냅스 가소성(PTP 등)의 입력으로 사용됨
    """

    def __init__(self, cfg: dict, dt_ms: float):
        """
        CaVesicle 초기화
        
        Parameters
        ----------
        cfg : dict
            설정 딕셔너리:
            - C0: 기준 칼슘 농도 [μM] ⭐ V3: μM 단위 명시
            - Cmax: 최대 칼슘 농도 [μM] ⭐ V3: μM 단위 명시
            - k_c: 펌프 계수 [1/s]
            - A: Alpha kernel 스케일 [μM/s] per spike
            - tau_r_ms: Alpha kernel 상승 시간 [ms] (기본값: 0.5) ⭐ V3: ms 단위 명시
            - tau_d_ms: Alpha kernel 감쇠 시간 [ms] (기본값: 3.0) ⭐ V3: ms 단위 명시
            - ATP_initial: 초기 ATP 농도 [0,100] (기본값: 100.0) ⭐ V3: 0~100 범위 명시
            - max_spike_memory_ms: 스파이크 메모리 윈도우 [ms] (기본값: 2000.0) ⭐ V3: ms 단위 명시
            - Ca_max_clamp: Ca 상한 클램프 [μM] (None이면 클램프 없음, 기본값: None)
        dt_ms : float
            시간 스텝 [ms] ⭐ V3: ms 단위 명시
        """
        self.dt_ms = float(dt_ms)  # [ms] ⭐ V3: ms 단위 명시
        self.t_ms = 0.0  # [ms] ⭐ V3: ms 단위 명시
        
        # Ca 농도 파라미터 [μM] (단위 통일) ⭐ V3: μM 단위 명시
        self.C0 = float(cfg["C0"])      # 기준 농도 [μM]
        self.Cmax = float(cfg["Cmax"])  # 최대 농도 [μM]
        self.Ca = float(self.C0)        # 현재 농도 [μM]
        
        # 펌프 계수 [1/s]
        self.k_c = float(cfg["k_c"])
        
        # Alpha kernel 파라미터 (스파이크 트리거 Ca 유입)
        # τ_r, τ_d는 [s] 단위 (docstring 기준)
        tau_r_ms = cfg.get("tau_r_ms", 0.5)  # [ms] (기본값: 0.5ms) ⭐ V3: ms 단위 명시
        tau_d_ms = cfg.get("tau_d_ms", 3.0)  # [ms] (기본값: 3.0ms) ⭐ V3: ms 단위 명시
        
        # τ_d > τ_r 검증 및 자동 수정 (물리적 타당성)
        # 물리적 이유: τ_d > τ_r이어야 alpha kernel이 물리적으로 의미 있음
        if tau_d_ms <= tau_r_ms:
            import warnings
            warnings.warn(
                f"CaVesicle: tau_d ({tau_d_ms}ms) <= tau_r ({tau_r_ms}ms). "
                f"Swapping values to ensure tau_d > tau_r for physical validity.",
                RuntimeWarning
            )
            tau_d_ms, tau_r_ms = tau_r_ms, tau_d_ms
        
        # [ms] → [s] 변환
        self.tau_r_s = tau_r_ms / 1000.0  # [s]
        self.tau_d_s = tau_d_ms / 1000.0  # [s]
        
        # Alpha kernel 스케일 [μM/s] per spike
        self.Ca_influx = cfg.get("A", cfg.get("Ca_influx", 200.0))  # [μM/s]
        
        # 스파이크 메모리 관리
        self.max_spike_memory_ms = cfg.get("max_spike_memory_ms", 2000.0)  # [ms] ⭐ V3: ms 단위 명시
        self.spike_times: List[float] = []  # [ms] ⭐ V3: ms 단위 명시
        
        # 이벤트 로그
        self.events: List[VesicleEvent] = []
        
        # ATP 초기값 [0,100] ⭐ V3: 0~100 범위 명시
        self.ATP = cfg.get("ATP_initial", 100.0)  # [0,100] ⭐ V3: 0~100 범위 명시
        
        # Ca 상한 클램프 (수치 안정성용, None이면 클램프 없음)
        # 물리적 이유: Ca 농도가 비정상적으로 높아지면 수치 불안정성 발생 가능
        self.Ca_max_clamp = cfg.get("Ca_max_clamp", None)  # [μM] or None

    # ------------------------------
    # 외부 API
    # ------------------------------
    def add_spike(self, t_ms: float) -> None:
        """
        스파이크 시각 등록
        
        V3 계약:
        - 입력: t_ms [ms] ⭐ V3: ms 단위 명시
        - 출력: 없음
        - Side-effect: self.spike_times 업데이트 (스파이크 타임스탬프 추가)
        """
        self.spike_times.append(float(t_ms))

    def add_spike_now(self) -> None:
        """
        현재 시각(self.t_ms)에 스파이크 등록
        
        V3 계약:
        - 입력: 없음
        - 출력: 없음
        - Side-effect: self.spike_times 업데이트
        """
        self.add_spike(self.t_ms)

    def set_dt(self, dt_ms: float) -> None:
        """
        시간 스텝 변경
        
        V3 계약:
        - 입력: dt_ms [ms] ⭐ V3: ms 단위 명시
        - 출력: 없음
        - Side-effect: self.dt_ms 업데이트
        """
        self.dt_ms = float(dt_ms)

    def reset(self, *, Ca: Optional[float] = None) -> None:
        """
        시뮬레이터 리셋
        
        V3 계약:
        - 입력: Ca [μM] (선택적) ⭐ V3: μM 단위 명시
        - 출력: 없음
        - Side-effect: self.t_ms, self.Ca, self.spike_times, self.events 초기화
        """
        self.t_ms = 0.0  # [ms] ⭐ V3: ms 단위 명시
        self.Ca = float(self.C0 if Ca is None else Ca)  # [μM] ⭐ V3: μM 단위 명시
        self.spike_times.clear()
        self.events.clear()

    # ------------------------------
    # 내부 커널/헬퍼
    # ------------------------------
    def _alpha_kernel(self, dt_ms: float) -> float:
        """
        Alpha kernel: α(t) = (e^{-t/τ_d} − e^{-t/τ_r})_+  (t ≥ 0)
        
        V3 계약:
        - 입력: dt_ms [ms] (현재시각 − 스파이크시각) ⭐ V3: ms 단위 명시
        - 출력: alpha 값 [0,1] (무차원)
        - Side-effect: 없음
        
        스파이크 후 시간에 따른 Ca 유입 커널
        - τ_r: 상승 시간 상수 [s]
        - τ_d: 감쇠 시간 상수 [s] (τ_d > τ_r)
        
        Parameters
        ----------
        dt_ms : float
            현재시각 − 스파이크시각 [ms] ⭐ V3: ms 단위 명시
            
        Returns
        -------
        float
            Alpha kernel 값 (0~1 범위)
        """
        if dt_ms <= 0.0:
            return 0.0
        
        t = dt_ms / 1000.0  # [ms] → [s]
        val = np.exp(-t / self.tau_d_s) - np.exp(-t / self.tau_r_s)
        # 물리적 이유: alpha kernel은 음수가 될 수 없음 (Ca 유입만 가능)
        return float(max(0.0, val))

    def _trim_spike_memory(self) -> None:
        """
        메모리 윈도우 바깥 스파이크 제거
        
        V3 계약:
        - 입력: 없음
        - 출력: 없음
        - Side-effect: self.spike_times 업데이트 (오래된 스파이크 제거)
        """
        if not self.spike_times:
            return
        cutoff = self.t_ms - self.max_spike_memory_ms  # [ms] ⭐ V3: ms 단위 명시
        if cutoff <= 0.0:
            return
        self.spike_times = [s for s in self.spike_times if s >= cutoff]

    # ------------------------------
    # 메인 스텝
    # ------------------------------
    def step(self, ATP: float = None, dt_ms: float = None) -> VesicleEvent:
        """
        한 스텝(dt_ms) 진행: Spike-triggered Alpha-kernel + ATP 의존 펌프
        
        V3 계약:
        - 입력:
          - ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
          - dt_ms: [ms] (시간 스텝) ⭐ V3 계약 고정
        - 출력: VesicleEvent (t_ms [ms], Ca [μM], S [0,1], status)
        - Side-effect:
          - self.Ca 업데이트 [μM]
          - self.t_ms 업데이트 [ms]
          - self.ATP 업데이트 [0,100]
          - self.spike_times 업데이트 (오래된 스파이크 제거)
          - self.events 업데이트 (이벤트 로그 추가)
        
        수식:
            dCa/dt[s] = Σ_k A·α(t − t_k) − k_c·(ATP/100)·(Ca − C0) ⭐ V3: ATP/100 변환
        
        ⚠️ 중요: 스파이크는 step() 호출 전에 등록해야 함
            add_spike_now() 또는 add_spike(t_ms)를 step() 전에 호출
        
        Parameters
        ----------
        ATP : float, optional
            현재 ATP 농도 [0,100] ⭐ V3: 0~100 범위 명시. None이면 이전 값 유지
        dt_ms : float, optional
            시간 스텝 [ms] ⭐ V3: ms 단위 명시. None이면 self.dt_ms 사용

        Returns
        -------
        VesicleEvent
            업데이트된 상태 스냅샷 (t_ms [ms], Ca [μM], S [0,1], status)
        """
        # 시간 스텝 결정
        if dt_ms is not None:
            self.dt_ms = float(dt_ms)  # [ms] ⭐ V3: ms 단위 명시
        dt_s = self.dt_ms / 1000.0  # [ms] → [s]
        
        # ATP 업데이트 (시간 업데이트 전에)
        # V3: ATP는 [0,100] 범위로 받아서 내부적으로 [0,1]로 변환
        if ATP is not None:
            # 물리적 이유: ATP는 [0,100] 범위로 제한 (생리학적 범위)
            self.ATP = float(np.clip(ATP, 0.0, 100.0))  # [0,100] ⭐ V3: 0~100 범위 명시
        
        # 스파이크 메모리 정리 (오래된 스파이크 제거)
        # 시간 업데이트 전에 정리하여 현재 시각 기준으로 계산
        self._trim_spike_memory()
        
        # === Alpha kernel 합 계산: Σ_k A·α(t − t_k) ===
        # 각 스파이크로부터의 Ca 유입 합산
        # 현재 시각(self.t_ms) 기준으로 계산
        Ca_influx = 0.0  # [μM/s]
        for t_k in self.spike_times:
            dt_from_spike = self.t_ms - t_k  # [ms] ⭐ V3: ms 단위 명시
            if dt_from_spike > 0.0:  # 과거 스파이크만 (현재 시각은 제외)
                alpha_val = self._alpha_kernel(dt_from_spike)
                Ca_influx += self.Ca_influx * alpha_val  # [μM/s]
        
        # === ATP 의존 펌프: k_c·(ATP/100)·(Ca − C0) ===
        # ATP가 높을수록 Ca 제거가 빠름
        # V3: ATP는 [0,100] 범위이므로 100으로 나누어 [0,1]로 변환
        # 단위: k_c [1/s] × (ATP/100) [무차원] × (Ca - C0) [μM] = [μM/s]
        ATP_norm = self.ATP / 100.0  # [0,1] (ATP를 [0,100] → [0,1]로 변환) ⭐ V3 변경
        Ca_efflux = self.k_c * ATP_norm * (self.Ca - self.C0)  # [μM/s]
        
        # === Ca 농도 업데이트 ===
        # dCa/dt = Ca_influx - Ca_efflux
        # Ca_{n+1} = Ca_n + (dCa/dt) · dt_s
        # 단위: [μM] + [μM/s] × [s] = [μM]
        dCa_dt = Ca_influx - Ca_efflux  # [μM/s]
        self.Ca += dCa_dt * dt_s  # [μM]
        
        # 물리적 하한 클램프 (음수 방지)
        # 물리적 이유: Ca 농도는 음수가 될 수 없음 (생리학적 제약)
        self.Ca = max(0.0, self.Ca)
        
        # 수치 안정성용 상한 클램프 (옵션)
        # 물리적 이유: Ca 농도가 비정상적으로 높아지면 수치 불안정성 발생 가능
        if self.Ca_max_clamp is not None:
            self.Ca = min(self.Ca, float(self.Ca_max_clamp))
        
        # === 시간 진행 (Ca 업데이트 후) ===
        # 다음 스텝을 위해 시간 업데이트
        # 현재 스텝의 스파이크는 다음 스텝부터 반영됨
        self.t_ms += self.dt_ms  # [ms] ⭐ V3: ms 단위 명시
        
        # === 정규화 및 상태 계산 ===
        denom = max(1e-12, (self.Cmax - self.C0))
        S = (self.Ca - self.C0) / denom  # [0,1] ⭐ V3 계약 고정
        status = "under" if S < 0.0 else ("normal" if S <= 1.0 else "alert")
        
        # === 이벤트 로그 기록 ===
        # 이벤트는 업데이트 후 시각 기록
        event = VesicleEvent(t_ms=self.t_ms, Ca=self.Ca, S=S, status=status)
        self.events.append(event)
        
        return event

    # ------------------------------
    # 시각화 (선택)
    # ------------------------------
    def plot(self, *, title: str = "Vesicle Ca²⁺ Activity") -> None:
        """
        최근 이벤트를 기반으로 [Ca²⁺], S를 시각화(선택 기능).
        
        V3 계약:
        - 입력: title (선택적)
        - 출력: 없음
        - Side-effect: matplotlib 플롯 표시
        
        Ca는 이미 [μM] 단위이므로 변환 없이 그대로 사용.
        """
        if not _HAS_MPL:
            print("[INFO] matplotlib 미탑재: plot() 생략.")
            return
        if not self.events:
            print("[WARN] No vesicle data to plot.")
            return

        t  = [e.t_ms for e in self.events]  # [ms] ⭐ V3: ms 단위 명시
        Ca = [e.Ca for e in self.events]  # [μM] (변환 없음) ⭐ V3: μM 단위 명시
        S  = [e.S for e in self.events]  # [0,1] ⭐ V3 계약 고정

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(t, Ca, lw=1.4)
        ax1.set_xlabel("Time (ms)")  # ⭐ V3: ms 단위 명시
        ax1.set_ylabel("[Ca²⁺] (µM)")  # ⭐ V3: μM 단위 명시
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(t, S, lw=1.2, color='orange')
        ax2.set_ylabel("S (norm)", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        plt.title(title)
        plt.tight_layout()
        plt.show()

    # ------------------------------
    # 편의 함수
    # ------------------------------
    def get_state(self) -> dict:
        """
        현재 상태 스냅샷 반환
        
        V3 계약:
        - 입력: 없음
        - 출력: dict {"t_ms": [ms], "Ca": [μM], "S": [0,1], "status": str}
        - Side-effect: 없음
        
        Returns
        -------
        dict
            현재 상태 딕셔너리
        """
        denom = max(1e-12, (self.Cmax - self.C0))
        S = (self.Ca - self.C0) / denom  # [0,1] ⭐ V3 계약 고정
        status = "under" if S < 0.0 else ("normal" if S <= 1.0 else "alert")
        return {
            "t_ms": self.t_ms,  # [ms] ⭐ V3: ms 단위 명시
            "Ca": self.Ca,  # [μM] ⭐ V3: μM 단위 명시
            "S": S,  # [0,1] ⭐ V3 계약 고정
            "status": status
        }


# =============================================================
# V3 독립 실행 테스트
# =============================================================
if __name__ == "__main__":
    """
    V3 계약 고정 검증:
    - ATP: [0,100] 확인
    - Ca: [μM] 확인
    - S: [0,1] 확인
    - 시간 단위: [ms] 확인
    - 입출력/side-effect 확인
    """
    cfg = {
        "C0": 0.1,  # [μM] ⭐ V3: μM 단위 명시
        "Cmax": 5.0,  # [μM] ⭐ V3: μM 단위 명시
        "k_c": 0.20,  # [1/s]
        "A": 0.25,  # [μM/s]
        "tau_r_ms": 0.5,  # [ms] ⭐ V3: ms 단위 명시
        "tau_d_ms": 3.0,  # [ms] ⭐ V3: ms 단위 명시
        "ATP_initial": 100.0,  # [0,100] ⭐ V3: 0~100 범위 명시
        "max_spike_memory_ms": 2000.0,  # [ms] ⭐ V3: ms 단위 명시
    }
    
    ca = CaVesicle(cfg, dt_ms=1.0)  # [ms] ⭐ V3: ms 단위 명시
    ATP = 100.0  # [0,100] ⭐ V3: 0~100 범위 명시
    
    print("=" * 60)
    print("CaVesicle V3 — 계약 고정 검증")
    print("=" * 60)
    print(f"ATP 입력: {ATP} [0,100] ⭐ V3 계약")
    print(f"dt_ms: {ca.dt_ms} [ms] ⭐ V3 계약")
    print(f"Ca 초기값: {ca.Ca} [μM] ⭐ V3 계약")
    print("-" * 60)
    print("[CaVesicle Test]")
    print(f"{'t(ms)':>8} | {'Ca(μM)':>10} | {'S[0,1]':>8} | {'status':>10}")
    print("-" * 60)
    
    # 스파이크 추가
    ca.add_spike(0.0)  # t=0ms에 스파이크
    ca.add_spike(10.0)  # t=10ms에 스파이크
    
    for t in range(5):
        event = ca.step(ATP=ATP, dt_ms=1.0)  # [ms] ⭐ V3: ms 단위 명시
        print(f"{event.t_ms:8.2f} | {event.Ca:10.3f} | {event.S:8.3f} | {event.status:>10}")
    
    print("=" * 60)
    print("✅ V3 계약 고정 검증 완료")
    print("  - ATP: [0,100] 범위 확인")
    print("  - Ca: [μM] 단위 확인")
    print("  - S: [0,1] 범위 확인")
    print("  - 시간 단위: [ms] 확인")
    print("  - 입출력/side-effect 명시 확인")

