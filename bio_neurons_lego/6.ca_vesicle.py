# =============================================================
# ca_vesicle.py — Ca²⁺ Vesicle (Spike-triggered Alpha kernels)
# =============================================================
# 목적:
#   • 스파이크 시각 목록 {t_k}에 의해 유도되는 Ca²⁺ 유입(α-커널 합)과
#     ATP-의존 펌프에 의한 Ca 제거를 함께 모델링.
#   • 정규화 시그널 S=(Ca−C0)/(Cmax−C0) 및 상태 레이블(under/normal/alert) 제공.
#
# 수식(시간 단위 주의: dt_ms[ms], τ_r, τ_d는 [s]):
#   α(t)      = (e^{−t/τ_d} − e^{−t/τ_r})_+    ,  t ≥ 0
#   dCa/dt[s] =  Σ_k  A·α(t − t_k)  −  k_c·ATP·(Ca − C0)
#   Ca[t+Δt]  =  Ca[t] + (dCa/dt)·(Δt_ms / 1000)
#   S         =  (Ca − C0) / (Cmax − C0)
#
#   • A        : α-커널 스케일(유입 크기, 농도 단위/초에 맞춰 해석적 스케일)
#   • k_c      : 펌프 계수(ATP 의존), ATP↑ → 제거↑
#   • C0, Cmax : 기준 및 최대 스케일(정규화용 경계)
#
# 안전장치:
#   • τ_d ≤ τ_r면 물리적 의미가 사라지므로, 초기화 시 τ_d>τ_r 되도록 자동 수정.
#   • Ca는 음수가 되지 않도록 하한만 설정(물리적 클램프 X).
#   • 스파이크 메모리는 max_spike_memory_ms 윈도우로 관리.
#
# 반환:
#   VesicleEvent(t_ms, Ca, S, status) — 각 스텝 상태 스냅샷
#
# =============================================================

from __future__ import annotations
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
    t_ms: float
    Ca: float
    S: float
    status: str  # "under" | "normal" | "alert"


class CaVesicle:
    r"""
    Spike-triggered Ca²⁺ Vesicle Dynamics (Alpha-kernel + ATP pump)

    미분방정식 (연속시간, t는 초[s]):
        α(t)      = (exp(−t/τ_d) − exp(−t/τ_r))_+ ,  τ_d > τ_r > 0
        dCa/dt[s] =  Σ_k A·α(t − t_k) − k_c·ATP·(Ca − C0)

    이산 적분 (dt_ms[ms]):
        Ca_{n+1} = Ca_n + (dCa/dt)·(dt_ms/1000)

    정규화:
        S = (Ca − C0) / (Cmax − C0)

    상태 레이블:
        S < 0      → "under"
        0 ≤ S ≤ 1  → "normal"
        S > 1      → "alert"
    """

    def __init__(self, cfg: dict, dt_ms: float):
        self.dt_ms = float(dt_ms)
        self.t_ms = 0.0
        self.C0 = cfg["C0"]
        self.Cmax = cfg["Cmax"]
        self.Ca = float(self.C0)
        self.k_c = cfg["k_c"]
        self.Ca_influx = cfg.get("Ca_influx", cfg.get("A", 200.0))
        self.max_spike_memory_ms = cfg.get("max_spike_memory_ms", 2000.0)
        self.spike_times: List[float] = []
        self.events: List[VesicleEvent] = []
        self.prev_Vm = -70.0

    # ------------------------------
    # 외부 API
    # ------------------------------
    def add_spike(self, t_ms: float) -> None:
        """스파이크 시각 등록(단위: ms)."""
        self.spike_times.append(float(t_ms))

    def add_spike_now(self) -> None:
        """현재 시각(self.t_ms)에 스파이크 등록."""
        self.add_spike(self.t_ms)

    def set_dt(self, dt_ms: float) -> None:
        """시간 스텝(ms) 변경."""
        self.dt_ms = float(dt_ms)

    def reset(self, *, Ca: Optional[float] = None) -> None:
        """시뮬레이터 리셋(시각, 이벤트 로그 유지/삭제는 선택)."""
        self.t_ms = 0.0
        self.Ca = float(self.C0 if Ca is None else Ca)
        self.spike_times.clear()
        self.events.clear()

    # ------------------------------
    # 내부 커널/헬퍼
    # ------------------------------
    def _alpha_kernel(self, dt_ms: float) -> float:
        """
        α(t) = (e^{-t/τ_d} − e^{-t/τ_r})_+  (t ≥ 0)
        인자 dt_ms: 현재시각 − 스파이크시각 [ms]
        """
        if dt_ms <= 0.0:
            return 0.0
        t = dt_ms / 1000.0  # [s]
        val = np.exp(-t / self.tau_d_s) - np.exp(-t / self.tau_r_s)
        return float(max(0.0, val))

    def _trim_spike_memory(self) -> None:
        """메모리 윈도우 바깥 스파이크 제거."""
        if not self.spike_times:
            return
        cutoff = self.t_ms - self.max_spike_memory_ms
        if cutoff <= 0.0:
            return
        self.spike_times = [s for s in self.spike_times if s >= cutoff]

    # ------------------------------
    # 메인 스텝
    # ------------------------------
    def step(self, Vm: float, dt_ms: float) -> float:
        """
        한 스텝(dt_ms) 진행:
          • Vm을 받아서 발화 시에만 Ca 유입
          • Vm에 따른 지수 감쇠 유입 모델

        Parameters
        ----------
        Vm : float
            막전위 [mV]
        dt_ms : float
            시간 스텝 [ms]

        Returns
        -------
        float
            업데이트된 Ca 농도 [μM]
        """
        self.t_ms += dt_ms
        # Calcium update (stable form)
        Ca_in = self.Ca_influx * max(0.0, Vm) * np.exp(-Vm / 40.0)
        Ca_out = self.k_c * self.Ca
        self.Ca += dt_ms * (Ca_in - Ca_out)
        self.Ca = np.clip(self.Ca, 0.0, 2e-4)
        
        return self.Ca

    # ------------------------------
    # 시각화 (선택)
    # ------------------------------
    def plot(self, *, title: str = "Vesicle Ca²⁺ Activity") -> None:
        """최근 이벤트를 기반으로 [Ca²⁺], S를 시각화(선택 기능)."""
        if not _HAS_MPL:
            print("[INFO] matplotlib 미탑재: plot() 생략.")
            return
        if not self.events:
            print("[WARN] No vesicle data to plot.")
            return

        t  = [e.t_ms for e in self.events]
        Ca = [e.Ca * 1e6 for e in self.events]  # M → µM
        S  = [e.S for e in self.events]

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(t, Ca, lw=1.4)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("[Ca²⁺] (µM)")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(t, S, lw=1.2)
        ax2.set_ylabel("S (norm)")

        plt.title(title)
        plt.tight_layout()
        plt.show()

    # ------------------------------
    # 편의 함수
    # ------------------------------
    def get_state(self) -> dict:
        """현재 상태 스냅샷 반환."""
        denom = max(1e-12, (self.Cmax - self.C0))
        S = (self.Ca - self.C0) / denom
        status = "under" if S < 0.0 else ("normal" if S <= 1.0 else "alert")
        return {"t_ms": self.t_ms, "Ca": self.Ca, "S": S, "status": status}