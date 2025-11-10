# =============================================================
# dtg_system.py — Digital Twin Guidance (DTG) Layer
# =============================================================
# 목적:
#   - 뉴런의 에너지(E)와 위상(φ)을 동기화시키는 메타 제어 시스템.
#   - Mitochondria(ATP 생성계)와 Soma(HH 발화계)의 상위 조정자 역할.
#
# 주요 개념:
#   E : 메타 에너지(ATP와 연동되는 뉴런의 전반적 에너지 스칼라)
#   φ : 위상 (Phase; 0~2π 범위의 주기적 발진 상태)
#
# 수식:
#   dE/dt = g_sync · (ATP - E) - γ · (E - E0)
#   dφ/dt = ω0 + α · (E - E0)
#
#   - g_sync : ATP와 E 간 에너지 동기화 이득
#   - γ : 복원항 (E가 E0에서 벗어날수록 복귀시키는 계수)
#   - ω0 : 기본 위상속도(rad/ms)
#   - α : 에너지 변화가 위상속도에 미치는 민감도
#
# 출력:
#   (E, φ)
#   - E : 현재 메타 에너지
#   - φ : [0, 2π) 범위의 위상 (주기적 발진)
#
# 연결 구조:
#   DTG → Mito → HH Soma → Axon
#          ↑                  ↓
#          └─────── Feedback (ATP, Heat, CO₂, Ca)
# =============================================================

import numpy as np

class DTGSystem:
    r"""
    Digital Twin Guidance (DTG) — Energy–Phase Synchronizer
    -------------------------------------------------------
    Differential equations:
        dE/dt = g_sync (ATP - E) - γ (E - E0)
        dφ/dt = ω0 + α (E - E0)
    """

    def __init__(self, cfg: dict):
        """
        Parameters
        ----------
        cfg : dict
            CONFIG["DTG"] section, containing:
              - E0        : 기준 에너지 (steady-state)
              - omega0    : 기본 위상속도 [rad/ms]
              - alpha     : 에너지-위상 결합 계수
              - gamma     : 에너지 복원 계수
              - sync_gain : ATP-E 동조 이득
        """
        self.E0 = cfg.get("E0", 100.0)
        self.omega0 = cfg.get("omega0", 1.0)
        self.alpha = cfg.get("alpha", 0.03)
        self.gamma = cfg.get("gamma", 0.10)
        self.sync_gain = cfg.get("sync_gain", 0.20)

        # 초기 상태값
        self.E = float(self.E0)
        self.phi = 0.0  # [rad]

    def step(self, ATP: float, dt: float):
        """
        한 스텝(dt) 적분을 수행하여 에너지·위상을 갱신한다.

        Parameters
        ----------
        ATP : float
            Mitochondria Layer에서 공급받은 ATP 값.
        dt : float
            시간 스텝 [ms].

        Returns
        -------
        tuple
            (E, φ, dE, dφ)
            - E  : 갱신된 메타 에너지
            - φ  : [0, 2π)로 wrap된 위상(rad)
            - dE : 미분 항 (에너지 변화율)
            - dφ : 미분 항 (위상속도)
        """
        # --- 1) 에너지 변화율 계산 ---
        dE = self.sync_gain * (ATP - self.E) - self.gamma * (self.E - self.E0)
        self.E += dE * dt

        # --- 2) 위상 변화율 계산 ---
        dphi = self.omega0 + self.alpha * (self.E - self.E0)
        self.phi = (self.phi + dphi * dt) % (2 * np.pi)

        # --- 3) 안정화 처리 (E 폭주 방지; 선택적) ---
        self.E = np.clip(self.E, 0.0, self.E0 * 2.0)

        return self.E, self.phi, dE, dphi


# =============================================================
# 단독 실행 테스트
# =============================================================
if __name__ == "__main__":
    cfg = {
        "E0": 100.0,
        "omega0": 1.0,
        "alpha": 0.03,
        "gamma": 0.10,
        "sync_gain": 0.20,
    }

    dtg = DTGSystem(cfg)
    ATP = 120.0  # 테스트용 ATP 자극
    dt = 0.1

    print("[DTG Simulation Test]")
    for t in range(10):
        E, phi, dE, dphi = dtg.step(ATP, dt)
        print(f"t={t*dt:5.2f} ms | E={E:7.3f} | φ={phi:6.3f} rad | dE={dE:7.4f} | dφ={dphi:7.4f}")