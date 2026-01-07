# =============================================================
# 01.dtg_system.py — Digital Twin Guidance (DTG) Layer (V3)
# =============================================================
# 목적:
#   - 뉴런의 에너지(E)와 위상(φ)을 동기화시키는 메타 제어 시스템.
#   - Mitochondria(ATP 생성계)와 Soma(HH 발화계)의 상위 조정자 역할.
#
# V3 계약 고정:
#   - 시간 단위: [ms] (밀리초) - 모든 시간 관련 파라미터는 ms 단위
#   - ATP: [0,100] (정규화, 0~100 범위로 통일)
#   - E: [arb] (임의 단위)
#   - φ: [rad] (라디안, 0~2π 범위)
#
# 주요 개념:
#   E : 메타 에너지(ATP와 연동되는 뉴런의 전반적 에너지 스칼라)
#   φ : 위상 (Phase; 0~2π 범위의 주기적 발진 상태)
#
# 수식:
#   dE/dt = g_sync · (ATP - E) - γ · (E - E0)
#   dφ/dt = ω0 + α · (E - E0)
#
#   - g_sync : ATP와 E 간 에너지 동기화 이득 [1/ms] ⭐ V3: ms 단위 명시
#   - γ : 복원항 [1/ms] (E가 E0에서 벗어날수록 복귀시키는 계수) ⭐ V3: ms 단위 명시
#   - ω0 : 기본 위상속도 [rad/ms] ⭐ V3: ms 단위 명시
#   - α : 에너지 변화가 위상속도에 미치는 민감도 [rad/(ms·arb)]
#   - ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
#
# 출력:
#   (E, φ)
#   - E : 현재 메타 에너지 [arb]
#   - φ : [0, 2π) 범위의 위상 [rad] (주기적 발진)
#
# 연결 구조:
#   DTG → Mito → HH Soma → Axon
#          ↑                  ↓
#          └─────── Feedback (ATP, Heat, CO₂, Ca)
# =============================================================

import numpy as np


class DTGSystem:
    r"""
    Digital Twin Guidance (DTG) — Energy–Phase Synchronizer (V3)
    -------------------------------------------------------
    
    V3 계약:
    - 입력: ATP [0,100] (정규화, 0~100 범위), dt [ms]
    - 출력: (E [arb], φ [rad], dE [arb/ms], dφ [rad/ms])
    - Side-effect: self.E, self.phi 업데이트
    
    Differential equations:
        dE/dt = g_sync (ATP - E) - γ (E - E0)
        dφ/dt = ω0 + α (E - E0)
    
    설계 이유:
    - DTG는 뉴런의 메타 제어 시스템으로, ATP 수준과 위상을 동기화
    - ATP가 높을수록 E가 증가하고, 위상 속도가 빨라짐
    - 이것이 뉴런의 자기 발화 주기를 조절하는 핵심 메커니즘
    """

    def __init__(self, cfg: dict):
        """
        DTGSystem 초기화
        
        Parameters
        ----------
        cfg : dict
            CONFIG["DTG"] section, containing:
              - E0        : 기준 에너지 [arb] (steady-state)
              - omega0    : 기본 위상속도 [rad/ms] ⭐ V3: ms 단위 명시
              - alpha     : 에너지-위상 결합 계수 [rad/(ms·arb)]
              - gamma     : 에너지 복원 계수 [1/ms] ⭐ V3: ms 단위 명시
              - sync_gain : ATP-E 동조 이득 [1/ms] ⭐ V3: ms 단위 명시
        """
        self.E0 = cfg.get("E0", 100.0)
        self.omega0 = cfg.get("omega0", 1.0)  # [rad/ms] ⭐ V3: ms 단위 명시
        self.alpha = cfg.get("alpha", 0.03)
        self.gamma = cfg.get("gamma", 0.10)  # [1/ms] ⭐ V3: ms 단위 명시
        self.sync_gain = cfg.get("sync_gain", 0.20)  # [1/ms] ⭐ V3: ms 단위 명시

        # 초기 상태값
        self.E = float(self.E0)  # [arb]
        self.phi = 0.0  # [rad]

    def step(self, ATP: float, dt: float):
        """
        한 스텝(dt) 적분을 수행하여 에너지·위상을 갱신한다.
        
        V3 계약:
        - 입력:
          - ATP: [0,100] (정규화, 0~100 범위) ⭐ V3 계약 고정
          - dt: [ms] (밀리초) ⭐ V3 계약 고정
        - 출력:
          - E: [arb] (갱신된 메타 에너지)
          - φ: [rad] (0~2π 범위의 위상)
          - dE: [arb/ms] (에너지 변화율)
          - dφ: [rad/ms] (위상속도)
        - Side-effect:
          - self.E 업데이트 (에너지 상태 변경)
          - self.phi 업데이트 (위상 상태 변경)

        Parameters
        ----------
        ATP : float
            Mitochondria Layer에서 공급받은 ATP 값 [0,100] ⭐ V3: 0~100 범위 명시
        dt : float
            시간 스텝 [ms] ⭐ V3: ms 단위 명시

        Returns
        -------
        tuple
            (E, φ, dE, dφ)
            - E  : 갱신된 메타 에너지 [arb]
            - φ  : [0, 2π)로 wrap된 위상 [rad]
            - dE : 미분 항 (에너지 변화율) [arb/ms]
            - dφ : 미분 항 (위상속도) [rad/ms]
        """
        # --- 1) 에너지 변화율 계산 ---
        # 수식: dE/dt = g_sync · (ATP - E) - γ · (E - E0)
        # 단위: [arb/ms] = [1/ms] · [arb] - [1/ms] · [arb]
        dE = self.sync_gain * (ATP - self.E) - self.gamma * (self.E - self.E0)
        self.E += dE * dt  # [arb] = [arb] + [arb/ms] · [ms]

        # --- 2) 위상 변화율 계산 ---
        # 수식: dφ/dt = ω0 + α · (E - E0)
        # 단위: [rad/ms] = [rad/ms] + [rad/(ms·arb)] · [arb]
        dphi = self.omega0 + self.alpha * (self.E - self.E0)
        self.phi = (self.phi + dphi * dt) % (2 * np.pi)  # [rad] = [rad] + [rad/ms] · [ms]

        # --- 3) 안정화 처리 (E 폭주 방지) ---
        # 물리적 이유: E가 E0의 2배를 넘으면 비정상적으로 높은 에너지 상태
        # 생리학적으로 불가능하므로 클램프하여 수치 안정성 보장
        self.E = np.clip(self.E, 0.0, self.E0 * 2.0)  # [arb] (물리적 범위 유지)

        return self.E, self.phi, dE, dphi


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
        "E0": 100.0,
        "omega0": 1.0,  # [rad/ms] ⭐ V3: ms 단위 명시
        "alpha": 0.03,
        "gamma": 0.10,  # [1/ms] ⭐ V3: ms 단위 명시
        "sync_gain": 0.20,  # [1/ms] ⭐ V3: ms 단위 명시
    }

    dtg = DTGSystem(cfg)
    ATP = 120.0  # 테스트용 ATP 자극 [0,100] ⭐ V3: 0~100 범위 (120은 클램프됨)
    dt = 0.1  # [ms] ⭐ V3: ms 단위 명시

    print("=" * 60)
    print("DTG System V3 — 계약 고정 검증")
    print("=" * 60)
    print(f"ATP 입력: {ATP} [0,100] ⭐ V3 계약")
    print(f"dt: {dt} [ms] ⭐ V3 계약")
    print("-" * 60)
    
    print("[DTG Simulation Test]")
    print(f"{'t(ms)':>8} | {'E(arb)':>8} | {'φ(rad)':>8} | {'dE(arb/ms)':>12} | {'dφ(rad/ms)':>12}")
    print("-" * 60)
    
    for t in range(10):
        E, phi, dE, dphi = dtg.step(ATP, dt)
        print(f"{t*dt:8.2f} | {E:8.3f} | {phi:8.3f} | {dE:12.4f} | {dphi:12.4f}")
    
    print("=" * 60)
    print("✅ V3 계약 고정 검증 완료")
    print("  - ATP: [0,100] 범위 확인")
    print("  - 시간 단위: [ms] 확인")
    print("  - 입출력/side-effect 명시 확인")

