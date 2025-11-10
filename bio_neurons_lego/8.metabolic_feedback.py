# =============================================================
# metabolic_feedback.py — Heat·CO₂·Ca 기반 대사 피드백 루프
# =============================================================
# 목적:
#   • 미토콘드리아(Mitochondria)의 에너지 효율(η),
#     손실율(P_loss), 회복률(recover_k)을
#     발열(Heat), 이산화탄소(CO₂), 칼슘(Ca²⁺) 상태에 따라
#     동적으로 보정하는 생리학적 피드백 루프를 구현한다.
#
# 연동:
#   - 입력:  Mito (Heat, CO₂), CaVesicle.status("under"/"normal"/"alert")
#   - 출력:  Mito 내부 변수 (η, P_loss, recover_k)
#
# 생리학적 근거:
#   Heat ↑  → 미토콘드리아 효율(η) ↓
#   CO₂ ↑   → 에너지 손실률(P_loss) ↑
#   Ca alert → ATP 회복률(recover_k) ↑
#   Ca under → ATP 회복률(recover_k) ↓
#
# =============================================================

import numpy as np


class MetabolicFeedback:
    r"""
    MetabolicFeedback — Energy Homeostasis Feedback Controller
    ------------------------------------------------------------
    ⚙️ 역할:
        미토콘드리아의 대사 효율(η), 손실률(P_loss),
        회복률(recover_k)을 Heat·CO₂·Ca 상태에 따라 갱신한다.

    ------------------------------------------------------------
    📘 연동 계층:
        - 입력:  Mitochondria (Heat, CO₂), CaVesicle.status
        - 출력:  Mito 내부 변수 수정 (η, P_loss, recover_k)

    ------------------------------------------------------------
    📐 수식 요약:
        (1) 발열(Heat) → 효율 저하
            η(t+Δt) = η₀ − β_heat · (Heat − Heat₀)
            η ∈ [0.05, η₀]

        (2) 이산화탄소(CO₂) → 손실율 증가
            P_loss(t+Δt) = P_loss₀ · (1 + β_CO₂ · CO₂)

        (3) 칼슘(Ca²⁺) 상태 → 회복률 조정
            recover_k(t+Δt) =
                ┌ k₀ · (1 + λ_Ca)       , if Ca_status = "alert"
                ├ k₀ · (1 − λ_under)    , if Ca_status = "under"
                └ k₀                    , otherwise
    ------------------------------------------------------------
    """

    def __init__(self, mito, cfg=None):
        """
        Parameters
        ----------
        mito : object
            Mitochondria 인스턴스. (필수)
            다음 속성을 가져야 함:
                • mito.Heat
                • mito.CO2
                • mito.eta
                • mito.Ploss
                • mito.recover_k
        cfg : dict, optional
            피드백 계수 설정값. 기본값:
                β_heat   = 0.0015   # Heat → η 감소 계수
                β_CO₂    = 0.0010   # CO₂ → P_loss 증가 계수
                λ_Ca     = 0.3      # Ca alert 시 회복 강화 비율
                λ_under  = 0.1      # Ca under 시 회복 억제 비율
        """
        self.mito = mito
        self.cfg = cfg or {
            "beta_heat": 0.0015,
            "beta_co2": 0.0010,
            "lambda_ca": 0.3,
            "lambda_under": 0.1,
        }

        # --- 기준값 저장 ---
        #   기준 효율(η₀), 손실율(P_loss₀), 회복률(k₀)
        self.eta_base = getattr(mito, "eta0", 0.60)
        self.Ploss_base = getattr(mito, "Ploss", 1.5)
        self.recover_base = getattr(mito, "recover_k", 8.0)

    # =========================================================
    # 메인 피드백 업데이트
    # =========================================================
    def update(self, ca_status: str):
        """
        Heat·CO₂·Ca 상태에 따라 Mitochondria 내부 변수 보정.

        Parameters
        ----------
        ca_status : str
            "alert" | "normal" | "under"
            CaVesicle.get_state()["status"] 값 사용.
        """

        # -----------------------------------------------------
        # (1) Heat ↑ → 효율 η ↓
        # η(t+Δt) = η₀ − β_heat · (Heat − Heat₀)
        # -----------------------------------------------------
        delta_eta = - self.cfg["beta_heat"] * max(0.0, self.mito.Heat)
        new_eta = self.eta_base + delta_eta
        # η는 물리적 하한 0.05 이하로 떨어지지 않음
        self.mito.eta = float(np.clip(new_eta, 0.05, 1.0))

        # -----------------------------------------------------
        # (2) CO₂ ↑ → 손실률 P_loss ↑
        # P_loss(t+Δt) = P_loss₀ · (1 + β_CO₂ · CO₂)
        # -----------------------------------------------------
        new_Ploss = self.Ploss_base * (1.0 + self.cfg["beta_co2"] * max(0.0, self.mito.CO2))
        self.mito.Ploss = float(np.clip(new_Ploss, 0.0, 100.0))

        # -----------------------------------------------------
        # (3) Ca 상태 → 회복률 recover_k 조정
        # -----------------------------------------------------
        if ca_status == "alert":
            # 🔺 과활성 상태: ATP 회복률 강화
            new_recover = self.recover_base * (1.0 + self.cfg["lambda_ca"])
        elif ca_status == "under":
            # 🔻 비활성 상태: 회복 억제
            new_recover = self.recover_base * (1.0 - self.cfg["lambda_under"])
        else:
            # 🟢 정상 상태: 기본값 유지
            new_recover = self.recover_base

        self.mito.recover_k = float(np.clip(new_recover, 0.0, 50.0))

    # =========================================================
    # 상태 출력 (디버깅 및 로깅용)
    # =========================================================
    def summary(self) -> dict:
        """
        현재 피드백 조정 후의 Mitochondria 주요 변수 반환.
        """
        return {
            "eta": round(self.mito.eta, 5),
            "Ploss": round(self.mito.Ploss, 5),
            "recover_k": round(self.mito.recover_k, 5),
            "Heat": round(self.mito.Heat, 5),
            "CO2": round(self.mito.CO2, 5),
        }


# =============================================================
# 단독 테스트 (optional)
# =============================================================
if __name__ == "__main__":
    # 더미 Mito 클래스 생성
    class DummyMito:
        def __init__(self):
            self.eta0 = 0.6
            self.Heat = 100.0
            self.CO2 = 50.0
            self.eta = 0.6
            self.Ploss = 1.5
            self.recover_k = 8.0

    mito = DummyMito()
    fb = MetabolicFeedback(mito)

    # 테스트 케이스
    for status in ["under", "normal", "alert"]:
        fb.update(status)
        print(f"\n[Ca 상태: {status}]")
        print(fb.summary())