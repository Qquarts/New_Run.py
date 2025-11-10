# =============================================================
# mitochon_atp.py — Complete Bio-Metabolic Engine
# =============================================================
# 목적:
#   뉴런 내 미토콘드리아의 생리학적 ATP 생성/소비 과정을
#   실제 생화학 반응식 형태로 모델링한 완성형 코드.
#
#   구조:  Glucose + O₂ → ATP + Heat + CO₂
#
# 주요 기능:
#   - 동적 효율 계산 (Michaelis-Menten kinetics, O₂ 의존)
#   - Heat 확산 모델링 (1D 공간 확산, HeatGrid 통합)
#   - CO₂ 감쇠 (환경 균형)
#   - ATP 회복 메커니즘 (낮을 때 자동 회복)
#   - 수치 안정성 보장 (CFL 조건, 클램프)
#
# -------------------------------------------------------------
# [개념적 흐름]
# -------------------------------------------------------------
#   (1) Glucose 산화 입력 (Glycolysis + TCA)
#       P_in(Glu, O2) = k_glu·[Glu] + k_oxy·[O2]
#       - Glucose는 빠르지만 저효율
#       - O₂는 느리지만 고효율
#
#   (2) 에너지 저장 풀 (E_buf)
#       dE_buf/dt = P_in - P_loss - k_transfer·(E_buf - ATP)
#       - E_buf는 NADH, FADH₂, 전자전달계 에너지 풀에 해당
#
#   (3) ATP 합성 (ETC + ATP Synthase)
#       dATP/dt = η(O2)·k_transfer·(E_buf - ATP) - J_use
#       - η(O₂) = η₀·(O₂ / (O₂ + K_mO₂))
#       - J_use는 Na/K 펌프, Ca 펌프 등 ATP 소비 항
#
#   (4) 발열·CO₂ 누적 (대사 부산물)
#       Heat += (1 - η)·dATP_prod
#       CO₂  += c_CO₂·dATP_prod
#       - Heat는 1차원 공간 확산 (HeatGrid 사용)
#       - CO₂는 환경 감쇠 (k_co2)
#
#   (5) 회복 메커니즘
#       ATP < threshold → ATP += recover_k·(1 - ATP/100)
#
# -------------------------------------------------------------
# [단위 및 의미]
# -------------------------------------------------------------
#   - ATP, E_buf : arbitrary [a.u.]
#   - Glu, O₂     : 농도 [mM] 또는 상대 단위
#   - dt          : 시간 [ms]
#   - η           : 효율 (0~1)
#   - Heat, CO₂   : 누적량 [a.u.]
#
# -------------------------------------------------------------
# [요약 수식]
# -------------------------------------------------------------
#   η(O₂) = η₀ · O₂ / (O₂ + K_mO₂)
#   P_in(Glu,O₂) = k_glu·Glu + k_oxy·O₂
#   dE_buf/dt = P_in - P_loss - k_transfer·(E_buf - ATP)
#   dATP/dt   = η(O₂)·k_transfer·(E_buf - ATP) - J_use
# =============================================================

import numpy as np


class Mitochondria:
    r"""
    Biological Mitochondria Model — ATP Synthesis + Feedback
    ---------------------------------------------------------
    Simulates ATP generation from Glucose and Oxygen, including:
      - Dynamic efficiency (η)
      - Heat/CO₂ byproducts
      - Recovery when ATP is low
    """

    def __init__(self, cfg: dict):
        """
        Mitochondria 모델 초기화
        
        Parameters
        ----------
        cfg : dict
            설정 딕셔너리. 다음 키를 포함할 수 있음:
            - ATP0, Ebuf0, Heat0, CO2_0: 초기 상태값
            - k_transfer, Ploss, recover_k: 전환/손실/회복 계수
            - eta, k_glu, k_oxy, K_mO2: 효율 및 반응 계수
            - k_heat, k_co2, Heat_env, CO2_env: 환경 균형 파라미터
            - D_H, dx_heat, N: Heat 확산 파라미터 (HeatGrid 통합용)
        """
        # === 초기 상태값 ===
        self.ATP = float(cfg.get("ATP0", 100.0))       # [a.u.] - ATP 농도
        self.E_buf = float(cfg.get("Ebuf0", 80.0))     # [a.u.] - 에너지 버퍼 (NADH/FADH₂ 풀)
        self.Heat = float(cfg.get("Heat0", 0.0))       # [a.u.] - 누적 Heat (대사 부산물)
        self.CO2 = float(cfg.get("CO2_0", 0.0))        # [a.u.] - 누적 CO₂ (대사 부산물)

        # === 상수 파라미터 ===
        self.k_transfer = cfg.get("k_transfer", 0.4)     # E_buf→ATP 전환 계수 [1/ms]
        self.Ploss = cfg.get("Ploss", 1.5)               # 에너지 손실율 [a.u./ms]
        self.recover_k = cfg.get("recover_k", 8.0)       # ATP 회복 계수 [a.u./ms]
        self.recover_thresh = cfg.get("recover_thresh", 60.0)  # ATP 회복 임계값 [a.u.]
        self.ATP_clip = cfg.get("ATP_clip", (1.0, 110.0))  # ATP 클램프 범위 [a.u.]
        self.Ebuf_clip = cfg.get("Ebuf_clip", (15.0, 100.0))  # E_buf 클램프 범위 [a.u.]
        self.delta_transfer = cfg.get("delta_transfer", 5.0)  # ATP 전환 최소 차이 [a.u.]
        self.c_CO2 = cfg.get("c_CO2", 0.8)               # CO₂ 생성 계수 (ATP당 CO₂ 비율)

        # === 효율 및 반응 계수 ===
        self.eta0 = cfg.get("eta", 0.60)     # 기본 효율 (0~1, O₂ 포화 시 최대값)
        self.k_glu = cfg.get("k_glu", 0.8)   # Glucose 기여 계수 (Glycolysis 경로)
        self.k_oxy = cfg.get("k_oxy", 1.2)   # 산소 기여 계수 (ETC 경로)
        self.K_mO2 = cfg.get("K_mO2", 3.0)   # 미하엘리스-멘텐 상수 (O₂ 포화 농도 [mM])

        # === 환경 균형 파라미터 ===
        # Heat와 CO₂는 환경으로 확산/감쇠하여 균형을 유지
        self.k_heat = cfg.get("k_heat", 0.01)      # Heat 감쇠 계수 [1/ms] (환경으로의 확산율)
        self.k_co2 = cfg.get("k_co2", 0.01)        # CO₂ 감쇠 계수 [1/ms] (환경으로의 확산율)
        self.Heat_env = cfg.get("Heat_env", 0.0)   # 환경 Heat 균형값 [a.u.] (목표 Heat 수준)
        self.CO2_env = cfg.get("CO2_env", 0.0)     # 환경 CO₂ 균형값 [a.u.] (목표 CO₂ 수준)
        
        # === Heat 확산 파라미터 (확장형) ===
        # Heat는 1차원 공간(축삭 길이)을 따라 확산됨
        self.D_H = cfg.get("D_H", 0.0)             # Heat 확산 계수 [cm^2/ms] (0이면 확산 없음)
        self.dx_heat = cfg.get("dx_heat", 1.0e-3)  # 공간 간격 [cm] (HeatGrid 그리드 간격)
        
        # === HeatGrid 통합 (내부 관리) ===
        # HeatGrid는 1차원 열 확산을 모델링하는 보조 클래스
        # Mitochondria의 Heat 생성을 공간적으로 확산시킴
        self.heatgrid = HeatGrid(
            N=cfg.get("N", 121),                    # 그리드 노드 수 (축삭 세그먼트 수와 동일)
            dx=cfg.get("dx_heat", 1e-3),            # 공간 간격 [cm]
            D_H=cfg.get("D_H", 1e-6),               # Heat 확산 계수 [cm^2/ms]
            k_heat=cfg.get("k_heat", 0.01),         # Heat 감쇠 계수 [1/ms]
            H_env=cfg.get("Heat_env", 0.0)          # 환경 Heat 균형값 [a.u.]
        )

        # === 내부 상태 기록용 (디버깅 및 모니터링) ===
        self.last_eta = self.eta0                   # 마지막 스텝의 효율
        self.last_Pin = 0.0                         # 마지막 스텝의 에너지 유입량
        self.last_dATP = 0.0                        # 마지막 스텝의 ATP 생성량
        
        # === 현재 스텝에서 실제 사용된 효율을 기록 ===
        # eta는 O₂ 농도와 피드백에 따라 동적으로 변함
        self.eta = float(self.eta0)                 # 현재 효율 (동적 업데이트됨)

    # ---------------------------------------------------------
    # η(O₂): 산소 농도에 따른 효율
    # ---------------------------------------------------------
    def eta_dynamic(self, O2: float) -> float:
        """
        산소 농도에 따른 효율 계산 (Michaelis-Menten kinetics)
        
        산소 농도가 높을수록 효율이 증가하며, O₂ 포화 시 최대 효율(η₀)에 도달.
        
        Parameters
        ----------
        O2 : float
            산소 농도 [mM]
            
        Returns
        -------
        float
            효율 η (0.05 ~ η₀)
            
        수식
        ----
        η(O₂) = η₀ · (O₂ / (O₂ + K_mO₂))
        - O₂ = 0: 최소 효율 (0.05)
        - O₂ → ∞: 최대 효율 (η₀)
        - K_mO₂: 반포화 농도 (O₂ = K_mO₂일 때 η = η₀/2)
        """
        if O2 <= 0:
            return 0.05  # 산소 없을 때 최소 효율
        eta = self.eta0 * (O2 / (O2 + self.K_mO2))
        return float(np.clip(eta, 0.05, self.eta0))  # 안전 범위로 제한

    # ---------------------------------------------------------
    # P_in(Glu,O₂): 에너지 유입량
    # ---------------------------------------------------------
    def power_input(self, Glu: float, O2: float) -> float:
        """
        에너지 유입량 계산 (Glucose + O₂ 기반)
        
        두 가지 대사 경로를 합산:
        - Glycolysis (Glucose): 빠르지만 저효율
        - Electron Transport Chain (O₂): 느리지만 고효율
        
        Parameters
        ----------
        Glu : float
            혈중 Glucose 농도 [mM]
        O2 : float
            산소 농도 [mM]
            
        Returns
        -------
        float
            에너지 유입량 P_in [a.u./ms] (0~50 범위로 제한)
            
        수식
        ----
        P_in = k_glu·Glu + k_oxy·O₂
        """
        Pin = self.k_glu * Glu + self.k_oxy * O2
        return float(np.clip(Pin, 0.0, 50.0))  # 상한선 설정 (수치 안정성)

    # ---------------------------------------------------------
    # STEP: ATP 생성/소비 루프
    # ---------------------------------------------------------
    def step(self, dt: float, Glu: float, O2: float, J_use: float = 0.0):
        """
        한 스텝(dt) 동안의 ATP, E_buf, Heat, CO₂ 갱신.
        
        실행 순서:
        1. 에너지 유입 및 효율 계산 (Glucose + O₂)
        2. E_buf 축적 (에너지 저장 풀)
        3. E_buf → ATP 변환 (효율 적용)
        4. Heat/CO₂ 생성 (대사 부산물)
        5. Heat 확산 (HeatGrid 통합)
        6. CO₂ 감쇠 (환경 균형)
        7. ATP 소비 (J_use)
        8. ATP 회복 (낮을 때)
        9. 안정화 (클램프)

        Parameters
        ----------
        dt : float
            시간 스텝 [ms]
        Glu : float
            혈중 Glucose 농도 [mM]
        O2 : float
            산소 농도 [mM]
        J_use : float, optional
            ATP 소비율 [a.u./ms] (Na/K 펌프, Ca 펌프 등). 기본값: 0.0
            
        Returns
        -------
        dict
            현재 스텝의 상태를 담은 딕셔너리:
            - "ATP": ATP 농도 [a.u.]
            - "E_buf": 에너지 버퍼 [a.u.]
            - "Heat": Heat 값 [a.u.] (HeatGrid.H[0])
            - "CO2": CO₂ 농도 [a.u.]
            - "eta": 효율 (0~1)
            - "Pin": 에너지 유입량 [a.u./ms]
            - "dATP_prod": ATP 생성량 [a.u.]
        """
        # (1) 에너지 유입 및 효율 계산
        # Glucose와 O₂로부터 에너지 유입량 계산
        Pin = self.power_input(Glu, O2)
        # O₂ 농도에 따른 효율 계산 (Michaelis-Menten)
        eta = self.eta_dynamic(O2)
        # 상태 기록 (디버깅/모니터링용)
        self.last_Pin = Pin
        self.last_eta = eta
        self.eta = float(eta)  # 현재 스텝에서 실제 사용된 효율을 기록

        # (2) E_buf 축적
        # 에너지 유입량에서 손실을 뺀 순 유입량을 E_buf에 축적
        # E_buf는 NADH, FADH₂, 전자전달계 에너지 풀에 해당
        dEbuf = (Pin - self.Ploss) * dt
        self.E_buf += dEbuf

        # (3) E_buf → ATP 변환
        # E_buf가 ATP보다 충분히 높을 때만 전환 (delta_transfer 임계값)
        # 효율(η)을 적용하여 실제 ATP 생성량 계산
        dATP_prod = 0.0
        if self.E_buf > self.ATP + self.delta_transfer:
            dATP = self.k_transfer * (self.E_buf - self.ATP) * dt  # 전환량
            dATP_prod = eta * dATP  # 효율 적용 (1-η만큼은 Heat로 손실)
            self.ATP += dATP_prod
            self.E_buf -= dATP  # E_buf에서 전환된 만큼 차감
            self.last_dATP = dATP_prod

        # (4) Heat/CO₂ 생성 (대사 부산물)
        # ATP 생성 과정에서 발생하는 부산물
        # Heat = (1 - η) * dATP_prod (효율 손실분)
        # CO₂ = c_CO₂ * dATP_prod (대사 산물)
        if dATP_prod > 0.0:
            self.Heat += (1.0 - eta) * dATP_prod
            self.CO2  += self.c_CO2 * dATP_prod

        # (5) Heat 확산 자동 호출 (HeatGrid 통합)
        # Heat는 1차원 공간(축삭 길이)을 따라 확산됨
        # HeatGrid는 공간적 확산을 모델링하는 보조 클래스
        if dATP_prod > 0.0:
            # 첫 번째 위치(인덱스 0)에 Heat 소스 추가
            self.heatgrid.add_source(0, (1.0 - eta) * dATP_prod)
        # Heat 확산 계산 (CFL 조건 준수 서브스텝 포함)
        self.heatgrid.step(dt)
        # Heat 값은 HeatGrid의 첫 번째 위치 값으로 업데이트
        self.Heat = float(self.heatgrid.H[0])
        
        # (5.5) CO₂ 감쇠 (환경 균형)
        # CO₂는 환경으로 확산하여 점진적으로 감소
        # 지수 감쇠: CO₂ → CO₂_env로 수렴
        self.CO2 -= self.k_co2 * (self.CO2 - self.CO2_env) * dt
        self.CO2 = max(self.CO2, 0.0)  # 음수 방지

        # (6) ATP 소비
        # Na/K 펌프, Ca 펌프 등에서 소비되는 ATP
        if J_use > 0.0:
            self.ATP -= J_use * dt

        # (7) ATP 회복 메커니즘
        # ATP가 임계값 이하로 떨어지면 자동 회복
        # 회복 속도는 ATP 수준에 반비례 (낮을수록 빠르게 회복)
        if self.ATP < self.recover_thresh:
            self.ATP += self.recover_k * (1 - self.ATP / 100.0) * dt

        # (8) 안정화 (수치 안정성)
        # 모든 변수를 안전 범위로 제한
        self.ATP = float(np.clip(self.ATP, *self.ATP_clip))
        self.E_buf = float(np.clip(self.E_buf, *self.Ebuf_clip))
        self.CO2 = max(self.CO2, 0.0)  # CO₂는 0 이상
        self.Heat = max(self.Heat, 0.0)  # Heat는 0 이상

        return {
            "ATP": self.ATP,
            "E_buf": self.E_buf,
            "Heat": self.Heat,
            "CO2": self.CO2,
            "eta": eta,
            "Pin": Pin,
            "dATP_prod": dATP_prod,
        }


# =============================================================
# 2-1. Heat Diffusion Grid (1D Spatial)
# =============================================================
# 참고: 섹션 2-1은 Mitochondria(섹션 2)의 Heat 확산을 처리하는
# 보조 클래스로, Mitochondria와 밀접하게 연동되므로 2-1로 번호를 매김.
# 독립 클래스이지만 기능적으로 Mitochondria의 확장 모듈 역할.
# =============================================================

class HeatGrid:
    """
    1차원 열 확산(Heat diffusion) 모델
    
    축삭 길이를 따라 Heat가 공간적으로 확산되는 현상을 모델링.
    ATP 생성 과정에서 발생한 Heat가 축삭을 따라 퍼져나감.
    
    미분 방정식:
        ∂H/∂t = D_H·∇²H − k_heat·(H−H_env)
        
    여기서:
        - H: Heat 값 [a.u.]
        - D_H: Heat 확산 계수 [cm^2/ms]
        - k_heat: Heat 감쇠 계수 [1/ms]
        - H_env: 환경 Heat 균형값 [a.u.]
        
    특징:
        - CFL 조건을 자동으로 확인하여 안정적인 서브스텝 사용
        - D_H = 0인 경우 확산 없이 감쇠만 계산
        - Neumann 경계 조건 사용 (∂H/∂x = 0)
    """
    def __init__(self, N=121, dx=1.0e-3, D_H=1e-6, k_heat=0.01, H_env=0.0):
        """
        HeatGrid 초기화
        
        Parameters
        ----------
        N : int, optional
            그리드 노드 수 (축삭 세그먼트 수와 동일). 기본값: 121
        dx : float, optional
            공간 간격 [cm]. 기본값: 1.0e-3
        D_H : float, optional
            Heat 확산 계수 [cm^2/ms]. 기본값: 1e-6 (0이면 확산 없음)
        k_heat : float, optional
            Heat 감쇠 계수 [1/ms]. 기본값: 0.01
        H_env : float, optional
            환경 Heat 균형값 [a.u.]. 기본값: 0.0
        """
        self.N = N                  # 그리드 노드 수
        self.dx2 = dx * dx          # 공간 간격의 제곱 (Laplacian 계산용)
        self.D_H = D_H              # Heat 확산 계수 [cm^2/ms]
        self.k_heat = k_heat        # Heat 감쇠 계수 [1/ms]
        self.H_env = H_env          # 환경 Heat 균형값 [a.u.]
        self.H = np.zeros(N)        # Heat 값 배열 [a.u.]

    def add_source(self, idx: int, q: float):
        """특정 위치에 열(Heat) 발생량 추가"""
        if 0 <= idx < self.N:
            self.H[idx] += q

    def step(self, dt: float):
        """시간 적분으로 열 확산 계산 (CFL 조건 준수)"""
        # D_H = 0인 경우 확산 없이 감쇠만
        if self.D_H <= 0:
            self.H += -(self.H - self.H_env) * (1 - np.exp(-self.k_heat * dt))
            self.H[self.H < 0] = 0.0
            return self.H
        
        # CFL 조건: dt ≤ dx²/(2·D_H)
        dt_cfl = 0.9 * self.dx2 / (2.0 * self.D_H)
        n_sub = max(1, int(np.ceil(dt / dt_cfl)))
        dt_sub = dt / n_sub
        
        # 서브스텝으로 안정적 적분
        for _ in range(n_sub):
            lap = np.zeros_like(self.H)
            lap[1:-1] = (self.H[:-2] - 2*self.H[1:-1] + self.H[2:]) / self.dx2
            lap[0]  = 2*(self.H[1] - self.H[0]) / self.dx2   # Neumann BC
            lap[-1] = 2*(self.H[-2] - self.H[-1]) / self.dx2
            dHdt = self.D_H * lap - self.k_heat * (self.H - self.H_env)
            self.H += dt_sub * dHdt
        
        self.H[self.H < 0] = 0.0
        return self.H


# =============================================================
# 🔬 TEST SIMULATION (standalone)
# =============================================================
if __name__ == "__main__":
    cfg = {
        "ATP0": 90.0,
        "Ebuf0": 70.0,
        "eta": 0.65,
        "k_transfer": 0.4,
        "Ploss": 1.2,
        "recover_thresh": 60.0,
        "recover_k": 5.0,
        "k_glu": 0.8,
        "k_oxy": 1.2,
        "K_mO2": 3.0,
        "c_CO2": 0.9,
    }

    mito = Mitochondria(cfg)
    Glu = 5.0   # [mM]
    O2 = 5.0    # [mM]
    dt = 0.1
    print("[Bioenergy Simulation — Mitochondria Engine]")
    print(" t(ms) |  ATP   |  E_buf |  η  | Heat | CO₂ | Pin ")

    for t in range(50):
        out = mito.step(dt, Glu, O2, J_use=0.05)
        print(f"{t*dt:6.1f} | {out['ATP']:6.2f} | {out['E_buf']:6.2f} | "
              f"{out['eta']:5.3f} | {out['Heat']:5.2f} | {out['CO2']:5.2f} | {out['Pin']:5.2f}")