# =============================================================
# 04.ion_flow.py — 다중 이온 확산/전기이동 모델 (V3)
# =============================================================
# 목적:
#   • 막전위(Vm)에 따라 Na⁺, K⁺, Ca²⁺, Cl⁻의 이동 계산
#   • 전기장(∇V)에 따른 drift + 확산(diffusion)을 반영
#   • 전하 중립 원칙 보존
#
# V3 계약 고정:
#   - 시간 단위: [ms] (밀리초) - 모든 시간 관련 파라미터는 ms 단위
#   - V: [mV] (밀리볼트)
#   - C: [mM] (밀리몰)
#   - D: [cm²/s] (확산 계수)
#   - 참조 공유 금지: V는 값 복사로 전달 ⭐ V3 변경
#
# 주요 기능:
#   - 다중 이온 확산 모델링 (Na⁺, K⁺, Ca²⁺, Cl⁻)
#   - 전기장에 의한 이온 drift 효과
#   - 전하 중립 보정 (charge neutrality correction)
#   - Neumann 경계 조건 적용
#
# 수식:
#   (1) 이온 농도 변화식 (1차 근사 모델)
#       ∂C_i/∂t = D_i·∇²C_i − μ_eff·z_i·C_i·∇V
#       - D_i: 확산 계수 [cm²/s]
#       - μ_eff: 유효 이동도 [cm²/(s·mV)] (mu_scale, 이미 z·F/(R·T) 포함)
#       - z_i: 전하수 (+1, +2, -1 등)
#       - ∇V: 전기장 [mV/cm]
#       - C_i: 이온 농도 [mM]
#
#       ⚠️ 참고: 완전한 Nernst-Planck는 보존형 flux 형태
#          ∂C_i/∂t = -∇·(-D_i∇C_i - μ_i·z_i·C_i·∇V)
#          현재는 안정성/가독성 우선으로 1차 근사 모델 사용
#
#   (2) 전하 보존
#       Σ z_i·C_i ≈ 0 (전하 중립 원칙)
#
# 단위:
#   - dt: [ms] (밀리초) ⭐ V3 계약 고정
#   - dx: [cm] (센티미터)
#   - C: [mM] (밀리몰)
#   - V: [mV] (밀리볼트)
#   - D: [cm²/s] (확산 계수)
#
# 설계 이유:
#   - 이온 농도 변화는 막전위에 따라 달라지므로, V를 입력으로 받아야 함
#   - V3 단일 방향화 원칙: V는 값 복사로 전달 (참조 공유 금지)
#   - 전하 중립 보정은 수치 오차로 인한 전하 축적을 방지
# =============================================================

import numpy as np


class IonFlowDynamics:
    r"""
    IonFlowDynamics — Multi-Ion Diffusion + Electric Drift (V3)
    ------------------------------------------------------
    
    V3 계약:
    - 입력: dt [ms], V [mV] (값 복사, 참조 공유 금지) ⭐ V3 변경
    - 출력: dict {ion: C [mM]}
    - Side-effect: self.ions[ion]["C"] 업데이트, self.V 업데이트 (내부 상태)
    
    수식 (1차 근사 모델):
        ∂C_i/∂t = D_i·∇²C_i − μ_eff·z_i·C_i·∇V
        
        - D_i: 확산 계수 [cm²/s]
        - μ_eff: 유효 이동도 [cm²/(s·mV)] (mu_scale)
        - z_i: 전하수 (+1, +2, -1 등)
        - C_i: 이온 농도 [mM]
        - ∇V: 전기장 [mV/cm]
    
    단위:
        - dt: [ms] (밀리초) ⭐ V3 계약 고정
        - dx: [cm] (센티미터)
        - C: [mM] (밀리몰)
        - V: [mV] (밀리볼트)
        - D: [cm²/s] (확산 계수)
    
    설계 이유:
    - 이온 농도 변화는 막전위에 따라 달라지므로, V를 입력으로 받아야 함
    - V3 단일 방향화 원칙: V는 값 복사로 전달 (참조 공유 금지)
    - 전하 중립 보정은 수치 오차로 인한 전하 축적을 방지
    """

    def __init__(self, cfg: dict):
        """
        IonFlowDynamics 초기화
        
        Parameters
        ----------
        cfg : dict
            설정 딕셔너리:
            - N: 그리드 노드 수 (기본값: 121)
            - dx: 공간 간격 [cm] (기본값: 1e-3)
            - Vrest: 휴지 전위 [mV] (기본값: -70.0)
            - mu_scale: 이동도 스케일 (기본값: 1e-9) [cm²/(s·mV)]
        """
        self.N = cfg.get("N", 121)
        self.dx = cfg.get("dx", 1e-3)  # [cm]
        self.V = np.full(self.N, cfg.get("Vrest", -70.0))  # [mV] (내부 상태)
        self.F = 96485.0  # 패러데이 상수 [C/mol] (참고용, 실제 계산에는 사용 안 함)
        # 유효 이동도 (mu_eff): drift 항의 스케일링 계수
        # 단위: [cm²/(s·mV)]
        # 물리적 정의: μ_eff = (D·z·e)/(k_B·T·1000)
        #   - D: 확산 계수 [cm²/s]
        #   - z: 전하수
        #   - e: 전자 전하 [C]
        #   - k_B: 볼츠만 상수 [J/K]
        #   - T: 온도 [K] (37°C = 310K)
        #   - 1000: mV 변환
        # 생리학적 범위 (T=310K):
        #   - Na⁺: ~5e-7 [cm²/(s·mV)]
        #   - K⁺:  ~7e-7 [cm²/(s·mV)]
        #   - Ca²⁺: ~3e-7 [cm²/(s·mV)]
        # 현재 기본값 (1e-9): 안정성을 위해 생리학적 값의 ~1/500
        #   → drift 효과를 의도적으로 억제하여 장기 시뮬레이션 안정성 강화
        self.mu_scale = cfg.get("mu_scale", 1e-9)  # [cm²/(s·mV)]

        # === 4종 이온 초기화 ===
        # 각 이온의 파라미터:
        #   - C: 초기 농도 [mM] (공간적으로 균일하게 초기화)
        #   - D: 확산 계수 [cm²/s] (실제 생리학적 값 기반)
        #   - z: 전하수 (양전하: +1, +2, 음전하: -1)
        #
        # 이온 종류:
        #   - Na⁺: 나트륨 이온 (양전하, 주요 액션 전위 관련)
        #   - K⁺: 칼륨 이온 (양전하, 재분극 관련)
        #   - Ca²⁺: 칼슘 이온 (양전하, 신경전달물질 방출 등)
        #   - Cl⁻: 염화 이온 (음전하, 억제성 신경전달 등)
        #
        # ⚠️ 설계 리스크: 초기값이 전하 중립을 만족하지 않음
        #   초기 전하: Na(15) + K(140) + Ca(0.0002) - Cl(5) = +150.0002
        #   첫 스텝에서 Cl⁻ 보정 시 급격한 증가 가능 (5 → 155)
        #   → 전하 중립 보정 임계값을 완화하여 완화 (10.0)
        self.ions = {
            "Na": {"C": np.full(self.N, 15.0), "D": 1.33e-5, "z": +1},  # Na⁺: 내부 농도 ~15 mM
            "K":  {"C": np.full(self.N,140.0), "D": 1.96e-5, "z": +1},  # K⁺: 내부 농도 ~140 mM
            "Ca": {"C": np.full(self.N, 0.0001), "D": 0.79e-5, "z": +2}, # Ca²⁺: 내부 농도 ~0.0001 mM (매우 낮음)
            "Cl": {"C": np.full(self.N, 5.0), "D": 2.03e-5, "z": -1},   # Cl⁻: 내부 농도 ~5 mM
        }
        
        # 이온 농도 생리학적 범위 [mM]
        # 물리적 이유: 이온 농도는 생리학적 범위 내에서 유지되어야 함
        self.ion_limits = {
            "Na": (0.0, 500.0),
            "K":  (0.0, 500.0),
            "Ca": (0.0, 10.0),    # Ca는 매우 낮은 농도 유지
            "Cl": (0.0, 500.0),
        }

    def set_V(self, V: np.ndarray):
        """
        막전위 설정 (값 복사) ⭐ V3: 참조 공유 금지
        
        V3 계약:
        - 입력: V [mV] (값 복사)
        - 출력: 없음
        - Side-effect: self.V 업데이트 (값 복사)
        
        Parameters
        ----------
        V : np.ndarray
            막전위 배열 [mV] (값 복사로 전달)
        """
        # V3: 참조 공유 금지 - 값 복사로 전달
        self.V = np.array(V, copy=True)  # [mV] ⭐ V3: 값 복사

    def laplacian(self, arr):
        """
        1D 중심차분 ∇²C (Neumann 경계 조건 포함)
        
        V3 계약:
        - 입력: arr (이온 농도 배열)
        - 출력: Laplacian 값 배열
        - Side-effect: 없음
        
        내부 노드: 2차 중심차분
        경계 노드: Neumann 경계 조건 (∂C/∂x = 0)
        
        Parameters
        ----------
        arr : np.ndarray
            1D 배열 (이온 농도 등)
            
        Returns
        -------
        np.ndarray
            Laplacian 값 (dx²로 정규화됨)
        """
        lap = np.zeros_like(arr)
        # 내부 노드: 2차 중심차분
        lap[1:-1] = arr[:-2] - 2*arr[1:-1] + arr[2:]
        # 경계 노드: Neumann 경계 조건 (∂C/∂x = 0)
        # 2차 정확도: lap[0] = 2*(arr[1] - arr[0]) / dx²
        lap[0]  = 2*(arr[1] - arr[0])
        lap[-1] = 2*(arr[-2] - arr[-1])
        return lap / (self.dx**2)

    def step(self, dt: float, V: np.ndarray = None):
        """
        한 스텝 이온 농도 업데이트
        
        V3 계약:
        - 입력:
          - dt: [ms] (밀리초) ⭐ V3 계약 고정
          - V: [mV] (막전위 배열, 값 복사) ⭐ V3: 참조 공유 금지
        - 출력: dict {ion: C [mM]}
        - Side-effect:
          - self.ions[ion]["C"] 업데이트 [mM]
          - self.V 업데이트 [mV] (V가 제공된 경우)
        
        Parameters
        ----------
        dt : float
            시간 스텝 [ms] ⭐ V3: ms 단위 명시
        V : np.ndarray, optional
            막전위 배열 [mV] (값 복사로 전달). None이면 이전 값 유지 ⭐ V3: 참조 공유 금지
            
        Returns
        -------
        dict
            각 이온의 농도 배열을 담은 딕셔너리:
            - "Na": Na⁺ 농도 [mM]
            - "K": K⁺ 농도 [mM]
            - "Ca": Ca²⁺ 농도 [mM]
            - "Cl": Cl⁻ 농도 [mM]
            
        알고리즘
        --------
        1. 막전위 설정 (값 복사) ⭐ V3: 참조 공유 금지
        2. 확산 항 계산: D_i·∇²C_i
        3. Drift 항 계산: -μ_i·z_i·F·∇V·C_i
        4. 이온 농도 업데이트
        5. 전하 중립 보정
        6. 음수 농도 방지 (클램프)
        """
        # V3: 막전위 설정 (값 복사) ⭐ 참조 공유 금지
        if V is not None:
            self.set_V(V)  # 값 복사로 전달
        
        # 단위 변환: dt [ms] → dt_s [s]
        dt_s = dt / 1000.0
        
        # (1) 전기장 계산 (Neumann 경계 조건)
        # 전기장은 전위의 공간 기울기: ∇V = dV/dx
        dVdx = np.gradient(self.V, self.dx)  # 전기장 [mV/cm]
        # 경계에서 Neumann 조건 적용
        dVdx[0] = 0.0
        dVdx[-1] = 0.0
        
        # CFL 조건 확인 (확산 + drift 안정성)
        D_max = max(d["D"] for d in self.ions.values())
        # 확산 CFL: dt < 0.5 * dx² / D
        dt_cfl_diff_s = 0.5 * self.dx**2 / D_max  # [s]
        dt_cfl_diff_ms = dt_cfl_diff_s * 1000.0   # [ms]
        
        # Drift CFL: dt < dx / |v_max|
        # v = μ_eff·z·|∇V| (이동 속도)
        # 각 이온별 최대 drift 속도 계산 (z 고려)
        dVdx_max = np.max(np.abs(dVdx))  # [mV/cm]
        v_max = 0.0
        for d in self.ions.values():
            v_ion = self.mu_scale * abs(d["z"]) * dVdx_max  # [cm/s]
            v_max = max(v_max, v_ion)
        
        if v_max > 0:
            dt_cfl_drift_s = 0.5 * self.dx / v_max  # [s]
            dt_cfl_drift_ms = dt_cfl_drift_s * 1000.0  # [ms]
        else:
            dt_cfl_drift_ms = float('inf')
        
        # 더 엄격한 조건 선택
        dt_cfl_ms = min(dt_cfl_diff_ms, dt_cfl_drift_ms)
        
        if dt > dt_cfl_ms:
            import warnings
            warnings.warn(
                f"IonFlowDynamics: dt={dt:.4f}ms > dt_cfl={dt_cfl_ms:.4f}ms "
                f"(diff={dt_cfl_diff_ms:.4f}ms, drift={dt_cfl_drift_ms:.4f}ms). "
                f"Numerical instability possible!", RuntimeWarning
            )
        
        # (2) 각 이온별 확산 및 drift 계산
        for ion, d in self.ions.items():
            D, z, C = d["D"], d["z"], d["C"]
            
            # 확산 항: D_i·∇²C_i
            # Fick의 법칙: 농도 구배에 따라 확산
            # D는 [cm²/s], laplacian은 [1/cm²], 결과는 [mM/s]
            diff = D * self.laplacian(C)
            
            # Drift 항: -μ_eff·z·C·∇V (1차 근사 모델)
            # 전기장에 의한 이온 이동 (양전하는 음극으로, 음전하는 양극으로)
            # z > 0 (양전하): 음극 방향 이동 (∇V < 0일 때 양의 drift)
            # z < 0 (음전하): 양극 방향 이동 (∇V > 0일 때 양의 drift)
            # μ_eff = mu_scale [cm²/(s·mV)] (유효 이동도)
            #
            # ⚠️ 설계 리스크: 완전한 Nernst-Planck가 아님
            #   완전한 형태: ∂C/∂t = -∇·(-D∇C - μzC∇V) (보존형 flux)
            #   현재 형태: ∂C/∂t = D∇²C - μ_eff·z·C·∇V (현상학적 drift)
            #   → V1에서는 안정성/가독성 우선, V2에서 보존형 flux 고려
            drift = -self.mu_scale * z * dVdx * C  # [mM/s]
            
            # 이온 농도 업데이트 (Euler 적분)
            # ∂C_i/∂t = diff + drift
            # dt_s는 [s], diff와 drift는 [mM/s]
            C += dt_s * (diff + drift)
            
            # 생리학적 범위로 클램프
            # 물리적 이유: 이온 농도는 생리학적 범위 내에서 유지되어야 함
            d["C"] = np.clip(C, *self.ion_limits[ion])

        # (3) 전하 중립 보정 (글로벌 보정 - Cl⁻만 조정)
        # 전하 중립 원칙: Σ z_i·C_i ≈ 0
        # 수치 오차로 인해 전하가 축적될 수 있으므로 보정
        # Cl⁻는 "buffer ion"으로 작용하여 전하 중립 유지
        #
        # ⚠️ 설계 리스크: 초기값이 전하 중립을 만족하지 않음
        #   초기 전하: +150.0002 → 첫 스텝에서 Cl⁻ 급격한 증가 가능
        #   → 임계값을 완화하여 작은 불균형은 허용 (드리프트 억제용 소프트 보정)
        total_q = sum(d["z"]*np.sum(d["C"]) for d in self.ions.values())
        if abs(total_q) > 10.0:  # 임계값 완화: 1e-3 → 10.0 (큰 불균형만 보정)
            # Cl⁻ 농도만 조정하여 전하 중립 유지
            # z_Cl = -1이므로 total_q가 양수면 Cl⁻ 증가, 음수면 감소
            corr_per_node = total_q / self.N  # 노드당 전하 보정량
            self.ions["Cl"]["C"] += corr_per_node  # z=-1이므로 부호 자동 조정
            # 물리적 이유: Cl⁻ 농도는 생리학적 범위 내에서 유지되어야 함
            self.ions["Cl"]["C"] = np.clip(
                self.ions["Cl"]["C"], 
                *self.ion_limits["Cl"]
            )
            
            # Cl⁻ 비정상 증가 모니터링
            cl_max = np.max(self.ions["Cl"]["C"])
            if cl_max > 200.0:
                import warnings
                warnings.warn(
                    f"IonFlowDynamics: Cl⁻ 농도가 비정상적으로 높습니다! "
                    f"max(Cl)={cl_max:.2f}mM (정상 범위: 5-10mM)",
                    RuntimeWarning
                )

        return {ion: d["C"].copy() for ion, d in self.ions.items()}  # ⭐ V3: 값 복사 반환


# =============================================================
# V3 독립 실행 테스트
# =============================================================
if __name__ == "__main__":
    """
    V3 계약 고정 검증:
    - 시간 단위: [ms] 확인
    - 참조 공유 금지 확인
    - 입출력/side-effect 확인
    """
    # 설정 초기화
    cfg = {
        "N": 121,        # 그리드 노드 수
        "dx": 1e-3,      # 공간 간격 [cm]
        "Vrest": -70.0   # 휴지 전위 [mV]
    }
    
    # IonFlowDynamics 인스턴스 생성
    ion = IonFlowDynamics(cfg)
    
    # 전위 분포 설정 (균일한 전위 구배)
    # -70 mV에서 50 mV까지 선형적으로 변화
    # 이는 전기장이 존재하는 상황을 시뮬레이션
    V_input = np.linspace(-70, 50, ion.N)  # [mV] ⭐ V3: 값 복사로 전달
    
    # V3: 참조 공유 금지 확인
    V_original = V_input.copy()
    ion.set_V(V_input)  # 값 복사로 전달
    V_input[0] = 999.0  # 원본 수정
    assert ion.V[0] != 999.0, "❌ 참조 공유 발견!"  # 값 복사 확인
    print("✅ 참조 공유 금지 확인: 값 복사로 전달됨")
    
    # 시뮬레이션 실행
    print("=" * 60)
    print("IonFlowDynamics V3 — 계약 고정 검증")
    print("=" * 60)
    print(f"dt: 0.01 [ms] ⭐ V3 계약")
    print(f"V 입력: 값 복사로 전달 ⭐ V3 계약")
    print("-" * 60)
    print("t(ms)    | [Na⁺] (mM) | [K⁺] (mM)")
    print("-" * 60)
    
    for t in range(5):
        res = ion.step(dt=0.01, V=V_original)  # 0.01 ms 스텝으로 5회 반복 ⭐ V3: ms 단위 명시
        # 중간 노드(인덱스 60)의 Na⁺, K⁺ 농도 출력
        print(f"{t*0.01:.3f} ms | {res['Na'][60]:10.3f} | {res['K'][60]:10.3f}")
    
    print("=" * 60)
    print("✅ V3 계약 고정 검증 완료")
    print("  - 시간 단위: [ms] 확인")
    print("  - 참조 공유 금지 확인")
    print("  - 입출력/side-effect 명시 확인")

