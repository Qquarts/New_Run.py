# =============================================================
# ionflow_dynamics.py — 다중 이온 확산/전기이동 모델
# =============================================================
# 목적:
#   • 막전위(Vm)에 따라 Na⁺, K⁺, Ca²⁺, Cl⁻의 이동 계산
#   • 전기장(∇V)에 따른 drift + 확산(diffusion)을 반영
#   • 전하 중립 원칙 보존
#
# 주요 기능:
#   - 다중 이온 확산 모델링 (Na⁺, K⁺, Ca²⁺, Cl⁻)
#   - 전기장에 의한 이온 drift 효과
#   - 전하 중립 보정 (charge neutrality correction)
#   - Neumann 경계 조건 적용
#
# 수식:
#   (1) 이온 농도 변화식
#       ∂C_i/∂t = D_i∇²C_i − μ_i·z_i·F·∇V
#       - D_i: 확산 계수 [cm²/s]
#       - μ_i: 이동도
#       - z_i: 전하수 (+1, +2, -1 등)
#       - F: 패러데이 상수 [C/mol]
#       - ∇V: 전기장 [mV/cm]
#
#   (2) 이온 전류
#       j_i = −D_i∇C_i + μ_i·z_i·C_i·∇V
#
#   (3) 전하 보존
#       Σ z_i·C_i ≈ 0 (전하 중립 원칙)
#
# 단위:
#   - dt: [ms] (밀리초)
#   - dx: [cm] (센티미터)
#   - C: [mM] (밀리몰)
#   - V: [mV] (밀리볼트)
#   - D: [cm²/s] (확산 계수)
# =============================================================

import numpy as np

class IonFlowDynamics:
    r"""
    IonFlowDynamics — Multi-Ion Diffusion + Electric Drift
    ------------------------------------------------------
    ∂C_i/∂t = D_i∇²C_i − μ_i·z_i·F·∇V
    
    단위:
        - dt: [ms] (밀리초)
        - dx: [cm] (센티미터)
        - C: [mM] (밀리몰)
        - V: [mV] (밀리볼트)
        - D: [cm²/s] (확산 계수)
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
        """
        self.N = cfg.get("N", 121)
        self.dx = cfg.get("dx", 1e-3)  # [cm]
        self.V = np.full(self.N, cfg.get("Vrest", -70.0))  # [mV]
        self.F = 96485.0  # 패러데이 상수 [C/mol] (전하량 단위 변환)
        # 이동도 스케일: drift 항의 과도한 영향 완화 (장기 안정성 향상)
        # 작은 값: drift 효과 감소 → 확산 중심, 안정적
        # 큰 값: drift 효과 증가 → 전기장 영향 강화, 불안정 가능
        self.mu_scale = 1e-9  # [PATCH] 1e-8 → 1e-9 (장기 시뮬레이션 안정성 강화)

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
        self.ions = {
            "Na": {"C": np.full(self.N, 15.0), "D": 1.33e-5, "z": +1},  # Na⁺: 내부 농도 ~15 mM
            "K":  {"C": np.full(self.N,140.0), "D": 1.96e-5, "z": +1},  # K⁺: 내부 농도 ~140 mM
            "Ca": {"C": np.full(self.N, 0.0001), "D": 0.79e-5, "z": +2}, # Ca²⁺: 내부 농도 ~0.0001 mM (매우 낮음)
            "Cl": {"C": np.full(self.N, 5.0), "D": 2.03e-5, "z": -1},   # Cl⁻: 내부 농도 ~5 mM
        }

    def laplacian(self, arr):
        """
        1D 중심차분 ∇²C (Neumann 경계 조건 포함)
        
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

    def step(self, dt: float):
        """
        한 스텝 이온 농도 업데이트
        
        Parameters
        ----------
        dt : float
            시간 스텝 [ms]
            
        Returns
        -------
        dict
            각 이온의 농도 배열을 담은 딕셔너리
            
        알고리즘
        --------
        1. 확산 항 계산: D_i·∇²C_i
        2. Drift 항 계산: -μ_i·z_i·F·∇V·C_i
        3. 이온 농도 업데이트
        4. 전하 중립 보정
        5. 음수 농도 방지 (클램프)
        """
        # (1) 전기장 계산
        # 전기장은 전위의 공간 기울기: ∇V = dV/dx
        dVdx = np.gradient(self.V, self.dx)  # 전기장 [mV/cm]
        
        # (2) 각 이온별 확산 및 drift 계산
        for ion, d in self.ions.items():
            D, z, C = d["D"], d["z"], d["C"]
            
            # 확산 항: D_i·∇²C_i
            # Fick의 법칙: 농도 구배에 따라 확산
            diff = D * self.laplacian(C)
            
            # Drift 항: -μ_i·z_i·F·∇V·C_i
            # 전기장에 의한 이온 이동 (양전하는 음극으로, 음전하는 양극으로)
            # z > 0 (양전하): 음극 방향 이동 (∇V < 0일 때 양의 drift)
            # z < 0 (음전하): 양극 방향 이동 (∇V > 0일 때 양의 drift)
            drift = -self.mu_scale * z * self.F * dVdx * C
            
            # 이온 농도 업데이트 (Euler 적분)
            # ∂C_i/∂t = diff + drift
            C += dt * (diff + drift)
            
            # 음수 농도 방지 (물리적 제약)
            d["C"] = np.clip(C, 0.0, None)

        # (3) 전하 중립 보정
        # 전하 중립 원칙: Σ z_i·C_i ≈ 0
        # 수치 오차로 인해 전하가 축적될 수 있으므로 보정
        total_q = sum(d["z"]*np.sum(d["C"]) for d in self.ions.values())
        if abs(total_q) > 1e-3:  # 임계값 초과 시 보정
            # 보정량 계산: 총 전하를 이온 수와 노드 수로 나눔
            corr = -total_q / (self.N * len(self.ions))
            for ion, d in self.ions.items():
                # 전하수에 따라 보정 (양전하는 감소, 음전하는 증가)
                d["C"] += corr * np.sign(d["z"])
                # [PATCH] 전하 중립 보정 후 추가 클램프
                # 기능: 전하 중립 보정으로 인해 음수 농도가 발생할 수 있으므로 0 이상으로 제한
                # 효과: 이온 농도가 음수가 되는 것을 방지하여 안정성 향상
                d["C"] = np.clip(d["C"], 0.0, None)  # 음수 농도 방지

        return {ion: d["C"] for ion, d in self.ions.items()}


# =============================================================
# 🔬 단독 테스트 (Standalone Test)
# =============================================================
# 이 모듈을 직접 실행할 때 테스트 코드가 실행됩니다.
# 간단한 이온 확산 시뮬레이션을 수행하여 동작을 확인합니다.
# =============================================================
if __name__ == "__main__":
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
    ion.V[:] = np.linspace(-70, 50, ion.N)
    
    # 시뮬레이션 실행
    print("=" * 60)
    print("IonFlowDynamics Test — Multi-Ion Diffusion + Drift")
    print("=" * 60)
    print("t(ms)    | [Na⁺] (mM) | [K⁺] (mM)")
    print("-" * 60)
    
    for t in range(5):
        res = ion.step(dt=0.01)  # 0.01 ms 스텝으로 5회 반복
        # 중간 노드(인덱스 60)의 Na⁺, K⁺ 농도 출력
        print(f"{t*0.01:.3f} ms | {res['Na'][60]:10.3f} | {res['K'][60]:10.3f}")
    
    print("=" * 60)
    print("Test completed ✅")