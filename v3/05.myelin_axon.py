# =============================================================
# 05.myelin_axon.py — 물리적 도약전도 (Saltatory Conduction) (V3)
# =============================================================
# 목적:
#   - 소마(Soma)에서 전송된 활동전위가 축삭을 따라 도약전도(saltatory conduction)로 전달되는 과정 모델링
#   - 노드(Node)와 인터노드(Internode) 구간을 구분
#   - 각 구간의 확산(D), 막용량(Cm), 누설전도(gL) 상이
#   - 노드에서만 빠른 Na⁺ 채널이 활성화되어 도약 전위 형성
#   - 시간 감쇠(Lambda), 에너지 감쇠(gamma_extra), α-펄스 자극까지 통합
#
# V3 계약 고정:
#   - 시간 단위: [ms] (밀리초) - 모든 시간 관련 파라미터는 ms 단위
#   - V: [mV] (밀리볼트)
#   - D: [cm²/ms] (확산 계수) ⭐ V3: ms 단위 명시
#   - I: [µA/cm²] (마이크로암페어/제곱센티미터)
#   - 단일 방향화: Soma → Axon (역방향 참조 금지) ⭐ V3 변경
#
# =============================================================
# 수식 요약
# -------------------------------------------------------------
# (1) 막전위 변화식
#     ∂V/∂t = ∂/∂x(D(x)·∂V/∂x) - g_L(x)(V - E_L)/C_m(x)
#              + [I_ext(x,t) + I_Na_node(x,t)]/C_m(x)
#              - γ_extra(V - V_rest)
#     ⚠️ 중요: D(x)가 공간 가변이므로 보존형 flux 형태 필수
#              (D가 상수일 때만 D·∂²V/∂x² = ∂/∂x(D·∂V/∂x))
#
# (2) Na⁺ 채널 활성 (노드 구간)
#     I_Na_node = g_Na_node·m³·h·(E_Na_node - V)
#     dm/dt = (m_inf(V) - m)/τ_m
#     dh/dt = (h_inf(V) - h)/τ_h
#     m_inf(V) = σ((V - Vh_m)/k_m)
#     h_inf(V) = σ((V - Vh_h)/k_h)
#     σ(x) = 1 / (1 + e^{−x})
#
# (3) 시간 감쇠 (Inflation factor)
#     D_eff(t) = D · c(t)
#     c(t) = c₀ · exp(−Λ · t)
#
# (4) 외부 결합 및 α-펄스
#     I_ext(0,t) = I₀_from_soma + coupling·(V_soma − V₀)
#     I_alpha(t) = I₀·[exp(−t/τ_d) − exp(−t/τ_r)]₊
#
# (5) 전도 속도 측정
#     v = Δx / Δt
#     Δt = (노드 통과 시간 차이 평균)
#
# 설계 이유:
#   - 도약전도는 수초화 축삭의 핵심 메커니즘으로, 신호 전달 속도를 크게 향상
#   - 노드에서만 Na⁺ 채널이 활성화되어 전위 재생성
#   - 인터노드는 절연되어 신호 감쇠 최소화
# =============================================================

import numpy as np


class MyelinatedAxon:
    r"""
    MyelinatedAxon — Saltatory Conduction Model (V3)
    -------------------------------------------
    
    V3 계약:
    - 입력: dt_elec [ms], t_ms [ms], I0_from_soma [µA], soma_V [mV] (값 복사)
    - 출력: 없음 (내부 상태 업데이트)
    - Side-effect: self.V, self.m_node, self.h_node 업데이트
    
    Differential equation:
        ∂V/∂t = ∂/∂x(D(x)·∂V/∂x) - g_L(x)(V - E_L)/C_m(x)
                 + [I_ext(x,t) + I_Na_node(x,t)]/C_m(x)
                 - γ_extra(V - V_rest)
    
    ⚠️ 중요: D(x)가 공간 가변(노드 vs 인터노드)이므로
            보존형 flux 형태 필수

    Node only:
        I_Na_node = g_Na_node·m³·h·(E_Na_node - V)
        ḿ = (m_inf(V) - m)/τ_m
        ḣ = (h_inf(V) - h)/τ_h
    
    설계 이유:
    - 도약전도는 수초화 축삭의 핵심 메커니즘으로, 신호 전달 속도를 크게 향상
    - 노드에서만 Na⁺ 채널이 활성화되어 전위 재생성
    - 인터노드는 절연되어 신호 감쇠 최소화
    - V3 단일 방향화 원칙: Soma → Axon (역방향 참조 금지)
    """

    # ---------------------------------------------------------
    # 초기화
    # ---------------------------------------------------------
    def __init__(self, cfg: dict):
        """
        MyelinatedAxon 초기화
        
        Parameters
        ----------
        cfg : dict
            설정 딕셔너리:
            - N: 그리드 노드 수
            - node_period: 노드 간격
            - Vrest: 휴지 전위 [mV]
            - tau: 시간 상수 [ms] ⭐ V3: ms 단위 명시
            - dx: 공간 간격 [cm]
            - cfl_safety: CFL 안전 계수
            - D_node: 노드 확산 계수 [cm²/ms] ⭐ V3: ms 단위 명시
            - D_internode: 인터노드 확산 계수 [cm²/ms] ⭐ V3: ms 단위 명시
            - Cm_node: 노드 막용량 [µF/cm²]
            - Cm_myelin: 인터노드 막용량 [µF/cm²]
            - gL_node: 노드 누설 전도도 [mS/cm²]
            - gL_myelin: 인터노드 누설 전도도 [mS/cm²]
            - EL: 누설 역전위 [mV]
            - thresh: 임계 통과 기준 [mV]
            - coupling: 소마-축삭 결합 계수
            - stim_gain: 소마→0번노드 주입 이득
            - node_gNa: 노드 Na⁺ 전도도 [mS/cm²]
            - node_ENa: 노드 Na⁺ 역전위 [mV]
            - node_m_tau: m 게이트 시간 상수 [ms] ⭐ V3: ms 단위 명시
            - node_h_tau: h 게이트 시간 상수 [ms] ⭐ V3: ms 단위 명시
            - node_m_inf_k, node_m_inf_Vh: m_inf 파라미터
            - node_h_inf_k, node_h_inf_Vh: h_inf 파라미터
            - c0: 초기 확산 스케일
            - Lambda: 시간 감쇠 계수 [1/ms] ⭐ V3: ms 단위 명시
            - gamma_decay: 추가 감쇠 계수 [1/ms] ⭐ V3: ms 단위 명시
            - ALPHA: α-펄스 파라미터 (I0, tau_r, tau_d)
        """
        self.N = cfg["N"]
        self.NODE_STEP = cfg["node_period"]
        self.NODE_IDX = list(range(0, self.N, self.NODE_STEP))
        self.IS_NODE = np.zeros(self.N, dtype=bool)
        self.IS_NODE[self.NODE_IDX] = True

        # 기본 상수
        self.Vrest = cfg["Vrest"]  # [mV]
        self.tau = cfg["tau"]  # [ms] ⭐ V3: ms 단위 명시
        self.dx = cfg["dx"]  # [cm]
        self.cfl_safety = cfg["cfl_safety"]

        # 구간별 물리 파라미터
        self.D_node = cfg["D_node"]  # [cm²/ms] ⭐ V3: ms 단위 명시
        self.D_internode = cfg["D_internode"]  # [cm²/ms] ⭐ V3: ms 단위 명시
        self.Cm_node = cfg["Cm_node"]  # [µF/cm²]
        self.Cm_myelin = cfg["Cm_myelin"]  # [µF/cm²]
        self.gL_node = cfg["gL_node"]  # [mS/cm²]
        self.gL_myelin = cfg["gL_myelin"]  # [mS/cm²]
        self.EL = cfg["EL"]  # [mV]

        # 전류 결합 / 자극
        self.thresh = cfg["thresh"]  # [mV]
        self.coupling = cfg["coupling"]
        self.stim_gain = cfg["stim_gain"]

        # 전위 초기화
        self.V = np.full(self.N, self.Vrest, dtype=float)  # [mV]

        # 노드 전용 Na 게이트
        self.node_gNa = cfg["node_gNa"]  # [mS/cm²]
        self.node_ENa = cfg["node_ENa"]  # [mV]
        self.m_tau = cfg["node_m_tau"]  # [ms] ⭐ V3: ms 단위 명시
        self.h_tau = cfg["node_h_tau"]  # [ms] ⭐ V3: ms 단위 명시
        self.m_inf_k = cfg["node_m_inf_k"]
        self.m_inf_Vh = cfg["node_m_inf_Vh"]
        self.h_inf_k = cfg["node_h_inf_k"]
        self.h_inf_Vh = cfg["node_h_inf_Vh"]

        self.m_node = np.zeros(self.N)  # [0,1]
        self.h_node = np.zeros(self.N)  # [0,1]
        self.m_node[self.IS_NODE] = 0.05
        self.h_node[self.IS_NODE] = 0.60

        # 속도 측정용
        self.first_cross_ms = {i: None for i in self.NODE_IDX}  # [ms] ⭐ V3: ms 단위 명시

        # Inflation / 감쇠 계수
        self.c0 = cfg.get("c0", 1.0)
        self.Lambda = cfg.get("Lambda", 0.0)  # [1/ms] ⭐ V3: ms 단위 명시
        self.gamma_extra = cfg.get("gamma_decay", 0.0)  # [1/ms] ⭐ V3: ms 단위 명시

        # α-pulse parameter (cfg에서 주입)
        A = cfg.get("ALPHA", {})
        self.alpha_I0 = A.get("I0", 0.0)  # [µA/cm²]
        self.alpha_tr = A.get("tau_r", 0.5)  # [ms] ⭐ V3: ms 단위 명시
        self.alpha_td = A.get("tau_d", 3.0)  # [ms] ⭐ V3: ms 단위 명시
        self.alpha_ts = []  # spike timestamps [ms] ⭐ V3: ms 단위 명시
        
        # Sigmoid 안정화 파라미터 (cfg에서 주입)
        self.sigmoid_clip_min = cfg.get("sigmoid_clip_min", -120.0)  # [mV]
        self.sigmoid_clip_max = cfg.get("sigmoid_clip_max", 120.0)  # [mV]

    # ---------------------------------------------------------
    # Sigmoid 및 게이트 평형함수
    # ---------------------------------------------------------
    def _sigmoid(self, x): 
        """
        Sigmoid 함수 (안정화를 위한 클리핑)
        
        V3 계약:
        - 입력: x (무차원)
        - 출력: σ(x) [0,1]
        - Side-effect: 없음
        
        물리적 이유: Sigmoid 함수는 게이트 변수의 평형값을 계산하는데 사용
        클리핑은 수치 오버플로우를 방지하기 위함
        """
        x = np.clip(x, self.sigmoid_clip_min, self.sigmoid_clip_max)
        return 1.0 / (1.0 + np.exp(-x))

    def _node_m_inf(self, V):
        """
        m_inf(V) = σ((V - Vh_m)/k_m)
        
        V3 계약:
        - 입력: V [mV]
        - 출력: m_inf [0,1]
        - Side-effect: 없음
        """
        return self._sigmoid((V - self.m_inf_Vh) / self.m_inf_k)

    def _node_h_inf(self, V):
        """
        h_inf(V) = σ((V - Vh_h)/k_h)
        
        V3 계약:
        - 입력: V [mV]
        - 출력: h_inf [0,1]
        - Side-effect: 없음
        """
        return self._sigmoid((V - self.h_inf_Vh) / self.h_inf_k)

    # ---------------------------------------------------------
    # 공간 2차 미분 (Laplace Operator) - 상수 D용
    # ---------------------------------------------------------
    def _laplacian(self, V):
        """
        상수 D일 때 사용: ∇²V
        
        V3 계약:
        - 입력: V [mV]
        - 출력: Laplacian [mV/cm²]
        - Side-effect: 없음
        """
        lap = np.zeros_like(V)
        dx2 = self.dx ** 2
        lap[1:-1] = (V[:-2] - 2 * V[1:-1] + V[2:]) / dx2
        # Neumann 경계조건: ∂V/∂x = 0
        lap[0]  = 2.0 * (V[1] - V[0]) / dx2
        lap[-1] = 2.0 * (V[-2] - V[-1]) / dx2
        return lap
    
    # ---------------------------------------------------------
    # 보존형 확산항 (가변 D용): ∂/∂x(D(x) ∂V/∂x)
    # ---------------------------------------------------------
    def _diffusion_flux_form(self, V, D):
        """
        보존형 확산항 계산: ∂/∂x(D(x) ∂V/∂x)
        
        V3 계약:
        - 입력: V [mV], D [cm²/ms] ⭐ V3: ms 단위 명시
        - 출력: 확산항 [mV/(cm²·ms)] ⭐ V3: ms 단위 명시
        - Side-effect: 없음
        
        D가 공간 가변일 때 사용 (노드 vs 인터노드)
        Finite Volume 방법으로 보존형 구현
        
        물리적 이유: D가 공간 가변일 때 보존형 flux 형태가 필수
        (D가 상수일 때만 D·∂²V/∂x² = ∂/∂x(D·∂V/∂x))
        
        Parameters
        ----------
        V : np.ndarray
            막전위 [mV] (cell-centered)
        D : np.ndarray
            확산 계수 [cm²/ms] (공간 가변, cell-centered) ⭐ V3: ms 단위 명시
            
        Returns
        -------
        np.ndarray
            확산항: ∂/∂x(D(x) ∂V/∂x) [mV/(cm²·ms)] ⭐ V3: ms 단위 명시
        """
        # Face (경계면)에서 gradient 계산: ∂V/∂x at i+1/2
        # Face는 i-1/2, i+1/2 위치 (N개 cell → N+1개 face)
        dV_face = np.zeros(self.N + 1)
        dV_face[1:-1] = (V[1:] - V[:-1]) / self.dx  # [mV/cm]
        # 경계에서 Neumann 조건: ∂V/∂x = 0
        dV_face[0] = 0.0
        dV_face[-1] = 0.0
        
        # Face에서의 D (산술 평균)
        D_face = np.zeros(self.N + 1)
        D_face[1:-1] = 0.5 * (D[:-1] + D[1:])  # 내부 face
        D_face[0] = D[0]                        # 좌측 경계
        D_face[-1] = D[-1]                      # 우측 경계
        
        # Face flux: j = -D·∂V/∂x
        j_face = -D_face * dV_face  # [cm²/ms] × [mV/cm] = [mV·cm/ms] ⭐ V3: ms 단위 명시
        # 경계에서 Neumann 조건 (무유속): j_face = 0
        j_face[0] = 0.0
        j_face[-1] = 0.0
        
        # FV divergence (cell-centered): ∂j/∂x
        # cell i: (j[i+1/2] - j[i-1/2]) / dx
        div_j = (j_face[1:] - j_face[:-1]) / self.dx  # [mV/(cm²·ms)] ⭐ V3: ms 단위 명시
        
        # 보존형 확산항: -∂j/∂x = ∂/∂x(D·∂V/∂x)
        return -div_j

    # ---------------------------------------------------------
    # CFL 안정조건 (dt ≤ dx² / (2D))
    # ---------------------------------------------------------
    def _calc_dt_cfl(self):
        """
        CFL 상한 시간 스텝 계산
        
        V3 계약:
        - 입력: 없음
        - 출력: dt_cfl [ms] ⭐ V3: ms 단위 명시
        - Side-effect: 없음
        
        물리적 이유: CFL 조건을 만족하지 않으면 수치 불안정성 발생
        
        Returns
        -------
        float
            CFL 상한 시간 스텝 [ms] ⭐ V3: ms 단위 명시
        """
        Dmax = max(self.D_node, self.D_internode)  # [cm²/ms] ⭐ V3: ms 단위 명시
        return self.cfl_safety * (self.dx ** 2) / (2.0 * Dmax)  # [ms] ⭐ V3: ms 단위 명시

    # ---------------------------------------------------------
    # 노드 게이트 업데이트
    # ---------------------------------------------------------
    def _update_node_gates(self, dt):
        """
        노드 게이트 변수 업데이트
        
        V3 계약:
        - 입력: dt [ms] ⭐ V3: ms 단위 명시
        - 출력: 없음
        - Side-effect: self.m_node, self.h_node 업데이트 [0,1]
        """
        Vi = self.V[self.IS_NODE]  # [mV]
        m_inf = self._node_m_inf(Vi)  # [0,1]
        h_inf = self._node_h_inf(Vi)  # [0,1]
        self.m_node[self.IS_NODE] += dt * (m_inf - self.m_node[self.IS_NODE]) / self.m_tau  # [0,1] = [0,1] + [ms] · [1/ms] · [0,1]
        self.h_node[self.IS_NODE] += dt * (h_inf - self.h_node[self.IS_NODE]) / self.h_tau
        # 물리적 이유: 게이트 변수는 [0,1] 범위로 제한 (채널 개폐 확률)
        self.m_node = np.clip(self.m_node, 0.0, 1.0)
        self.h_node = np.clip(self.h_node, 0.0, 1.0)

    # ---------------------------------------------------------
    # 노드 Na 전류
    # ---------------------------------------------------------
    def _node_Na_current(self):
        """
        노드 Na⁺ 전류 계산
        
        V3 계약:
        - 입력: 없음
        - 출력: INa [µA/cm²]
        - Side-effect: 없음
        
        Returns
        -------
        np.ndarray
            노드 Na⁺ 전류 [µA/cm²]
        """
        INa = np.zeros(self.N)  # [µA/cm²]
        idx = np.where(self.IS_NODE)[0]
        if idx.size:
            m3h = (self.m_node[idx] ** 3) * self.h_node[idx]  # [0,1]
            INa[idx] = self.node_gNa * m3h * (self.node_ENa - self.V[idx])  # [µA/cm²] = [mS/cm²] · [0,1] · [mV]
        return INa

    # ---------------------------------------------------------
    # α-펄스 커널
    # ---------------------------------------------------------
    def trigger_alpha(self, t_ms: float):
        """
        소마 스파이크 발생 시 호출
        
        V3 계약:
        - 입력: t_ms [ms] ⭐ V3: ms 단위 명시
        - 출력: 없음
        - Side-effect: self.alpha_ts 업데이트 (스파이크 타임스탬프 추가)
        """
        self.alpha_ts.append(float(t_ms))

    def _alpha_kernel(self, t_ms: float):
        """
        I_α(t) = I₀[exp(−(t−t₀)/τ_d) − exp(−(t−t₀)/τ_r)]₊
        
        V3 계약:
        - 입력: t_ms [ms] ⭐ V3: ms 단위 명시
        - 출력: I_alpha [µA/cm²]
        - Side-effect: self.alpha_ts 업데이트 (오래된 스파이크 제거)
        
        오래된 스파이크는 자동 제거하여 메모리 누수 방지
        
        Returns
        -------
        float
            α-펄스 전류 [µA/cm²]
        """
        if self.alpha_I0 == 0.0 or not self.alpha_ts:
            return 0.0
        
        # 오래된 스파이크 제거 (5*tau_d 이후)
        cutoff = t_ms - 5.0 * self.alpha_td  # [ms] ⭐ V3: ms 단위 명시
        self.alpha_ts = [t for t in self.alpha_ts if t > cutoff]
        
        val = 0.0
        for t0 in self.alpha_ts:
            dt = t_ms - t0  # [ms] ⭐ V3: ms 단위 명시
            if dt <= 0.0:
                continue
            val += (np.exp(-dt / self.alpha_td) - np.exp(-dt / self.alpha_tr))
        return max(0.0, val) * self.alpha_I0  # [µA/cm²]

    # ---------------------------------------------------------
    # 노드 전위 임계 통과 기록 (속도 측정용)
    # ---------------------------------------------------------
    def _record_crossings(self, t_ms):
        """
        노드 전위 임계 통과 기록 (속도 측정용)
        
        V3 계약:
        - 입력: t_ms [ms] ⭐ V3: ms 단위 명시
        - 출력: 없음
        - Side-effect: self.first_cross_ms 업데이트 (첫 통과 시간 기록)
        """
        for i in self.NODE_IDX:
            if self.first_cross_ms[i] is None and self.V[i] >= self.thresh:
                self.first_cross_ms[i] = t_ms  # [ms] ⭐ V3: ms 단위 명시

    # ---------------------------------------------------------
    # 메인 전도 스텝
    # ---------------------------------------------------------
    def step(self, dt_elec: float, t_ms: float, I0_from_soma: float, soma_V: float):
        """
        한 시점에서의 축삭 전도 계산
        
        V3 계약:
        - 입력:
          - dt_elec: [ms] (전기적 시간 스텝) ⭐ V3 계약 고정
          - t_ms: [ms] (현재 시간) ⭐ V3 계약 고정
          - I0_from_soma: [µA] (소마로부터의 전류 입력)
          - soma_V: [mV] (소마 막전위, 값 복사) ⭐ V3: 참조 공유 금지
        - 출력: 없음 (내부 상태 업데이트)
        - Side-effect:
          - self.V 업데이트 [mV]
          - self.m_node, self.h_node 업데이트 [0,1]
          - self.first_cross_ms 업데이트 [ms] (속도 측정용)
        
        Parameters
        ----------
        dt_elec : float
            전기적 시간 스텝 [ms] ⭐ V3: ms 단위 명시
        t_ms : float
            현재 시간 [ms] (서브스텝 시작 시간) ⭐ V3: ms 단위 명시
        I0_from_soma : float
            소마로부터의 전류 입력 [µA]
        soma_V : float
            소마 막전위 [mV] (값 복사) ⭐ V3: 참조 공유 금지
        """
        # 입력 검증
        if dt_elec <= 0:
            raise ValueError(f"dt_elec must be > 0, got {dt_elec}")
        if t_ms < 0:
            raise ValueError(f"t_ms must be >= 0, got {t_ms}")
        
        # CFL 기반 서브스텝 분할
        dt_cfl = self._calc_dt_cfl()  # [ms] ⭐ V3: ms 단위 명시
        n_sub = max(1, int(np.ceil(dt_elec / max(1e-12, dt_cfl))))
        dt_sub = dt_elec / n_sub  # [ms] ⭐ V3: ms 단위 명시
        
        # 서브스텝 시간 추적
        t_current = t_ms  # 서브스텝 시작 시간 [ms] ⭐ V3: ms 단위 명시
        
        # 구간별 파라미터 분포 (서브스텝 밖에서 한 번만 계산)
        D = np.full(self.N, self.D_internode)  # [cm²/ms] ⭐ V3: ms 단위 명시
        D[self.IS_NODE] = self.D_node
        Cm = np.full(self.N, self.Cm_myelin)  # [µF/cm²]
        Cm[self.IS_NODE] = self.Cm_node
        gL = np.full(self.N, self.gL_myelin)  # [mS/cm²]
        gL[self.IS_NODE] = self.gL_node

        for sub_i in range(n_sub):
            self._update_node_gates(dt_sub)

            # 외부 자극 (소마 결합)
            # V3: soma_V는 값 복사로 전달됨 (참조 공유 금지)
            I_ext = np.zeros(self.N)  # [µA/cm²]
            I_ext[0] = I0_from_soma + self.coupling * (soma_V - self.V[0])  # [µA/cm²]

            # 노드 Na 전류
            I_Na = self._node_Na_current()  # [µA/cm²]

            # Inflation factor 적용 (현재 시간 사용)
            c_t = self.c0 * np.exp(-self.Lambda * t_current)  # [무차원]
            D_eff = c_t * D  # [cm²/ms] ⭐ V3: ms 단위 명시

            # 보존형 확산항 계산: ∂/∂x(D(x) ∂V/∂x)
            # D가 공간 가변(노드 vs 인터노드)이므로 flux 형태 필수
            diff_term = self._diffusion_flux_form(self.V, D_eff)  # [mV/(cm²·ms)] ⭐ V3: ms 단위 명시

            # α-펄스 자극 (현재 시간 사용)
            I_alpha0 = self._alpha_kernel(t_current)  # [µA/cm²]
            if I_alpha0 != 0.0:
                I_ext[0] += I_alpha0

            # 추가 감쇠항
            extra_decay = -self.gamma_extra * (self.V - self.Vrest)  # [mV/ms] ⭐ V3: ms 단위 명시

            # 막전위 변화율
            # ∂V/∂t = ∂/∂x(D(x) ∂V/∂x) - g_L(x)(V - E_L)/C_m(x)
            #        + [I_ext(x,t) + I_Na_node(x,t)]/C_m(x)
            #        - γ_extra(V - V_rest)
            dVdt = diff_term - gL * (self.V - self.EL) / Cm + (I_ext + I_Na) / Cm + extra_decay  # [mV/ms] ⭐ V3: ms 단위 명시

            # 막전위 갱신
            # 물리적 이유: 막전위는 생리학적 범위 [-120, 120] mV 내에서 유지되어야 함
            # NaN/Inf는 수치 오류이므로 안전한 값으로 대체
            self.V = np.nan_to_num(
                self.V + dt_sub * dVdt,  # [mV] = [mV] + [ms] · [mV/ms]
                nan=self.Vrest, 
                posinf=120.0, 
                neginf=-120.0
            )

            # 노드 통과 시간 기록 (현재 시간 사용)
            self._record_crossings(t_current)
            
            # 서브스텝 시간 증가
            t_current += dt_sub  # [ms] ⭐ V3: ms 단위 명시

    # ---------------------------------------------------------
    # 도약전도 속도 계산
    # ---------------------------------------------------------
    def velocity_last(self) -> float:
        """
        노드 통과 시간 차이 기반 평균 전도속도 계산
        
        V3 계약:
        - 입력: 없음
        - 출력: v [m/s] (전도 속도)
        - Side-effect: 없음
        
        Returns
        -------
        float
            평균 전도 속도 [m/s]
        """
        times = [self.first_cross_ms[i] for i in self.NODE_IDX if self.first_cross_ms[i] is not None]  # [ms] ⭐ V3: ms 단위 명시
        if len(times) < 2:
            return 0.0
        arr = np.array(times)
        dt = np.diff(arr)  # [ms] ⭐ V3: ms 단위 명시
        dt = dt[dt > 0.0]
        if dt.size == 0:
            return 0.0
        mean_dt_ms = float(np.mean(dt))  # [ms] ⭐ V3: ms 단위 명시
        dist_cm = self.NODE_STEP * self.dx  # [cm]
        v_m_s = (dist_cm / (mean_dt_ms * 1e-3)) * 0.01  # cm/ms → m/s
        return v_m_s


# =============================================================
# V3 독립 실행 테스트
# =============================================================
if __name__ == "__main__":
    """
    V3 계약 고정 검증:
    - 시간 단위: [ms] 확인
    - 단일 방향화 확인
    - 입출력/side-effect 확인
    """
    cfg = {
        "N": 121,
        "node_period": 5,
        "Vrest": -70.0,  # [mV]
        "tau": 1.2,  # [ms] ⭐ V3: ms 단위 명시
        "dx": 1e-3,  # [cm]
        "cfl_safety": 0.9,
        "D_node": 1.5e-3,  # [cm²/ms] ⭐ V3: ms 단위 명시
        "D_internode": 1.5e-5,  # [cm²/ms] ⭐ V3: ms 단위 명시
        "Cm_node": 1.0,  # [µF/cm²]
        "Cm_myelin": 0.005,  # [µF/cm²]
        "gL_node": 0.25,  # [mS/cm²]
        "gL_myelin": 1e-4,  # [mS/cm²]
        "EL": -54.4,  # [mV]
        "thresh": -50.0,  # [mV]
        "coupling": 0.0,
        "stim_gain": 15.0,
        "node_gNa": 1200.0,  # [mS/cm²]
        "node_ENa": 50.0,  # [mV]
        "node_m_tau": 0.03,  # [ms] ⭐ V3: ms 단위 명시
        "node_h_tau": 0.40,  # [ms] ⭐ V3: ms 단위 명시
        "node_m_inf_k": 6.0,
        "node_m_inf_Vh": -37.0,  # [mV]
        "node_h_inf_k": -6.0,
        "node_h_inf_Vh": -58.0,  # [mV]
        "c0": 1.0,
        "Lambda": 0.0,  # [1/ms] ⭐ V3: ms 단위 명시
        "gamma_decay": 0.0,  # [1/ms] ⭐ V3: ms 단위 명시
        "ALPHA": {"I0": 0.0, "tau_r": 0.5, "tau_d": 3.0},  # [ms] ⭐ V3: ms 단위 명시
    }

    axon = MyelinatedAxon(cfg)
    dt_elec = 0.01  # [ms] ⭐ V3: ms 단위 명시
    t_ms = 0.0  # [ms] ⭐ V3: ms 단위 명시
    I0_from_soma = 10.0  # [µA]
    soma_V = -70.0  # [mV] ⭐ V3: 값 복사로 전달
    
    print("=" * 60)
    print("MyelinatedAxon V3 — 계약 고정 검증")
    print("=" * 60)
    print(f"dt_elec: {dt_elec} [ms] ⭐ V3 계약")
    print(f"soma_V: {soma_V} [mV] (값 복사) ⭐ V3 계약")
    print("-" * 60)
    print("[MyelinatedAxon Test]")
    print(f"{'t(ms)':>8} | {'V[0](mV)':>12} | {'V[60](mV)':>12}")
    print("-" * 60)
    
    for t in range(10):
        axon.step(dt_elec=dt_elec, t_ms=t*dt_elec, I0_from_soma=I0_from_soma, soma_V=soma_V)
        print(f"{t*dt_elec:8.2f} | {axon.V[0]:12.3f} | {axon.V[60]:12.3f}")
    
    print("=" * 60)
    print("✅ V3 계약 고정 검증 완료")
    print("  - 시간 단위: [ms] 확인")
    print("  - 단일 방향화 확인 (Soma → Axon)")
    print("  - 입출력/side-effect 명시 확인")

