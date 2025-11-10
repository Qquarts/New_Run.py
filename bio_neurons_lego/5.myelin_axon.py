# =============================================================
# myelinated_axon.py — 물리적 도약전도 (Saltatory Conduction)
# =============================================================
# 목적:
#   - 소마(Soma)에서 전송된 활동전위가 축삭을 따라 도약전도(saltatory conduction)로 전달되는 과정 모델링
#   - 노드(Node)와 인터노드(Internode) 구간을 구분
#   - 각 구간의 확산(D), 막용량(Cm), 누설전도(gL) 상이
#   - 노드에서만 빠른 Na⁺ 채널이 활성화되어 도약 전위 형성
#   - 시간 감쇠(Lambda), 에너지 감쇠(gamma_extra), α-펄스 자극까지 통합
#
# =============================================================
# 수식 요약
# -------------------------------------------------------------
# (1) 막전위 변화식
#     ∂V/∂t = D(x)·∂²V/∂x² - g_L(x)(V - E_L)/C_m(x)
#              + [I_ext(x,t) + I_Na_node(x,t)]/C_m(x)
#              - γ_extra(V - V_rest)
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
# =============================================================

import numpy as np

class MyelinatedAxon:
    r"""
    MyelinatedAxon — Saltatory Conduction Model
    -------------------------------------------
    ∂V/∂t = D(x)∂²V/∂x² - g_L(x)(V - E_L)/C_m(x)
             + [I_ext(x,t) + I_Na_node(x,t)]/C_m(x)
             - γ_extra(V - V_rest)

    Node only:
        I_Na_node = g_Na_node·m³·h·(E_Na_node - V)
        ḿ = (m_inf(V) - m)/τ_m
        ḣ = (h_inf(V) - h)/τ_h
    """

    # ---------------------------------------------------------
    # 초기화
    # ---------------------------------------------------------
    def __init__(self, cfg: dict):
        self.N = cfg["N"]
        self.NODE_STEP = cfg["node_period"]
        self.NODE_IDX = list(range(0, self.N, self.NODE_STEP))
        self.IS_NODE = np.zeros(self.N, dtype=bool)
        self.IS_NODE[self.NODE_IDX] = True

        # 기본 상수
        self.Vrest = cfg["Vrest"]
        self.tau = cfg["tau"]
        self.dx = cfg["dx"]
        self.cfl_safety = cfg["cfl_safety"]

        # 구간별 물리 파라미터
        self.D_node = cfg["D_node"]
        self.D_internode = cfg["D_internode"]
        self.Cm_node = cfg["Cm_node"]
        self.Cm_myelin = cfg["Cm_myelin"]
        self.gL_node = cfg["gL_node"]
        self.gL_myelin = cfg["gL_myelin"]
        self.EL = cfg["EL"]

        # 전류 결합 / 자극
        self.thresh = cfg["thresh"]
        self.coupling = cfg["coupling"]
        self.stim_gain = cfg["stim_gain"]

        # 전위 초기화
        self.V = np.full(self.N, self.Vrest, dtype=float)

        # 노드 전용 Na 게이트
        self.node_gNa = cfg["node_gNa"]
        self.node_ENa = cfg["node_ENa"]
        self.m_tau = cfg["node_m_tau"]
        self.h_tau = cfg["node_h_tau"]
        self.m_inf_k = cfg["node_m_inf_k"]
        self.m_inf_Vh = cfg["node_m_inf_Vh"]
        self.h_inf_k = cfg["node_h_inf_k"]
        self.h_inf_Vh = cfg["node_h_inf_Vh"]

        self.m_node = np.zeros(self.N)
        self.h_node = np.zeros(self.N)
        self.m_node[self.IS_NODE] = 0.05
        self.h_node[self.IS_NODE] = 0.60

        # 속도 측정용
        self.first_cross_ms = {i: None for i in self.NODE_IDX}

        # Inflation / 감쇠 계수
        self.c0 = cfg.get("c0", 1.0)
        self.Lambda = cfg.get("Lambda", 0.0)       # per ms
        self.gamma_extra = cfg.get("gamma_decay", 0.0)

        # α-pulse parameter (from global CONFIG)
        from __main__ import CONFIG
        A = CONFIG.get("ALPHA", {})
        self.alpha_I0 = A.get("I0", 0.0)
        self.alpha_tr = A.get("tau_r", 0.5)
        self.alpha_td = A.get("tau_d", 3.0)
        self.alpha_ts = []  # spike timestamps (ms)

    # ---------------------------------------------------------
    # Sigmoid 및 게이트 평형함수
    # ---------------------------------------------------------
    @staticmethod
    def _sigmoid(x): 
        x = np.clip(x, -120.0, 120.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _node_m_inf(self, V):
        """m_inf(V) = σ((V - Vh_m)/k_m)"""
        return self._sigmoid((V - self.m_inf_Vh) / self.m_inf_k)

    def _node_h_inf(self, V):
        """h_inf(V) = σ((V - Vh_h)/k_h)"""
        return self._sigmoid((V - self.h_inf_Vh) / self.h_inf_k)

    # ---------------------------------------------------------
    # 공간 2차 미분 (Laplace Operator)
    # ---------------------------------------------------------
    def _laplacian(self, V):
        lap = np.zeros_like(V)
        dx2 = self.dx ** 2
        lap[1:-1] = (V[:-2] - 2 * V[1:-1] + V[2:]) / dx2
        # Neumann 경계조건: ∂V/∂x = 0
        lap[0]  = 2.0 * (V[1] - V[0]) / dx2
        lap[-1] = 2.0 * (V[-2] - V[-1]) / dx2
        return lap

    # ---------------------------------------------------------
    # CFL 안정조건 (dt ≤ dx² / (2D))
    # ---------------------------------------------------------
    def _calc_dt_cfl(self):
        Dmax = max(self.D_node, self.D_internode)
        return self.cfl_safety * (self.dx ** 2) / (2.0 * Dmax)

    # ---------------------------------------------------------
    # 노드 게이트 업데이트
    # ---------------------------------------------------------
    def _update_node_gates(self, dt):
        Vi = self.V[self.IS_NODE]
        m_inf = self._node_m_inf(Vi)
        h_inf = self._node_h_inf(Vi)
        self.m_node[self.IS_NODE] += dt * (m_inf - self.m_node[self.IS_NODE]) / self.m_tau
        self.h_node[self.IS_NODE] += dt * (h_inf - self.h_node[self.IS_NODE]) / self.h_tau
        self.m_node = np.clip(self.m_node, 0.0, 1.0)
        self.h_node = np.clip(self.h_node, 0.0, 1.0)

    # ---------------------------------------------------------
    # 노드 Na 전류
    # ---------------------------------------------------------
    def _node_Na_current(self):
        INa = np.zeros(self.N)
        idx = np.where(self.IS_NODE)[0]
        if idx.size:
            m3h = (self.m_node[idx] ** 3) * self.h_node[idx]
            INa[idx] = self.node_gNa * m3h * (self.node_ENa - self.V[idx])
        return INa

    # ---------------------------------------------------------
    # α-펄스 커널
    # ---------------------------------------------------------
    def trigger_alpha(self, t_ms: float):
        """소마 스파이크 발생 시 호출"""
        self.alpha_ts.append(float(t_ms))

    def _alpha_kernel(self, t_ms: float):
        """I_α(t) = I₀[exp(−(t−t₀)/τ_d) − exp(−(t−t₀)/τ_r)]₊"""
        if self.alpha_I0 == 0.0 or not self.alpha_ts:
            return 0.0
        val = 0.0
        for t0 in self.alpha_ts:
            dt = t_ms - t0
            if dt <= 0.0:
                continue
            val += (np.exp(-dt / self.alpha_td) - np.exp(-dt / self.alpha_tr))
        return max(0.0, val) * self.alpha_I0

    # ---------------------------------------------------------
    # 노드 전위 임계 통과 기록 (속도 측정용)
    # ---------------------------------------------------------
    def _record_crossings(self, t_ms):
        for i in self.NODE_IDX:
            if self.first_cross_ms[i] is None and self.V[i] >= self.thresh:
                self.first_cross_ms[i] = t_ms

    # ---------------------------------------------------------
    # 메인 전도 스텝
    # ---------------------------------------------------------
    def step(self, dt_elec: float, t_ms: float, I0_from_soma: float, soma_V: float):
        """한 시점에서의 축삭 전도 계산"""
        # CFL 기반 서브스텝 분할
        dt_cfl = self._calc_dt_cfl()
        n_sub = max(1, int(np.ceil(dt_elec / max(1e-12, dt_cfl))))
        dt_sub = dt_elec / n_sub

        for _ in range(n_sub):
            self._update_node_gates(dt_sub)

            # 구간별 파라미터 분포
            D = np.full(self.N, self.D_internode)
            D[self.IS_NODE] = self.D_node
            Cm = np.full(self.N, self.Cm_myelin)
            Cm[self.IS_NODE] = self.Cm_node
            gL = np.full(self.N, self.gL_myelin)
            gL[self.IS_NODE] = self.gL_node

            # 외부 자극 (소마 결합)
            I_ext = np.zeros(self.N)
            I_ext[0] = I0_from_soma + self.coupling * (soma_V - self.V[0])

            # 노드 Na 전류
            I_Na = self._node_Na_current()

            # 확산항 계산
            lap = self._laplacian(self.V)

            # Inflation factor 적용
            c_t = self.c0 * np.exp(-self.Lambda * t_ms)
            D_eff = c_t * D

            # α-펄스 자극
            I_alpha0 = self._alpha_kernel(t_ms)
            if I_alpha0 != 0.0:
                I_ext[0] += I_alpha0

            # 추가 감쇠항
            extra_decay = -self.gamma_extra * (self.V - self.Vrest)

            # 막전위 변화율
            dVdt = D_eff * lap - gL * (self.V - self.EL) / Cm + (I_ext + I_Na) / Cm + extra_decay

            # 막전위 갱신
            self.V = np.nan_to_num(self.V + dt_sub * dVdt, nan=self.Vrest, posinf=120.0, neginf=-120.0)

            # 노드 통과 시간 기록
            self._record_crossings(t_ms)

    # ---------------------------------------------------------
    # 도약전도 속도 계산
    # ---------------------------------------------------------
    def velocity_last(self) -> float:
        """노드 통과 시간 차이 기반 평균 전도속도 계산 (m/s)"""
        times = [self.first_cross_ms[i] for i in self.NODE_IDX if self.first_cross_ms[i] is not None]
        if len(times) < 2:
            return 0.0
        arr = np.array(times)
        dt = np.diff(arr)
        dt = dt[dt > 0.0]
        if dt.size == 0:
            return 0.0
        mean_dt_ms = float(np.mean(dt))
        dist_cm = self.NODE_STEP * self.dx
        v_m_s = (dist_cm / (mean_dt_ms * 1e-3)) * 0.01  # cm/ms → m/s
        return v_m_s