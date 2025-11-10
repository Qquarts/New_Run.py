# =============================================================
# Qquarts co Present 지은이 : GNJz
# =============================================================

# =============================================================
# new_run_v1.py — 통합 파이프라인 시뮬레이션 (Integration Testbed)
# =============================================================
#
# 📋 파일 구조 설명:
#   이 파일은 "제어판 시뮬레이션 파일"로, 모든 클래스를 한 곳에 모아
#   개발/디버깅/확인을 용이하게 합니다.
#
# 🔧 개발 워크플로우:
#   1. 각 클래스의 원본 코드는 분리된 파일에 있음:
#      - pipeline/dtg_system.py
#      - bio_neurons/mitochon_atp.py
#      - bioneurons/hh_soma.py
#      - bioneurons/axon_gate.py
#      - bioneurons/ca_vesicle.py
#      - 기타 등등...
#
#   2. 이 파일(new_run_v1.py)은:
#      - 모든 클래스를 복사하여 인라인으로 포함
#      - 원본 파일 수정 없이 여기서 바로 코드 수정/확인 가능
#      - 파이프라인 전체를 한 눈에 보고 조립/테스트
#
# ✅ 이 구조의 장점:
#   - 빠른 개발/디버깅: 모든 코드를 한 파일에서 확인
#   - 원본 파일 영향 없음: 여기서 수정해도 원본은 안전
#   - 통합 테스트 용이: 전체 파이프라인을 한 곳에서 실행
#   - 제어판 역할: 모든 모듈을 조립하여 시뮬레이션 실행
#
# ⚠️ 주의사항:
#   - 원본 파일과 동기화 유지 필요 (주기적으로 원본 반영)
#   - 안정화 후에는 import 방식으로 전환 고려 가능
#
# =============================================================
# 수식 요약 (모든 계층 통합)
# =============================================================

# [DTG Layer — Energy–Phase Dynamics]
# -----------------------------------
# dE/dt   = g_sync · (ATP - E) - γ · (E - E0)
# dφ/dt   = ω0 + α · (E - E0)
#  └ 위상(φ)과 에너지(E)를 동기화시키는 메타 제어 방정식
#  └ 출력: φ(t), E(t) → Mito 입력

# [Mitochondria Layer — Energy Metabolism]
# ----------------------------------------
# dE_buf/dt = (P_in - P_loss) - J_transfer
# dATP/dt   = κ · (E_buf - ATP) - J_use
# J_transfer = k_transfer · (E_buf - ATP)
# J_use ≈ J_NaK + J_Ca   # (Na/K 펌프 및 Ca 펌프의 ATP 소비)
# Heat ↑ = (1 - η) · J_transfer
# CO2  ↑ = c_CO2 · J_transfer
#  └ E_buf → ATP 변환 (효율 η)
#  └ Heat, CO₂ 누적은 대사 부산물

# [Hodgkin–Huxley Soma — Membrane Potential]
# ------------------------------------------
# C_m · dV/dt = g_Na·m³·h·(E_Na - V) + g_K·n⁴·(E_K - V) + g_L·(E_L - V) + I_ext - I_pump
# dm/dt = α_m(V)·(1 - m) - β_m(V)·m
# dh/dt = α_h(V)·(1 - h) - β_h(V)·h
# dn/dt = α_n(V)·(1 - n) - β_n(V)·n
# I_pump = g_pump · (1 - e^{-(ATP/ATP₀)}) · (V - E_pump)
#  └ 막전위 발화 및 회복 (ATP 의존 펌프 포함 가능)

# [Myelinated Axon — Physical Saltatory Conduction]
# -------------------------------------------------
# ∂V/∂t = D(x)·∂²V/∂x² - (V - V_rest)/τ
#          + [ I_ext - g_L(x)·(V - E_L) + I_Na_node ] / C_m(x)
# I_Na_node = g_Na_node·m³·h·(E_Na_node - V)   (노드 위치에서만)
# dm/dt = (m_inf(V) - m)/τ_m
# dh/dt = (h_inf(V) - h)/τ_h
# m_inf(V) = σ((V - Vh_m)/k_m)
# h_inf(V) = σ((V - Vh_h)/k_h)
# σ(x) = 1 / (1 + e^{-x})
#  └ 노드 구간만 활성 도약전도, 인터노드 구간은 전도 억제

# [Ca²⁺ Vesicle — Synaptic Release Dynamics]
# ------------------------------------------
# d[Ca]/dt = Σ_k A·α(t - t_k) - k_c·ATP·([Ca] - [Ca]_0)
# α(t) = (e^{-t/τ_d} - e^{-t/τ_r})_+     # 스파이크 트리거 α-커널
# S = ([Ca] - [Ca]_0) / ([Ca]_max - [Ca]_0)
# P_in(Mito) = P_in₀·(1 + λ·S_alert)     # (Ca–Mito 피드백 확장 가능)
#  └ Ca 농도 상승 → 소포 방출 → ATP 펌프 소모

# [Energy–Chemical Feedback Loop]
# -------------------------------------------
# CO₂ ↑ → P_loss = P_loss₀·(1 + β_CO2·CO₂)
# Heat ↑ → η = η₀ - β_heat·(Heat - Heat₀)
# Ca alert → Mito recover_k = k₀·(1 + λ_Ca·S_alert)
#  └ 대사 부산물이 다시 에너지 효율에 영향
#  └ 구현: MetabolicFeedback 클래스 (섹션 8) 참조

# [DTG–Soma Phase Coupling]
# -------------------------------------
# I_ext = I_base + A_φ · sin(φ)
#  └ 위상 φ(t)가 발화 주기를 조절하는 내적 발진 구조
#  └ 구현: run_pipeline() 내부에서 DTG 위상으로 I_ext 변조

# =============================================================
# 전체 루프:
#   DTG → Mito → HH Soma → Axon → Ca²⁺ Vesicle
#     ↑                                 ↓
#     └──────────── Feedback (ATP·Heat·CO₂·Ca)
# =============================================================

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import pandas as pd

# =============================================================
# Optional Color Output (Console visualization helper)
# =============================================================
# 역할:
# - colorama 모듈이 있으면 컬러 출력 활성화
# - 없으면 흑백 모드로 안전하게 동작
# - 계산/시뮬레이션 결과에는 영향 없음
# =============================================================

try:
    from colorama import Fore, Style
    HAS_COLOR = True
except ImportError:
    # colorama 미설치 시, 빈 문자열로 대체 → 흑백 안전 모드
    class _NoColor:
        GREEN = YELLOW = RED = CYAN = MAGENTA = ""
    class _NoStyle:
        RESET_ALL = ""
    Fore = _NoColor()
    Style = _NoStyle()
    HAS_COLOR = False

# =============================================================
# 0. GLOBAL CONFIG  (Pipeline-Ready / CFL-Stable / Bio-Complete)
# =============================================================

CONFIG = {
    # ------------------ DTG (Energy–Phase Dynamics) ------------------
    # dE/dt = g_sync · (ATP - E) - γ · (E - E0)
    # dφ/dt = ω0 + α · (E - E0)
    "DTG": {
        "E0": 100.0,
        "omega0": 1.0,
        "alpha": 0.03,
        "gamma": 0.10,
        "sync_gain": 0.20,
    },

    # ------------------ MITO (Energy Metabolism) ---------------------
    # dE_buf/dt = (Pin - Ploss) - k_transfer·(E_buf - ATP)
    # dATP/dt   = k_transfer·(E_buf - ATP) - J_use
    # Heat↑ = (1-η)·k_transfer·(E_buf - ATP)_+,  CO2↑ = c_CO2·(...)
    "MITO": {
        "ATP0": 105.0,
        "Ebuf0": 70.0,
        "Pin": 10.0,
        "Ploss": 1.2,
        "recover_k": 8.0,
        "recover_thresh": 60.0,
        "delta_transfer": 5.0,
        "ATP_clip": (80.0, 120.0),
        "Ebuf_clip": (15.0, 100.0),
        "k_transfer": 0.3,
        "eta": 0.60,
        "c_CO2": 0.80,
        "Heat0": 0.0,
        "CO2_0": 0.0,
        "D_H": 1e-6,     # 활성화 (실제 확산)
        "dx_heat": 1e-3,
        "k_heat": 0.01,
        "Heat_env": 0.0,
        "CO2_env": 0.0,
    },

    # ------------------ HH Soma (Membrane Potential) -----------------
    # C_m dV/dt = gNa m³h(ENa−V) + gK n⁴(EK−V) + gL(EL−V) + I_ext − I_pump
    # I_pump = g_pump · (1 - exp[-ATP/ATP0_ref]) · (V - E_pump)
    "HH": {
        "V0": -70.0,
        "gNa": 220.0,
        "gK": 26.0,
        "gL": 0.08,
        "ENa": 50.0,
        "EK": -77.0,
        "EL": -54.4,
        "spike_thresh": 0.0,
        # 🚀 활성화: ATP↔막전위 펌프 피드백
        "use_pump": True,
        "g_pump": 0.5,
        "E_pump": -70.0,
        "ATP0_ref": 100.0,
        "g_pump_consume": 0.02,
    },

    # ------------------ Myelinated Axon (Saltatory) ------------------
    # ∂V/∂t = D(x)∂²V/∂x² - gL(x)(V−EL)/Cm(x) + [I_ext + I_Na_node]/Cm(x)
    # Node Na gate:  ẋ = (x_inf(V) - x)/τ_x,  I_Na_node = gNa_node m³ h (ENa - V)
    "AXON": {
        "N": 121,
        "node_period": 5,         # 0,5,10,... are nodes
        "Vrest": -70.0,
        "EL": -54.4,
        "tau": 1.2,
        "dx": 1.0e-3,             # [cm]  (CFL 계산 기준)
        "D_node": 1.5e-4,         # [cm^2/ms]  # ✅ 1.5e-3 → 1.5e-4 (CFL 완화)
        "D_internode": 1.5e-6,    # [cm^2/ms]  # ✅ 1.5e-5 → 1.5e-6
        "Cm_node": 1.0,
        "Cm_myelin": 0.005,
        "gL_node": 0.25,
        "gL_myelin": 1.0e-4,
        "thresh": -50.0,
        "cfl_safety": 0.9,

        # Node fast Na
        "node_gNa": 1200.0,
        "node_ENa": 50.0,
        "node_m_tau": 0.03,
        "node_h_tau": 0.40,
        "node_m_inf_k": 6.0,
        "node_m_inf_Vh": -37.0,
        "node_h_inf_k": -6.0,
        "node_h_inf_Vh": -58.0,

        # 🚀 활성화: 소마↔축삭 결합 & 초기 구동력
        "coupling": 3.0,
        "stim_gain": 260.0,

        # optional modifiers
        "c0": 1.0,
        "Lambda": 0.0,
        "gamma_decay": 0.0,
    },

    # ------------------ Ca²⁺ Vesicle (Release) -----------------------
    # d[Ca]/dt = Σ A·α(t−t_k) − k_c·ATP·([Ca]−[Ca]_0)
    "CA": {
        "C0": 1e-7,
        "Cmax": 5e-6,
        "A": 0.25e-6,
        "tau_r": 0.0005,   # [s] (0.5 ms)
        "tau_d": 0.08,     # [s] (80 ms)
        "k_c": 0.02,
        "max_spike_memory_ms": 2000.0,
        "dt_ms": 0.02,
    },

    # ------------------ Integrator / Run ------------------------------
    # ⚠️ CFL: dt_elec ≤ 0.9 * dx^2 / (2*D_max)
    # dx=1e-3, D_max=1.5e-3 → dt_cfl ≈ 0.9*(1e-6)/(2*1.5e-3) ≈ 0.00030 ms
    "RUN": {
        "T_ms": 300,
        "dt_bio": 1.0,
        "dt_elec": 0.02,     # 세밀도 향상 (quick 버전과 동기화)
        "print_every_ms": 5,
        "log_interval": 5,
        "ms_per_sim_ms": 0.4,
        "color": True,
    },

    # ------------------ Alpha Pulse (optional) ------------------------
    # Iα(t) = I0 · (e^{-t/τ_d} − e^{-t/τ_r})_+
    "ALPHA": {
        "I0": 50.0,
        "tau_r": 0.5,    # [ms]
        "tau_d": 3.0,    # [ms]
    },

    # ------------------ Energy Ledger (optional) ----------------------
    "LEDGER": {
        "xi_prod": 0.0,
        "chi_spike": 0.0,
        "zeta_leak": 0.0,
    },

    # ------------------ Solver Methods -------------------------------
    # [PATCH] 수치 적분 방법 선택 (각 모듈별로 다른 solver 사용 가능)
    # 기능: 각 모듈의 미분 방정식을 적분할 때 사용할 수치 방법을 지정
    # - DTG: 에너지-위상 동기화 (euler: 기본, rk4: 더 정확하지만 느림)
    # - MITO: ATP 대사 (rk4: 고정밀도 필요)
    # - HH: Hodgkin-Huxley 막전위 (rk4: 게이트+막전위 동시 적분)
    # - CA: Ca²⁺ 농도 (heun: 중간 정확도, semi-implicit도 가능)
    # - AXON: 축삭 전도 (cfl_euler: CFL 조건 만족하는 Euler, 서브스텝 포함)
    #
    # ⚙️ Solver Integration Policy:
    #   - Euler: 기본 테스트용, 빠르지만 1차 정확도
    #   - Heun: 2차 정확도, Ca·Heat 등 비선형 완화에 적합
    #   - RK4 : 4차 정확도, DTG/Mito/HH 정밀 시뮬에 적합
    #   - cfl_euler: 축삭용 내부 서브스텝 포함, 안정성 확보 전용

    "SOLVER": {
        "DTG": "euler",        # "rk4"로 바꿔도 됨 (더 정확하지만 계산 비용 증가)
        "MITO": "rk4",         # 4차 Runge-Kutta: ATP 대사 정밀도 향상
        "HH":   "rk4",         # 4차 Runge-Kutta: 게이트+막전위 동시 적분으로 정확도 향상
        "CA":   "heun",        # Heun 방법: 중간 정확도, semi-implicit도 가능
        "AXON": "cfl_euler"    # CFL 조건 만족 Euler: 안정성 보장, 서브스텝 포함
    },
}

# =============================================================
# a. Solver Utilities (Numerical Integration Methods)
# =============================================================
# 수치 적분 방법 유틸리티 함수들
# =============================================================

def rk4_step(f, y, dt):
    """
    [PATCH] 4차 Runge-Kutta 방법으로 한 스텝 적분
    
    기능: 미분 방정식 dy/dt = f(y)를 4차 Runge-Kutta 방법으로 적분
    - Euler 방법보다 정확도가 높음 (4차 정확도)
    - 계산 비용은 4배 증가 (k1, k2, k3, k4 계산 필요)
    - MITO, HH 모듈에서 사용 (SOLVER 설정에서 "rk4" 지정 시)
    
    알고리즘:
    1. k1 = f(y)                     # 현재 점에서의 기울기
    2. k2 = f(y + 0.5*dt*k1)         # 중간 점에서의 기울기
    3. k3 = f(y + 0.5*dt*k2)         # 중간 점에서의 기울기 (개선)
    4. k4 = f(y + dt*k3)             # 끝 점에서의 기울기
    5. y_new = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)  # 가중 평균
    
    Parameters
    ----------
    f : callable
        미분 방정식의 우변 함수: dy/dt = f(y)
    y : array-like
        현재 상태 벡터
    dt : float
        시간 스텝 크기
        
    Returns
    -------
    array-like
        다음 스텝의 상태 벡터
    """
    k1 = f(y)                        # 현재 점에서의 기울기
    k2 = f(y + 0.5*dt*k1)            # 중간 점 1에서의 기울기
    k3 = f(y + 0.5*dt*k2)            # 중간 점 2에서의 기울기
    k4 = f(y + dt*k3)                # 끝 점에서의 기울기
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)  # 가중 평균으로 최종 값 계산

def heun_step(f, y, dt):
    """
    Heun 방법 (개선된 Euler 방법)으로 한 스텝 적분
    
    기능: 미분 방정식 dy/dt = f(y)를 Heun 방법으로 적분
    - Euler 방법보다 정확도가 높음 (2차 정확도)
    - 계산 비용은 2배 증가 (k1, k2 계산 필요)
    - CA 모듈에서 사용 (SOLVER 설정에서 "heun" 지정 시)
    
    알고리즘:
    1. k1 = f(y)                     # 현재 점에서의 기울기
    2. y_pred = y + dt*k1           # Euler 예측값
    3. k2 = f(y_pred)               # 예측 점에서의 기울기
    4. y_new = y + 0.5*dt*(k1 + k2) # 두 기울기의 평균 사용
    
    Parameters
    ----------
    f : callable
        미분 방정식의 우변 함수: dy/dt = f(y)
    y : array-like
        현재 상태 벡터
    dt : float
        시간 스텝 크기
        
    Returns
    -------
    array-like
        다음 스텝의 상태 벡터
    """
    k1 = f(y)                        # 현재 점에서의 기울기
    y_pred = y + dt * k1            # Euler 예측값
    k2 = f(y_pred)                  # 예측 점에서의 기울기
    return y + 0.5 * dt * (k1 + k2)  # 두 기울기의 평균으로 최종 값 계산

def dtg_rhs(dtg_obj, ATP):
    """
    DTG 시스템의 미분 방정식 우변 함수 생성
    
    기능: DTG 객체와 ATP 값을 받아서 미분 방정식의 우변 함수를 반환
    - DTG 시스템의 E, phi 미분 방정식을 정의
    - θ→φ 결합 (bidirectional coupling) 포함
    - rk4_step, heun_step 등과 함께 사용
    
    미분 방정식:
        dE/dt = g_sync * (ATP - E) - γ * (E - E0)
        dφ/dt = ω0 + α * (E - E0) + k_res * sin(θ_ext - φ)  (θ→φ 결합 포함)
    
    Parameters
    ----------
    dtg_obj : DTGSystem
        DTG 시스템 객체
    ATP : float
        현재 ATP 농도
        
    Returns
    -------
    callable
        미분 방정식의 우변 함수 f(y) = [dE/dt, dφ/dt]
        입력: y = [E, phi] (상태 벡터)
        출력: [dE/dt, dφ/dt] (미분 값 벡터)
    """
    def f(y):
        E, phi = y
        dE = dtg_obj.sync_gain * (ATP - E) - dtg_obj.gamma * (E - dtg_obj.E0)
        dphi = dtg_obj.omega0 + dtg_obj.alpha * (E - dtg_obj.E0)
        # θ→φ 결합 (bidirectional coupling)
        if dtg_obj.theta_ext is not None and dtg_obj.k_res > 0.0:
            dphi += dtg_obj.k_res * np.sin(dtg_obj.theta_ext - phi)
        return np.array([dE, dphi])
    return f

# =============================================================
# 1. dtg_system.py — Digital Twin Guidance (DTG) Layer
# =============================================================
# 목적:
#   - 뉴런의 에너지(E)와 위상(φ)을 동기화시키는 메타 제어 시스템.
#   - Mitochondria(ATP 생성계)와 Soma(HH 발화계)의 상위 조정자 역할.

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
        
        # θ→φ 결합 파라미터 (추가)
        self.k_res = 0.0           # θ→φ 결합 강도
        self.theta_ext = None     # 외부 θ (SynapticResonance.theta)

    def set_resonance(self, theta: float, k_res: float):
        """
        외부 시냅스 위상(theta)과 결합 강도를 설정한다.
        
        Parameters
        ----------
        theta : float
            외부 시냅스 위상 [rad] (SynapticResonance.theta)
        k_res : float
            θ→φ 결합 강도 (0 이상 권장)
        """
        self.theta_ext = float(theta)
        self.k_res = float(max(0.0, k_res))

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
        # (기존) ω0 + α(E−E0)
        dphi = self.omega0 + self.alpha * (self.E - self.E0)
        
        # (추가) θ→φ 결합: + k_res·sin(θ−φ)
        if self.theta_ext is not None and self.k_res > 0.0:
            dphi += self.k_res * np.sin(self.theta_ext - self.phi)
        
        self.phi = (self.phi + dphi * dt) % (2 * np.pi)

        # --- 3) 안정화 처리 (E 폭주 방지; 선택적) ---
        self.E = np.clip(self.E, 0.0, self.E0 * 2.0)

        return self.E, self.phi, dE, dphi

    # =========================================================
    # PATCH #2: Bidirectional phase coupling
    # =========================================================
    def apply_resonance_feedback(self, theta, k_back=0.05):
        """
        시냅스 위상(theta)이 DTG 위상(phi)에 역피드백을 주도록 함.
        theta : SynapticResonance.theta
        k_back : 역결합 계수 (0~0.2 권장)
        """
        # φ ← φ + k_back * sin(θ − φ)
        delta = k_back * np.sin(theta - self.phi)
        self.phi = (self.phi + delta) % (2*np.pi)

# =============================================================
# 2. mitochon_atp.py — Complete Bio-Metabolic Engine
# =============================================================
# 목적:
#   뉴런 내 미토콘드리아의 생리학적 ATP 생성/소비 과정을
#   실제 생화학 반응식 형태로 모델링한 완성형 코드.
#
#   구조:  Glucose + O₂ → ATP + Heat + CO₂ .

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
        # === 초기 상태값 ===
        self.ATP = float(cfg.get("ATP0", 100.0))       # [a.u.]
        self.E_buf = float(cfg.get("Ebuf0", 80.0))     # [a.u.]
        self.Heat = float(cfg.get("Heat0", 0.0))
        self.CO2 = float(cfg.get("CO2_0", 0.0))
        
        # HeatGrid를 위한 N, dx 파라미터 (CONFIG에서 가져올 수 있도록)
        # cfg에 직접 없으면 AXON 설정에서 가져옴 (run_pipeline에서 전달 권장)

        # === 상수 파라미터 ===
        self.k_transfer = cfg.get("k_transfer", 0.4)     # E_buf→ATP 전환 계수
        self.Ploss = cfg.get("Ploss", 1.5)               # 손실율
        self.recover_k = cfg.get("recover_k", 8.0)       # ATP 회복 계수
        self.recover_thresh = cfg.get("recover_thresh", 60.0)
        self.ATP_clip = cfg.get("ATP_clip", (1.0, 110.0))
        self.Ebuf_clip = cfg.get("Ebuf_clip", (15.0, 100.0))
        self.delta_transfer = cfg.get("delta_transfer", 5.0)
        self.c_CO2 = cfg.get("c_CO2", 0.8)

        # === 효율 및 반응 계수 ===
        self.eta0 = cfg.get("eta", 0.60)     # 기본 효율
        self.k_glu = cfg.get("k_glu", 0.8)   # Glucose 기여 계수
        self.k_oxy = cfg.get("k_oxy", 1.2)   # 산소 기여 계수
        self.K_mO2 = cfg.get("K_mO2", 3.0)   # 미하엘리스-멘텐 상수 (O₂ 포화)

        # === 환경 균형 파라미터 ===
        self.k_heat = cfg.get("k_heat", 0.01)      # Heat 감쇠 계수 [1/ms]
        self.k_co2 = cfg.get("k_co2", 0.01)        # CO2 감쇠 계수 [1/ms]
        self.Heat_env = cfg.get("Heat_env", 0.0)   # 환경 Heat 균형값
        self.CO2_env = cfg.get("CO2_env", 0.0)      # 환경 CO2 균형값
        
        # === Heat 확산 파라미터 (확장형) ===
        self.D_H = cfg.get("D_H", 0.0)             # Heat 확산 계수 [cm^2/ms]
        self.dx_heat = cfg.get("dx_heat", 1.0e-3)  # 공간 간격 [cm]
        
        # === HeatGrid 통합 (내부 관리) ===
        self.heatgrid = HeatGrid(
            N=cfg.get("N", 121),
            dx=cfg.get("dx_heat", 1e-3),
            D_H=cfg.get("D_H", 1e-6),
            k_heat=cfg.get("k_heat", 0.01),
            H_env=cfg.get("Heat_env", 0.0)
        )

        # 내부 상태 기록용
        self.last_eta = self.eta0
        self.last_Pin = 0.0
        self.last_dATP = 0.0
        
        # 현재 스텝에서 실제 사용된 효율을 기록
        self.eta = float(self.eta0)

    # ---------------------------------------------------------
    # η(O₂): 산소 농도에 따른 효율
    # ---------------------------------------------------------
    def eta_dynamic(self, O2: float) -> float:
        """
        η(O₂) = η₀ · (O₂ / (O₂ + K_mO₂))
        """
        if O2 <= 0:
            return 0.05
        eta = self.eta0 * (O2 / (O2 + self.K_mO2))
        return float(np.clip(eta, 0.05, self.eta0))

    # ---------------------------------------------------------
    # P_in(Glu,O₂): 에너지 유입량
    # ---------------------------------------------------------
    def power_input(self, Glu: float, O2: float) -> float:
        """
        P_in = k_glu·Glu + k_oxy·O₂
        - Glucose는 Glycolysis, O₂는 ETC
        """
        Pin = self.k_glu * Glu + self.k_oxy * O2
        return float(np.clip(Pin, 0.0, 50.0))

    # ---------------------------------------------------------
    # 미분 방정식 우변 함수 (RK4 등 solver에서 사용)
    # ---------------------------------------------------------
    def derivatives(self, y, Pin, eta, J_use):
        """
        Mitochondria 미분 방정식의 우변 함수
        
        기능: E_buf와 ATP의 미분 방정식을 정의
        - dE_buf/dt = Pin - Ploss
        - dATP/dt = k_transfer * (E_buf - ATP) * eta - J_use
        
        Parameters
        ----------
        y : array-like
            상태 벡터 [E_buf, ATP]
        Pin : float
            에너지 유입량 (power input)
        eta : float
            효율 (0~1)
        J_use : float
            ATP 소비율
            
        Returns
        -------
        array-like
            미분 값 벡터 [dE_buf/dt, dATP/dt]
        """
        E_buf, ATP = y
        dEbuf_dt = (Pin - self.Ploss)
        dATP_dt  = self.k_transfer * (E_buf - ATP) * eta - J_use
        return np.array([dEbuf_dt, dATP_dt])
    
    # ---------------------------------------------------------
    # STEP: ATP 생성/소비 루프
    # ---------------------------------------------------------
    def step(self, dt: float, Glu: float, O2: float, J_use: float = 0.0, 
             H_left: float = None, H_right: float = None):
        """
        한 스텝(dt) 동안의 ATP, E_buf, Heat, CO₂ 갱신.

        Parameters
        ----------
        dt : float
            시간 [ms]
        Glu : float
            혈중 Glucose 농도
        O2 : float
            산소 농도
        J_use : float
            ATP 소비율 (Na/K 펌프 등)
        H_left : float, optional
            왼쪽 이웃 노드의 Heat 값 (확산 계산용)
        H_right : float, optional
            오른쪽 이웃 노드의 Heat 값 (확산 계산용)
        """
        # (1) 에너지 유입 및 효율 계산
        Pin = self.power_input(Glu, O2)
        eta_oxy = self.eta_dynamic(O2)
        
        # 최종 효율: O2로 제한된 효율 vs 피드백으로 낮춰진 기본효율(eta0) 중 작은 값
        eta = min(eta_oxy, getattr(self, "eta0", eta_oxy))
        
        self.last_Pin = Pin
        self.last_eta = eta
        self.eta = float(eta)  # <- 실제 사용 η를 객체에 반영

        # (2-3) E_buf와 ATP 업데이트 (SOLVER 설정에 따라 적분 방법 선택)
        # [PATCH] SOLVER 설정에 따라 적분 방법 선택
        # - "rk4": rk4_step 사용 (더 정확하지만 계산 비용 증가)
        # - 그 외: 기본 Euler 방법 사용
        if CONFIG["SOLVER"]["MITO"] == "rk4":
            # 4차 Runge-Kutta 방법 사용
            # [PATCH] RK4 방법으로 E_buf와 ATP를 동시에 적분
            # 기능: derivatives 메서드를 사용하여 미분 방정식을 정의하고 rk4_step으로 적분
            # 효과: Euler 방법보다 정확도가 높음 (4차 정확도)
            y = np.array([self.E_buf, self.ATP])
            y = rk4_step(lambda y_: self.derivatives(y_, Pin, eta, J_use), y, dt)
            self.E_buf, self.ATP = y
            
            # dATP_prod 계산 (Heat/CO₂ 생성용)
            # [NOTE] RK4 적분 후 E_buf와 ATP가 업데이트되었으므로,
            #        Heat/CO₂ 생성을 위한 dATP_prod는 변화량으로 계산
            #        (RK4 적분 과정에서 이미 효율이 반영됨)
            dATP_prod = 0.0
            if self.E_buf > self.ATP + self.delta_transfer:
                # RK4 적분 후의 변화량을 근사적으로 계산
                # [NOTE] 실제로는 RK4 적분 과정에서 이미 효율이 반영되었으므로,
                #        이 계산은 Heat/CO₂ 생성을 위한 근사값
                dATP = self.k_transfer * (self.E_buf - self.ATP) * dt
                dATP_prod = eta * dATP
                self.last_dATP = dATP_prod
        else:
            # 기본 Euler 방법 사용
            # (2) E_buf 축적
            dEbuf = (Pin - self.Ploss) * dt
            self.E_buf += dEbuf

            # (3) E_buf → ATP 변환
            dATP_prod = 0.0
            if self.E_buf > self.ATP + self.delta_transfer:
                dATP = self.k_transfer * (self.E_buf - self.ATP) * dt
                dATP_prod = eta * dATP
                self.ATP += dATP_prod
                self.E_buf -= dATP
                self.last_dATP = dATP_prod

        # (4) Heat/CO₂ 생성
        if dATP_prod > 0.0:
            self.Heat += (1.0 - eta) * dATP_prod
            self.CO2  += self.c_CO2 * dATP_prod

        # (5) Heat 확산 자동 호출 (HeatGrid 통합)
        if dATP_prod > 0.0:
            self.heatgrid.add_source(0, (1.0 - eta) * dATP_prod)
        self.heatgrid.step(dt)
        self.Heat = float(self.heatgrid.H[0])
        
        # (5.5) CO₂ 감쇠
        self.CO2  -= self.k_co2 * (self.CO2 - self.CO2_env) * dt
        self.CO2 = max(self.CO2, 0.0)

        # (6) ATP 소비
        if J_use > 0.0:
            self.ATP -= J_use * dt

        # (7) ATP 회복 메커니즘
        if self.ATP < self.recover_thresh:
            self.ATP += self.recover_k * (1 - self.ATP / 100.0) * dt

        # (8) 안정화
        self.ATP = float(np.clip(self.ATP, *self.ATP_clip))
        self.E_buf = float(np.clip(self.E_buf, *self.Ebuf_clip))

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
# 2-1. heat_grid.py — Heat Diffusion Grid (1D Spatial)
# =============================================================
# 참고: 섹션 2-1은 Mitochondria(섹션 2)의 Heat 확산을 처리하는
# 보조 클래스로, Mitochondria와 밀접하게 연동되므로 2-1로 번호를 매김.
# 독립 클래스이지만 기능적으로 Mitochondria의 확장 모듈 역할.
# =========================================
# [PATCH 1] Heat 확산용 보조 클래스 추가
# =========================================

class HeatGrid:
    """
    간단한 1차원 열 확산(Heat diffusion) 모델
    ∂H/∂t = D_H·∇²H − k_heat·(H−H_env)
    """
    def __init__(self, N=121, dx=1.0e-3, D_H=1e-6, k_heat=0.01, H_env=0.0):
        self.N = N
        self.dx2 = dx * dx
        self.D_H = D_H
        self.k_heat = k_heat
        self.H_env = H_env
        self.H = np.zeros(N)

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
# 3. hh_soma.py — Hodgkin–Huxley 막전위 모델 (ATP 펌프 + ATP 소비율 포함)
# =============================================================
# 목적:
#   • 뉴런 소마(Soma)의 막전위를 계산하는 기본 전기생리 모델
#   • 나트륨(Na⁺), 칼륨(K⁺), 누설(Leak) 채널 포함
#   • ATP 의존 Na⁺/K⁺ 펌프 및 ATP 소비율(J_use) 계산 포함

import numpy as np


class HHSoma:
    r"""
    Hodgkin–Huxley Soma Model with ATP-dependent Na⁺/K⁺ Pump
    --------------------------------------------------------
    dV/dt = g_Na·m³h·(E_Na−V) + g_K·n⁴·(E_K−V) + g_L·(E_L−V) + I_ext − I_pump
    I_pump = g_pump·(1−e^{−ATP/ATP₀})·(V−E_pump)
    J_use  = g_pump_consume·|I_pump|
    """

    def __init__(self, cfg: dict, ionflow=None):
        # ------------------ 막전위 / 채널 파라미터 ------------------
        self.V = float(cfg["V0"])
        # [PATCH V3] 기본 전도도 저장 (Heat 피로 효과용)
        # 기능: Heat 피로 효과로 인한 전도도 감소를 계산하기 위해 기본값 저장
        # 효과: gNa0, gK0를 저장하여 Heat에 따라 동적으로 전도도 조정 가능
        self.gNa0 = float(cfg["gNa"])  # 기본 Na⁺ 전도도
        self.gK0 = float(cfg["gK"])    # 기본 K⁺ 전도도
        self.gNa = self.gNa0  # 현재 Na⁺ 전도도 (Heat 피로에 따라 변동)
        self.gK = self.gK0   # 현재 K⁺ 전도도 (Heat 피로에 따라 변동)
        self.gL = cfg["gL"]
        self.ENa, self.EK, self.EL = cfg["ENa"], cfg["EK"], cfg["EL"]
        self.spike_thresh = cfg["spike_thresh"]

        # ------------------ ATP 펌프 파라미터 ------------------
        self.use_pump = cfg.get("use_pump", True)
        self.g_pump = cfg.get("g_pump", 0.5)
        self.E_pump = cfg.get("E_pump", -70.0)
        self.ATP0_ref = cfg.get("ATP0_ref", 100.0)

        # ATP 소비율 변환 계수 (µA → ATP/ms)
        self.g_pump_consume = cfg.get("g_pump_consume", 0.005)

        # ------------------ I_pump 스케일링 팩터 ------------------
        self.I_pump_scale = 1.0  # ATP에 따른 펌프 효율 조절

        # ------------------ 게이트 초기값 ------------------
        self.m, self.h, self.n = 0.05, 0.60, 0.32
        
        # ------------------ IonFlowDynamics 통합 (선택적) ------------------
        self.ionflow = ionflow
        
        # [PATCH V3] Heat 피로 감쇠 상수
        # 기능: Heat 증가에 따른 전도도 감소 비율 정의
        # 효과: Heat 1°C 증가당 전도도 1% 감소 (기본값: beta_heat = 0.01)
        #   - Heat = 37°C: 전도도 100%
        #   - Heat = 47°C: 전도도 90% (10% 감소)
        #   - Heat = 57°C: 전도도 80% (20% 감소)
        self.beta_heat = cfg.get("beta_heat", 0.01)  # Heat 1°C 증가당 전도도 1% 감소

    # =========================================================
    # α(V), β(V) — 게이트 개폐 속도 상수
    # =========================================================
    @staticmethod
    def _am(V):
        """Na⁺ 활성화 (m 게이트) α(V)"""
        x = V + 40.0
        return 0.1*x/(1.0 - np.exp(-x/10.0)) if abs(x) > 1e-6 else 1.0

    @staticmethod
    def _bm(V):
        """Na⁺ 활성화 (m 게이트) β(V)"""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _ah(V):
        """Na⁺ 비활성화 (h 게이트) α(V)"""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _bh(V):
        """Na⁺ 비활성화 (h 게이트) β(V)"""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def _an(V):
        """K⁺ 활성화 (n 게이트) α(V)"""
        x = V + 55.0
        return 0.01*x/(1.0 - np.exp(-x/10.0)) if abs(x) > 1e-6 else 0.1

    @staticmethod
    def _bn(V):
        """K⁺ 활성화 (n 게이트) β(V)"""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    # =========================================================
    # 미분 방정식 우변 함수 (RK4 등 solver에서 사용)
    # =========================================================
    def derivatives(self, y, I_ext, ATP, Heat=37.0):
        """
        Hodgkin-Huxley 미분 방정식의 우변 함수 (Heat, Na+ 피드백 포함)
        
        기능: V, m, h, n의 미분 방정식을 정의
        - dV/dt = I_Na + I_K + I_L + I_ext - I_pump
        - dm/dt = am*(1-m) - bm*m
        - dh/dt = ah*(1-h) - bh*h
        - dn/dt = an*(1-n) - bn*n
        
        Parameters
        ----------
        y : array-like
            상태 벡터 [V, m, h, n]
        I_ext : float
            외부 전류
        ATP : float
            ATP 농도
        Heat : float, optional
            온도 [°C] (기본값: 37.0°C, Q10 효과 및 Heat 피로 효과 적용용)
            
        Returns
        -------
        tuple
            (미분 값 벡터 [dV/dt, dm/dt, dh/dt, dn/dt], I_pump)
            - 미분 값 벡터: [dV/dt, dm/dt, dh/dt, dn/dt]
            - I_pump: ATP 펌프 전류 (J_use 계산용)
        """
        V, m, h, n = y
        am, bm = self._am(V), self._bm(V)
        ah, bh = self._ah(V), self._bh(V)
        an, bn = self._an(V), self._bn(V)
        
        # [PATCH] 온도 의존성 적용 (Q10 효과)
        # 기능: 온도에 따라 모든 게이트 속도 상수를 스케일링
        # 효과: 온도가 높을수록 게이트 반응 속도가 빨라짐 (생리학적 현실 반영)
        Q10 = 3.0
        T_diff = (Heat - 37.0)
        rate_scale = Q10 ** (T_diff / 10.0)
        am *= rate_scale; bm *= rate_scale
        ah *= rate_scale; bh *= rate_scale
        an *= rate_scale; bn *= rate_scale
        
        # [PATCH V3] Heat 피로 효과: gNa, gK 감소
        # 기능: Heat 증가에 따라 Na⁺, K⁺ 채널 전도도 감소
        # 효과: 온도가 높을수록 채널 전도도가 감소하여 피로 효과 발생
        #   - Heat = 37°C: 전도도 100%
        #   - Heat = 47°C: 전도도 90% (10% 감소)
        #   - Heat = 57°C: 전도도 80% (20% 감소)
        #   - 최소 전도도: 10% (완전 차단 방지)
        fatigue_scale = max(0.1, 1.0 - self.beta_heat * max(0.0, Heat - 37.0))
        gNa = self.gNa0 * fatigue_scale
        gK = self.gK0 * fatigue_scale
        
        dmdt = am*(1-m) - bm*m
        dhdt = ah*(1-h) - bh*h
        dndt = an*(1-n) - bn*n
        I_Na = gNa*(m**3)*h*(self.ENa-V)
        I_K  = gK*(n**4)*(self.EK-V)
        I_L  = self.gL*(self.EL-V)
        I_pump = self.g_pump*(1-np.exp(-ATP/self.ATP0_ref))*(V-self.E_pump)
        dVdt = I_Na + I_K + I_L + I_ext - I_pump
        return np.array([dVdt, dmdt, dhdt, dndt]), I_pump
    
    # =========================================================
    # Step 함수 — 시간 적분
    # =========================================================
    def step(self, dt: float, I_ext: float = 0.0, ATP: float = 100.0,
             ENa_override: float = None, EK_override: float = None,
             Heat: float = 37.0):
        """
        한 스텝(dt[ms]) 적분 수행:
        - 게이트 갱신
        - 이온 전류 계산
        - ATP 펌프 전류 및 ATP 소비율 계산
        
        Parameters
        ----------
        dt : float
            시간 스텝 [ms]
        I_ext : float
            외부 전류
        ATP : float
            ATP 농도
        ENa_override : float, optional
            ENa 역전위 override 값 (None이면 self.ENa 사용)
        EK_override : float, optional
            EK 역전위 override 값 (None이면 self.EK 사용)
        Heat : float, optional
            온도 [°C] (기본값: 37.0°C, Q10 효과 적용용)
        """
        V = self.V
        
        # 역전위 선택 (override 우선)
        ENa = self.ENa if ENa_override is None else ENa_override
        EK  = self.EK  if EK_override  is None else EK_override

        # ------------------ 0) 온도 의존성 (Q10 효과) 계산 ------------------
        # [PATCH] 온도 의존성 추가 (Q10 효과)
        # 기능: 온도에 따라 게이트 속도 상수(am, bm, ah, bh, an, bn)를 스케일링
        # 효과: 온도가 높을수록 게이트 반응 속도가 빨라짐 (생리학적 현실 반영)
        # Q10: 10도 증가 시 반응 속도가 Q10배 증가 (일반적으로 2-4)
        Q10 = 3.0
        T_diff = (Heat - 37.0)
        rate_scale = Q10 ** (T_diff / 10.0)

        # ------------------ 1-4) 게이트 및 막전위 업데이트 (SOLVER 설정에 따라 적분 방법 선택) ------------------
        # [PATCH] SOLVER 설정에 따라 적분 방법 선택
        # - "rk4": rk4_step 사용 (더 정확하지만 계산 비용 증가)
        # - 그 외: 기본 Euler 방법 사용
        if CONFIG["SOLVER"]["HH"] == "rk4":
            # 4차 Runge-Kutta 방법 사용
            # [PATCH] RK4 방법으로 V, m, h, n을 동시에 적분
            # 기능: derivatives 메서드를 사용하여 미분 방정식을 정의하고 rk4_step으로 적분
            # 효과: Euler 방법보다 정확도가 높음 (4차 정확도), 게이트와 막전위를 동시에 적분
            # 주의: ENa_override, EK_override를 사용하려면 derivatives 메서드를 수정해야 함
            #       현재는 self.ENa, self.EK를 사용하므로 override가 적용되지 않음
            y = np.array([self.V, self.m, self.h, self.n])
            # [PATCH V3] derivatives가 이제 (미분값, I_pump) 튜플을 반환하므로 수정
            deriv_func = lambda y_: self.derivatives(y_, I_ext, ATP, Heat)[0]
            y = rk4_step(deriv_func, y, dt)
            self.V, self.m, self.h, self.n = y
            
            # [0,1] 범위로 제한
            self.m, self.h, self.n = np.clip([self.m, self.h, self.n], 0.0, 1.0)
            self.V = np.nan_to_num(self.V, nan=-70.0, posinf=120.0, neginf=-120.0)
            
            # [PATCH V3] ATP 펌프 전류 계산 (derivatives에서 직접 가져옴)
            _, I_pump = self.derivatives(y, I_ext, ATP, Heat)
            
            # 전류 계산 (반환값용, Heat 피로 효과 반영)
            V_curr = self.V
            fatigue_scale = max(0.1, 1.0 - self.beta_heat * max(0.0, Heat - 37.0))
            gNa_curr = self.gNa0 * fatigue_scale
            gK_curr = self.gK0 * fatigue_scale
            INa = gNa_curr * (self.m ** 3) * self.h * (ENa - V_curr)
            IK  = gK_curr * (self.n ** 4) * (EK  - V_curr)
            IL  = self.gL * (self.EL - V_curr)
        else:
            # 기본 Euler 방법 사용
            # ------------------ 1) 게이트 업데이트 (온도 의존성 적용) ------------------
            am, bm = self._am(V), self._bm(V)
            ah, bh = self._ah(V), self._bh(V)
            an, bn = self._an(V), self._bn(V)
            
            # [PATCH] 온도 의존성 적용 (Q10 효과)
            # 기능: 온도에 따라 모든 게이트 속도 상수를 스케일링
            # 효과: 온도가 높을수록 게이트 반응 속도가 빨라짐
            am *= rate_scale; bm *= rate_scale
            ah *= rate_scale; bh *= rate_scale
            an *= rate_scale; bn *= rate_scale

            self.m += dt * (am * (1.0 - self.m) - bm * self.m)
            self.h += dt * (ah * (1.0 - self.h) - bh * self.h)
            self.n += dt * (an * (1.0 - self.n) - bn * self.n)

            # [0,1] 범위로 제한
            self.m, self.h, self.n = np.clip([self.m, self.h, self.n], 0.0, 1.0)

            # ------------------ 2) 채널 전류 계산 (Heat 피로 효과 적용) ------------------
            # [PATCH V3] Heat 피로 효과: gNa, gK 감소
            # 기능: Heat 증가에 따라 Na⁺, K⁺ 채널 전도도 감소
            # 효과: 온도가 높을수록 채널 전도도가 감소하여 피로 효과 발생
            fatigue_scale = max(0.1, 1.0 - self.beta_heat * max(0.0, Heat - 37.0))
            gNa_curr = self.gNa0 * fatigue_scale
            gK_curr = self.gK0 * fatigue_scale
            INa = gNa_curr * (self.m ** 3) * self.h * (ENa - V)
            IK  = gK_curr * (self.n ** 4) * (EK  - V)
            IL  = self.gL * (self.EL - V)

            # ------------------ 3) ATP 펌프 전류 계산 ------------------
            I_pump = 0.0
            if self.use_pump:
                # ATP 농도에 따라 포화되는 비선형 함수
                factor = (1.0 - np.exp(-ATP / self.ATP0_ref))
                # --- 수정 보완점 #2: ATP 농도에 따른 펌프 억제 추가 ---
                K_ATP = 10.0  # ATP affinity constant
                ATP_mod = ATP / (K_ATP + ATP)
                I_pump = self.g_pump * self.I_pump_scale * factor * ATP_mod * (V - self.E_pump)

            # ------------------ 4) 막전위 갱신 ------------------
            dV = INa + IK + IL + I_ext - I_pump
            self.V = np.nan_to_num(V + dt * dV, nan=-70.0, posinf=120.0, neginf=-120.0)
        
        # ------------------ 4.5) IonFlowDynamics 자동 업데이트 (있는 경우) ------------------
        if self.ionflow is not None:
            self.ionflow.V[:] = self.V
            self.ionflow.step(dt)

        # ------------------ 5) ATP 소비율 계산 ------------------
        # [PATCH V3] Na⁺ 기반 J_use 계산
        # 기능: Na⁺ 내부 농도에 비례한 ATP 소모
        # 효과: Na⁺ 농도가 높을수록 ATP 소모 증가 (생리학적 현실 반영)
        #   - Na⁺ 농도가 높으면 Na/K 펌프가 더 많이 작동하여 ATP 소모 증가
        #   - Na_norm = Na_i / 50.0 (50 mM 기준 정규화, 0~2 범위로 제한)
        #   - J_use = g_pump_consume * |I_pump| * Na_norm
        if self.ionflow is not None:
            Na_i = np.mean(self.ionflow.ions["Na"]["C"])
        else:
            Na_i = 15.0  # 기본 Na⁺ 내부 농도 [mM]
        Na_norm = np.clip(Na_i / 50.0, 0.0, 2.0)  # 50 mM 기준 정규화 (0~2 범위)
        J_use = self.g_pump_consume * abs(I_pump) * Na_norm

        # ③ HHSoma.step() 리턴값 통일 (딕셔너리)
        return {"V": self.V, "INa": INa, "IK": IK, "IL": IL, "I_pump": I_pump, "J_use": J_use}

    # =========================================================
    # I_pump 스케일링 설정
    # =========================================================
    def set_I_pump_scale(self, scale: float):
        """ATP에 따른 펌프 효율 조절"""
        self.I_pump_scale = float(np.clip(scale, 0.0, 1.0))

    # =========================================================
    # Spike 감지 함수
    # =========================================================
    def spiking(self) -> bool:
        """막전위가 임계값을 초과하면 스파이크로 간주"""
        return self.V > self.spike_thresh
    
    # =========================================================
    # PATCH #1: Nernst reversal update
    # =========================================================
    @staticmethod
    def nernst(E_out, E_in, z=1, T_K=310.0):
        """
        Nernst 방정식을 사용하여 역전위를 계산한다.
        
        Parameters
        ----------
        E_out : float
            세포 외부 이온 농도 [mM]
        E_in : float
            세포 내부 이온 농도 [mM]
        z : int
            이온의 전하 (Na⁺, K⁺: 1, Ca²⁺: 2, Cl⁻: -1)
        T_K : float
            온도 [K] (기본값: 310.0 K = 37°C)
        
        Returns
        -------
        float
            역전위 [mV]
        """
        # R=8.314 J/mol/K, F=96485 C/mol →  (R*T)/(z*F) ≈ 26.73 mV at 310K (z=1)
        RT_over_F = 26.73  # mV
        return RT_over_F/z * np.log(max(1e-12, E_out)/max(1e-12, E_in))
    
    def update_reversal_potentials(self, ionflow):
        """
        IonFlowDynamics 결과(농도장)를 기반으로 Nernst 전위를 갱신한다.
        ENa, EK, ECa, ECl을 동적으로 반영.
        """
        # 세포 내외 이온 농도
        # 내부([i])는 평균 50~70%, 외부([o])는 나머지 (단위: mM)
        Na_i = np.mean(ionflow.ions["Na"]["C"]) * 0.6
        Na_o = np.mean(ionflow.ions["Na"]["C"]) * 0.4
        K_i  = np.mean(ionflow.ions["K"]["C"]) * 0.7
        K_o  = np.mean(ionflow.ions["K"]["C"]) * 0.3
        Ca_i = np.mean(ionflow.ions["Ca"]["C"]) * 0.9
        Ca_o = np.mean(ionflow.ions["Ca"]["C"]) * 0.1
        Cl_i = np.mean(ionflow.ions["Cl"]["C"]) * 0.3
        Cl_o = np.mean(ionflow.ions["Cl"]["C"]) * 0.7

        # Nernst 식: E = (RT/zF) * ln([out]/[in]) [V] → [mV]
        self.ENa = self.nernst(Na_o, Na_i, z=1)
        self.EK  = self.nernst(K_o, K_i, z=1)
        self.ECa = self.nernst(Ca_o, Ca_i, z=2)
        self.ECl = -self.nernst(Cl_o, Cl_i, z=1)  # 음이온이므로 부호 반전

# =============================================================
# 4. ionflow_dynamics.py — 다중 이온 확산/전기이동 모델
# =============================================================
# 목적:
#   • 막전위(Vm)에 따라 Na⁺, K⁺, Ca²⁺, Cl⁻의 이동 계산
#   • 전기장(∇V)에 따른 drift + 확산(diffusion)을 반영

import numpy as np

class IonFlowDynamics:
    r"""
    IonFlowDynamics — Multi-Ion Diffusion + Electric Drift
    ------------------------------------------------------
    ∂C_i/∂t = D_i∇²C_i − μ_i·z_i·F·∇V
    """

    def __init__(self, cfg: dict):
        self.N = cfg.get("N", 121)
        self.dx = cfg.get("dx", 1e-3)
        self.V = np.full(self.N, cfg.get("Vrest", -70.0))
        self.F = 96485.0  # 패러데이 상수 [C/mol]
        # [PATCH] 이온 이동도 스케일 조정 (1e-8 → 1e-9)
        # 기능: 전기장에 의한 이온 drift 효과의 강도를 조정
        # 효과: 장기 시뮬레이션 안정성 강화 (이온 농도 급격한 변화 방지)
        #   - 작은 값: drift 효과 감소 → 확산 중심, 안정적
        #   - 큰 값: drift 효과 증가 → 전기장 영향 강화, 불안정 가능
        self.mu_scale = 1e-9  # [PATCH] 이동도 스케일 (1e-8 → 1e-9, 장기 시뮬 안정성 강화)

        # 4종 이온 초기화
        self.ions = {
            "Na": {"C": np.full(self.N, 15.0), "D": 1.33e-5, "z": +1},
            "K":  {"C": np.full(self.N,140.0), "D": 1.96e-5, "z": +1},
            "Ca": {"C": np.full(self.N, 0.0001), "D": 0.79e-5, "z": +2},
            "Cl": {"C": np.full(self.N, 5.0), "D": 2.03e-5, "z": -1},
        }

    def laplacian(self, arr):
        """1D 중심차분 ∇²C"""
        lap = np.zeros_like(arr)
        lap[1:-1] = arr[:-2] - 2*arr[1:-1] + arr[2:]
        return lap / (self.dx**2)

    def step(self, dt: float):
        """한 스텝(dt[ms]) 이온 농도 업데이트"""
        dVdx = np.gradient(self.V, self.dx)
        for ion, d in self.ions.items():
            D, z, C = d["D"], d["z"], d["C"]
            diff = D * self.laplacian(C)
            drift = -self.mu_scale * z * self.F * dVdx * C
            C += dt * (diff + drift)
            d["C"] = np.clip(C, 0.0, None)

        # 전하 중립 보정
        total_q = sum(d["z"]*np.sum(d["C"]) for d in self.ions.values())
        if abs(total_q) > 1e-3:
            corr = -total_q / (self.N * len(self.ions))
            for ion, d in self.ions.items():
                d["C"] += corr * np.sign(d["z"])
                # [PATCH] 전하 중립 보정 후 추가 클램프
                # 기능: 전하 중립 보정으로 인해 음수 농도가 발생할 수 있으므로 0 이상으로 제한
                # 효과: 이온 농도가 음수가 되는 것을 방지하여 안정성 향상
                d["C"] = np.clip(d["C"], 0.0, None)  # ← 추가 클램프

        return {ion: d["C"] for ion, d in self.ions.items()}

# =============================================================
# 5.myelinated_axon.py — 물리적 도약전도 (Saltatory Conduction)
# =============================================================
# 목적:
#   - 소마(Soma)에서 전송된 활동전위가 축삭을 따라 도약전도(saltatory conduction)로 전달되는 과정 모델링
#   - 노드(Node)와 인터노드(Internode) 구간을 구분
#   - 각 구간의 확산(D), 막용량(Cm), 누설전도(gL) 상이
#   - 노드에서만 빠른 Na⁺ 채널이 활성화되어 도약 전위 형성
#   - 시간 감쇠(Lambda), 에너지 감쇠(gamma_extra), α-펄스 자극까지 통합

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

        # α-pulse parameter (from global CONFIG, optional)
        try:
            import sys
            if hasattr(sys.modules.get('__main__', None), 'CONFIG'):
                CONFIG = sys.modules['__main__'].CONFIG
                A = CONFIG.get("ALPHA", {})
                self.alpha_I0 = A.get("I0", 0.0)
                self.alpha_tr = A.get("tau_r", 0.5)
                self.alpha_td = A.get("tau_d", 3.0)
            else:
                # 기본값 사용
                self.alpha_I0 = 0.0
                self.alpha_tr = 0.5
                self.alpha_td = 3.0
        except (ImportError, AttributeError):
            # CONFIG가 없을 때 기본값 사용
            self.alpha_I0 = 0.0
            self.alpha_tr = 0.5
            self.alpha_td = 3.0
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
        """
        ATP-dependent Na+ conductance modulation
        ATP 수준에 따라 Na+ 채널 전도도를 동적으로 조정합니다.
        """
        INa = np.zeros(self.N)
        idx = np.where(self.IS_NODE)[0]
        if idx.size:
            m3h = (self.m_node[idx] ** 3) * self.h_node[idx]
            
            # --- PATCH: ATP-dependent Na conductance modulation ---
            # ATP 수준에 따라 Na+ 전도도를 조정 (ATP가 높을수록 전도도 증가)
            A = getattr(self, "ATP_level", None)
            if A is not None:
                A0 = 100.0        # baseline ATP (tune as needed)
                dA = 50.0         # ATP scaling range
                lambda_A = 0.25   # modulation gain
                gNa_eff = self.node_gNa * (1.0 + lambda_A * np.tanh((A - A0) / dA))
            else:
                gNa_eff = self.node_gNa
            
            INa[idx] = gNa_eff * m3h * (self.node_ENa - self.V[idx])
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

# =============================================================
# 6. ca_vesicle.py — Ca²⁺ Vesicle (Spike-triggered Alpha kernels)
# =============================================================
# 목적:
#   • 스파이크 시각 목록 {t_k}에 의해 유도되는 Ca²⁺ 유입(α-커널 합)과
#     ATP-의존 펌프에 의한 Ca 제거를 함께 모델링.
#   • 정규화 시그널 S=(Ca−C0)/(Cmax−C0) 및 상태 레이블(under/normal/alert) 제공.

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
        # --- 파라미터/초기값 ---
        self.C0: float = float(cfg["C0"])
        self.Cmax: float = float(cfg["Cmax"])
        self.A: float = float(cfg["A"])            # α-커널 스케일
        self.tau_r_s: float = float(cfg["tau_r"])  # [s]
        self.tau_d_s: float = float(cfg["tau_d"])  # [s]
        self.k_c: float = float(cfg["k_c"])        # 펌프 계수(ATP 비례)
        self.k_atp_per_Ca: float = float(cfg.get("k_atp_per_Ca", 1.0))  # Ca당 ATP 소비 계수
        self.dt_ms: float = float(dt_ms)           # 적분 스텝 [ms]
        self.max_spike_memory_ms: float = float(cfg["max_spike_memory_ms"])

        # τ_d > τ_r 되도록 자동 보정 (수치/물리 안정)
        if not (self.tau_d_s > self.tau_r_s > 0.0):
            # 매우 근접하거나 역전된 경우 소폭 조정
            eps = 1e-4
            base = max(self.tau_r_s, 1e-3)
            self.tau_r_s = base
            self.tau_d_s = base + max(eps, 0.01 * base)

        # --- 상태 변수 ---
        self.t_ms: float = 0.0
        self.Ca: float = float(self.C0)
        self.spike_times: List[float] = []     # [ms]
        self.events: List[VesicleEvent] = []

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
    def step(self, ATP: float):
        """
        한 스텝(dt_ms) 진행:
          • α-커널 합으로 유입 계산
          • ATP-의존 펌프로 Ca 제거
          • Ca, S, status 업데이트 및 이벤트 로깅

        Parameters
        ----------
        ATP : float
            미토콘드리아 층에서 전달되는 ATP 레벨(무차원 스케일).
            k_c·ATP가 클수록 Ca 제거가 가속.

        Returns
        -------
        tuple
            (VesicleEvent, J_Ca_rate)
            - VesicleEvent: Ca 이벤트 정보
            - J_Ca_rate: Ca 펌프 ATP 소비율 [ATP/ms]
        """
        # 시간 진행
        self.t_ms += self.dt_ms

        # 스파이크 메모리 관리
        self._trim_spike_memory()

        # α-커널 유입 합
        influx = 0.0
        if self.spike_times:
            for ts in self.spike_times:
                influx += self.A * self._alpha_kernel(self.t_ms - ts)

        # 펌프(ATP 의존 제거) — (Ca − C0)에 비례
        pump = self.k_c * float(ATP) * max(0.0, (self.Ca - self.C0))

        # 미분항: dCa/dt [per second]
        dCa_dt = influx - pump

        # 이산 적분: Δt = dt_ms/1000 [s]
        self.Ca += dCa_dt * (self.dt_ms / 1000.0)

        # 안전: 지나친 음수 방지(물리 클램프는 하지 않되 하한만)
        self.Ca = max(self.Ca, self.C0 * 0.1)

        # 정규화 및 상태
        denom = max(1e-12, (self.Cmax - self.C0))
        S = (self.Ca - self.C0) / denom
        status = "under" if S < 0.0 else ("normal" if S <= 1.0 else "alert")

        # 이벤트 기록 (메모리 과다 방지: 필요 시 슬라이싱)
        ev = VesicleEvent(t_ms=float(self.t_ms), Ca=float(self.Ca), S=float(S), status=status)
        self.events.append(ev)
        if len(self.events) > 10000:
            self.events = self.events[-5000:]

        # Ca 펌프 ATP 소비율 계산 [ATP/ms]
        J_Ca_rate = self.k_atp_per_Ca * self.k_c * float(ATP) * max(0.0, (self.Ca - self.C0))
        
        return ev, J_Ca_rate

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
       
# =============================================================
# 7. ptp_plasticity.py — Post-Tetanic Potentiation (PTP) only
# =============================================================
# 개념/수식 요약
# -------------------------------------------------------------
# PTP는 고빈도 자극(tetanus) 후 수 초~수십 초 동안
#  방출확률 p 또는 시냅스 이득 w를 일시적으로 증가시키는
#  Ca²⁺-의존 단기 가소성.
#
# 상태변수:
#   R(t) : PTP “잔여 강화량”(무차원, 0 이상)
#
# 동역학:
#   dR/dt = -R / τ_ptp  +  A(Ca_res) · Σ_k δ(t - t_k)
#
#   • τ_ptp : PTP 감쇠 시정수 (초 단위; 실험적으로 10~60 s)
#   • A(Ca_res) : 스파이크 직후 잔여 Ca²⁺(또는 Ca 이벤트 S)에 비례하는 증분
#       A(Ca) = g_ptp · (Ca_norm)^n / ( (Ca_norm)^n + K^n )
#     여기서 Ca_norm = clamp( (Ca - C0)/(Cmax - C0), 0, 1 )
#
# 적용:
#   p_eff = clamp( p0 * (1 + R), 0, 1 )
#   또는
#   w_eff = w0 * (1 + R)
#
# 주의:
#   • 본 클래스는 “PTP 항만” 제공. 단기 facilitation/depression(τ_f, τ_d)은 포함하지 않음.
#   • Ca 입력은 네가 가진 CaVesicle의 (Ca, S) 중 하나를 전달하면 됨.
# =============================================================

from dataclasses import dataclass

@dataclass
class PTPConfig:
    tau_ptp_s: float = 20.0     # PTP 감쇠 시정수 [s]
    g_ptp: float    = 1.2       # PTP 첨가 이득(스파이크당 최대 증분 스케일)
    K_half: float   = 0.25      # Ca_norm의 반포화점 (0~1 범위)
    hill_n: int     = 3         # Hill 계수 (비선형 민감도)
    R_clip: tuple   = (0.0, 3.0)# R(t) 안전 범위 (증폭 한계)
    # p0, w0는 외부 시냅스가 갖고 있고, 여기서는 효과만 계산해 제공

class PTPPlasticity:
    r"""
    Post-Tetanic Potentiation (PTP) — Ca²⁺-dependent short-term potentiation

    State:
        R(t) ≥ 0  : residual potentiation, decays with τ_ptp

    Dynamics:
        dR/dt = -R/τ_ptp + A(Ca_res)·Σ δ(t - t_k)

    Effective modulation:
        p_eff = clamp(p0*(1 + R), 0, 1)
        w_eff = w0*(1 + R)
    """
    def __init__(self, cfg: PTPConfig):
        self.cfg = cfg
        self.R = 0.0   # 초기 PTP 잔여 강화량
        self.t_ms = 0.0

    # --- 내부: Hill형 증분 함수 A(Ca_norm) ---
    def _A_from_CaNorm(self, Ca_norm: float) -> float:
        Ca_norm = max(0.0, min(1.0, float(Ca_norm)))
        n = self.cfg.hill_n
        num = (Ca_norm ** n)
        den = (Ca_norm ** n) + (self.cfg.K_half ** n)
        return self.cfg.g_ptp * (num / den) if den > 0 else 0.0

    # --- 외부에서 스파이크 알림: Ca 또는 S를 넣어 증분 ---
    def on_spike(self, *, Ca: float = None, C0: float = None, Cmax: float = None, S: float = None):
        """
        스파이크 직후 호출.
        인자:
          • (옵션1) S : CaVesicle의 정규화 지표 S ∈ ℝ (보통 0~1)
          • (옵션2) Ca, C0, Cmax : 실농도 기반 입력(단위 일치 필수)
                Ca_norm = clamp((Ca - C0)/(Cmax - C0), 0, 1)
        둘 중 하나만 주면 됨. 둘 다 주면 S가 우선.
        """
        if S is None:
            # Ca로부터 정규화 S 계산
            if (Ca is None) or (C0 is None) or (Cmax is None) or (Cmax <= C0):
                raise ValueError("Provide either S, or Ca with (C0, Cmax).")
            Ca_norm = (Ca - C0) / (Cmax - C0)
        else:
            Ca_norm = float(S)

        dR = self._A_from_CaNorm(Ca_norm)
        self.R = max(self.cfg.R_clip[0], min(self.R + dR, self.cfg.R_clip[1]))

    # --- 시간 전진(지수 감쇠) ---
    def step(self, dt_ms: float):
        """
        dt_ms 만큼 시간 전진 (지수감쇠).
        """
        self.t_ms += dt_ms
        tau_ms = max(1e-9, self.cfg.tau_ptp_s * 1000.0)
        # 연속시간 해(지수감쇠) 사용: R ← R * exp(-dt/τ)
        decay = pow(2.718281828, -dt_ms / tau_ms)
        self.R *= decay
        # 안정화
        self.R = max(self.cfg.R_clip[0], min(self.R, self.cfg.R_clip[1]))
        return self.R

    # --- 외부에 줄 모듈레이션 팩터 ---
    def p_eff(self, p0: float) -> float:
        """방출확률 p의 PTP 적용값"""
        return max(0.0, min(1.0, p0 * (1.0 + self.R)))

    def w_eff(self, w0: float) -> float:
        """가중치/시냅스 이득의 PTP 적용값(상한은 외부에서 관리)"""
        return w0 * (1.0 + self.R)
        
# =============================================================
# 8. metabolic_feedback.py — Heat·CO₂·Ca 기반 대사 피드백 루프
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
        # (1) Heat ↑ → 효율 η0 낮추기 (기본 효율의 이동)
        # -----------------------------------------------------
        delta_eta0 = - self.cfg["beta_heat"] * max(0.0, self.mito.Heat)
        new_eta0 = self.eta_base + delta_eta0
        self.mito.eta0 = float(np.clip(new_eta0, 0.05, 1.0))

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
# 9. synaptic_resonance.py — Ca²⁺ 기반 커플링 게인 공명 모델
# =============================================================
# 목적:
#   시냅스의 내부 위상 θ(t)가 상위 발진자(DTG System)의 위상 φ(t)에
#   동기화(phase locking)되는 과정을 모델링한다.
#
#   이때 결합 강도(coupling gain, K)가 칼슘 신호(S)에 의해
#   동적으로 조절되는 구조를 포함한다.
#
#   ┌──────────────────────────────────────┐
#   │ dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)   │
#   └──────────────────────────────────────┘
#
#   • θ : 시냅스 고유 위상 (synaptic phase)
#   • φ : 상위 발진 위상 (DTG phase)
#   • ω : 고유 위상속도 (intrinsic angular frequency)
#   • K : 기본 커플링 게인 (baseline coupling strength)
#   • λ : Ca²⁺ 민감도 (coupling modulation coefficient)
#   • S : Ca²⁺ 정규화 농도 (0~1)
#
#   ⇒ Ca²⁺가 높을수록 결합이 강해지고(동기화↑),
#     Ca²⁺가 낮을수록 각자 독립 진동(비동기화).
#
#   물리적 의미:
#     “Ca²⁺는 시냅스 공명의 커플링 게인으로 작용한다.”
#
# =============================================================

import numpy as np

class SynapticResonance:
    r"""
    SynapticResonance — Ca²⁺-modulated Phase Coupling Resonator
    ------------------------------------------------------------
    Differential equation:
        dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)

    where:
        • θ : synaptic phase (local oscillator)
        • φ : global DTG phase (energy-phase driver)
        • ω : intrinsic angular frequency [rad/ms]
        • K : baseline coupling gain (0 ≤ K ≤ 1)
        • λ : Ca²⁺ sensitivity (dimensionless)
        • S : normalized Ca²⁺ activity (0~1)

    Effective coupling:
        K_eff = K·(1 + λ·S)

    Integration (discrete time):
        θ_{t+Δt} = (θ_t + (ω + K_eff·sin(φ−θ_t))·Δt) mod 2π

    Biophysical interpretation:
        - High Ca²⁺ (S↑) → coupling gain ↑ → stronger phase-lock to φ
        - Low Ca²⁺ (S↓) → coupling gain ↓ → weak synchronization
    """

    def __init__(self, omega: float = 1.0, K: float = 0.05, lambda_ca: float = 1.0):
        """
        Parameters
        ----------
        omega : float
            Intrinsic oscillation frequency [rad/ms].
        K : float
            Baseline coupling strength (0 ≤ K ≤ 1).
        lambda_ca : float
            Calcium modulation coefficient λ (coupling sensitivity).
        """
        self.theta = 0.0          # Current synaptic phase θ [rad]
        self.omega = omega        # Intrinsic angular velocity
        self.K = K                # Base coupling gain
        self.lambda_ca = lambda_ca  # Ca²⁺-dependent modulation factor

    # -------------------------------------------------------------
    # Step Integration
    # -------------------------------------------------------------
    def step(self, dt: float, phi: float, S: float):
        r"""
        Integrate phase θ over dt [ms].

        Equation:
            dθ/dt = ω + K·sin(φ − θ)·(1 + λ·S)
            θ(t+Δt) = (θ + dθ·Δt) mod 2π

        Parameters
        ----------
        dt : float
            Integration step [ms].
        phi : float
            DTG (driver) phase [rad].
        S : float
            Normalized calcium activity (0~1).

        Returns
        -------
        tuple(float, float)
            (θ, Δθ) → (synaptic phase, phase difference φ−θ)
        """

        # 1) 유효 커플링 게인 계산 (Ca²⁺ 영향 반영)
        #    K_eff = K * (1 + λ·S)
        K_eff = self.K * (1.0 + self.lambda_ca * S)

        # 2) 위상 변화율 계산
        #    dθ/dt = ω + K_eff·sin(φ−θ)
        dtheta_dt = self.omega + K_eff * np.sin(phi - self.theta)

        # 3) 이산 적분 (Euler)
        #    θ ← θ + dθ·dt
        self.theta += dtheta_dt * dt

        # 4) 위상 wrap (0~2π)
        self.theta = self.theta % (2 * np.pi)

        # 5) 위상차 Δθ 계산
        delta_phi = (phi - self.theta)

        return self.theta, delta_phi

    # -------------------------------------------------------------
    # Spike-triggered learning update
    # -------------------------------------------------------------
    def on_spike(self, R: float, phi: float):
        """
        스파이크 시 호출되는 학습 업데이트.
        PTP 강화량(R)과 DTG 위상(φ)을 사용하여 시냅스 파라미터를 갱신.

        Parameters
        ----------
        R : float
            PTP 잔여 강화량 (potentiation factor)
        phi : float
            DTG 위상 [rad]
        """
        # PTP에 따라 커플링 강도 증가
        if R > 0.0:
            # 위상 동조에 따라 추가 보정 가능
            phase_bonus = 0.1 * np.cos(phi - self.theta)
            self.K = min(1.0, self.K + 0.01 * R * (1.0 + phase_bonus))

    # -------------------------------------------------------------
    # Optional helper: instantaneous coupling gain
    # -------------------------------------------------------------
    def coupling_gain(self, S: float) -> float:
        """현재 Ca²⁺ 값(S)에 따른 실시간 유효 커플링 게인 반환"""
        return self.K * (1.0 + self.lambda_ca * S)

# =============================================================
# 10. bio_neurons_run.py — Integrated Neuron Simulation Pipeline
# =============================================================
# 구성:
#   DTGSystem → Mitochondria → HHSoma → MyelinatedAxon
#      → CaVesicle → [PTPPlasticity, SynapticResonance, MetabolicFeedback]
# =============================================================
# Note: numpy, matplotlib, math는 파일 상단에서 이미 import됨
#       colorama는 Optional Color Output 섹션에서 처리됨

from time import perf_counter

# =============================================================
#  Main Integrated Pipeline (patched)
# =============================================================

def run_pipeline(T_ms: float | None = None):
    """
    Integrated Bio-Physical Neuron Simulation
    ----------------------------------------
    Adds missing feedback couplings:
        ① DTG phase → Soma I_ext modulation
        ② HH + Ca ATP consumption → Mito step()
        ③ Feedback(Heat/CO₂/Ca) → Mito efficiency(η)
        ④ Ca alert → transient metabolic boost
    """

    R = CONFIG["RUN"]
    T_ms = int(T_ms if T_ms is not None else R["T_ms"])
    dt_bio = float(R["dt_bio"])
    dt_elec = float(R["dt_elec"])
    # ---------------------------------------------------------
    # 1️⃣ Initialize modules
    # ---------------------------------------------------------
    dtg = DTGSystem(CONFIG["DTG"])
    mito = Mitochondria(CONFIG["MITO"])
    
    # ① IonFlowDynamics 생성 위치를 HHSoma 위로 이동
    ionflow = IonFlowDynamics(CONFIG["AXON"])
    soma = HHSoma(CONFIG["HH"], ionflow=ionflow)
    axon = MyelinatedAxon(CONFIG["AXON"])
    ca = CaVesicle(CONFIG["CA"], dt_ms=CONFIG["CA"]["dt_ms"])
    ptp = PTPPlasticity(PTPConfig())
    resonance = SynapticResonance(omega=1.0, K=0.03, lambda_ca=1.0)
    feedback = MetabolicFeedback(mito)
    
    # HeatGrid는 Mitochondria 내부에서 자동 관리됨

    print("[Neuron Pipeline Quick Run — with Velocity Log]")
    sys.stdout.flush()

    table1_data = []
    table2_data = []
    spike_events = []
    Vmap_data = []

    LOG_INTERVAL = R.get("log_interval", R.get("print_every_ms", 5))
    log_every = max(1, int(round(LOG_INTERVAL / max(dt_bio, 1e-9))))
    total_steps = int(round(T_ms / dt_bio))

    print("=" * 95); sys.stdout.flush()
    print("표 1: 생리학 파라미터"); sys.stdout.flush()
    print("=" * 95); sys.stdout.flush()
    print(f"{'t(ms)':>7} | {'ATP':>6} | {'Vm(mV)':>8} | {'φ(rad)':>7} | "
          f"{'Ca(μM)':>8} | {'PTP R':>7} | {'η(meta)':>7} | {'θ−φ':>7}")
    sys.stdout.flush()
    print("=" * 95); sys.stdout.flush()

    depol_count = 0
    spike_count = 0
    Vm_prev = soma.V
    t0 = perf_counter()

    # =========================================
    # [PATCH 2] ATP 스케일링 파라미터
    # =========================================
    # [PATCH] ATP 의존 Na/K 펌프 효율 조정 (soft sigmoid 함수 사용)
    # 기능: ATP 농도에 따라 펌프 효율을 부드럽게 조정 (급격한 변화 방지)
    # ATP_SOFT_REF: 기준 ATP 농도 (중간 효율 지점)
    # ATP_SOFT_K: 완화 계수 (큰 값일수록 부드러운 전환, 8.0 → 10.0으로 조정)
    #   - 큰 값: 펌프 응답곡선이 완만함 (overshoot 감소)
    #   - 작은 값: 펌프 응답곡선이 급격함 (빠른 반응)
    # MIN_SCALE: 최소 펌프 효율 (ATP가 매우 낮을 때도 일정 효율 유지)
    ATP_SOFT_REF = 80.0   # 기준 ATP (중간 효율 지점)
    ATP_SOFT_K = 10.0     # ✅ [PATCH] 완화 (8.0 → 10.0, 펌프 응답곡선 완화, overshoot 감소)
    MIN_SCALE = 0.2       # 최소 펌프 효율
    

    # ---------------------------------------------------------
    # 2️⃣ Simulation loop
    # ---------------------------------------------------------
    # =============================================================
    # Solver Flow Summary (Numerical Integration Order)
    # -------------------------------------------------------------
    # ① HH/Ion/Axon (Euler-CFL micro integration)
    # ② CaVesicle (Heun or Euler)
    # ③ Feedback(MetabolicFeedback) — Mito η, Ploss, recover_k 조정
    # ④ PTPPlasticity + SynapticResonance (phase learning)
    # ⑤ Mitochondria (ATP, Heat, CO₂ 갱신)
    # ⑥ DTGSystem (Energy–Phase synchronization; Euler or RK4)
    # =============================================================
    # 실제 실행 루프 (V1 구현 기준)
    #   HH/Ion/Axon (micro-steps, CFL)
    #     → Ca²⁺ Vesicle
    #     → Metabolic Feedback(Heat/CO₂/Ca)  # Mito 파라미터 보정
    #     → PTP (on_spike) → Resonance(θ)
    #     → (J_use = NaK + Ca) 집계
    #     → Mito (ATP, Heat, CO₂ 갱신)
    #     → DTG (ATP 기반 φ·E 갱신, θ→φ 역결합)
    # =============================================================
    for t in np.arange(0, T_ms, dt_bio):
        # 🚨 수정 보완점 #1: 매 bio step마다 NaK 소비량을 0으로 초기화
        J_NaK_amount = 0.0

        # =========================================
        # [PATCH 2] ATP 스케일링 (soft sigmoid)
        # =========================================
        sigmoid_arg = (mito.ATP - ATP_SOFT_REF) / ATP_SOFT_K
        sigmoid_val = 1.0 / (1.0 + np.exp(-sigmoid_arg))
        I_pump_scale = MIN_SCALE + (1.0 - MIN_SCALE) * sigmoid_val
        soma.set_I_pump_scale(I_pump_scale)

        # --- (1) 전기/이온 미세 반복: HH ↔ IonFlow ↔ Nernst 고정점 ---
        # (DTG 위상은 아래 (7)에서 계산됨 - 이전 스텝의 ATP 기반)
        MICRO_ITERS = 2          # 2~3 권장 (수렴 확인 후 1로 낮출 수 있음)
        for _micro in range(MICRO_ITERS):
            J_NaK_amount_iter = 0.0

            n_elec = int(round(dt_bio / dt_elec))
            spiked = False
            spk_prev = False

            for k in range(n_elec):
                t_e = t + k * dt_elec

                # (a) 이전 스텝의 이온 농도로 초기 역전위 계산
                # (첫 번째 반복: 이전 스텝의 이온 농도 사용)
                Na_out, K_out = 145.0, 5.0
                Na_in = max(1e-6, 15.0 + (ionflow.ions["Na"]["C"][0] - 15.0))
                K_in  = max(1e-6, 140.0 + (ionflow.ions["K"]["C"][0] - 140.0))
                ENa_dyn = HHSoma.nernst(Na_out, Na_in, z=1)
                EK_dyn  = HHSoma.nernst(K_out,  K_in,  z=1)

                # (b) DTG 위상 구동 → I_ext (phi는 아직 계산되지 않았으므로 이전 값 사용)
                # phi는 (7)에서 계산되므로, 여기서는 이전 스텝의 phi 사용
                # (또는 초기값 0.0)
                phi_current = getattr(dtg, 'phi', 0.0)
                I_ext_mod = 1.0 + 0.5 * np.cos(phi_current)
                I_base = 5.0 * I_ext_mod
                I_back = 0.1 * (axon.V[0] - soma.V)

                # (1) HH 전위 계산
                # [PATCH] Heat 파라미터 추가 (Q10 효과 적용)
                # 기능: 온도에 따라 게이트 반응 속도가 변화하도록 mito.Heat 값을 전달
                # 효과: 온도가 높을수록 게이트 반응 속도가 빨라짐 (생리학적 현실 반영)
                soma_result = soma.step(
                    dt_elec, I_ext=I_base - I_back, ATP=mito.ATP,
                    ENa_override=ENa_dyn, EK_override=EK_dyn,
                    Heat=mito.Heat
                )
                Vm = soma_result["V"]
                J_NaK_rate = soma_result["J_use"]
                J_NaK_amount_iter += J_NaK_rate * dt_elec

                # (2) HH가 갱신한 V로 IonFlow 업데이트
                # [PATCH] HH가 계산한 soma.V를 IonFlow에 반영하여 이온 농도 변화 계산
                # 기능: soma의 전위 변화 → 이온 농도 변화 → 다음 반복에서 더 정확한 Nernst 전위 계산
                ionflow.V[:] = soma.V
                ionflow.step(dt_elec)
                
                # (2.5) IonFlow 결과를 기반으로 Reversal Potentials 업데이트
                # [PATCH] IonFlow 업데이트 후 즉시 reversal potentials 갱신
                # 기능: 이온 농도 변화를 기반으로 ENa, EK, ECa, ECl을 동적으로 재계산
                # 효과: 다음 반복에서 더 정확한 채널 전류 계산 (Nernst 방정식 적용)
                soma.update_reversal_potentials(ionflow)

                # (e) 스파이크 이벤트
                if soma.spiking() and not spk_prev:
                    axon.trigger_alpha(t_e)
                    ca.add_spike(t_e)
                spk_prev = soma.spiking()
                if spk_prev: spiked = True

                # (f) 축삭 전도
                # [PATCH] ATP-dependent Na+ conductance modulation을 위해 ATP 수준 설정
                axon.ATP_level = mito.ATP
                I0 = CONFIG["AXON"]["stim_gain"] * (soma.V - axon.V[0])
                axon.step(dt_elec, t_ms=t_e, I0_from_soma=I0, soma_V=soma.V)

            # 미세 반복 누적 소비율을 평균화해 안정화
            if _micro == 0:
                J_NaK_amount = J_NaK_amount_iter
            else:
                J_NaK_amount = 0.5 * (J_NaK_amount + J_NaK_amount_iter)

        if -20 < soma.V < 40 and Vm_prev < -20:
            depol_count += 1
        if spiked:
            spike_count += 1
            
        Vm_prev = soma.V

        # --- (3) Ca · PTP · Feedback ---
        # P2 (Ca-ATP 소비 회계)는 ca.step의 J_Ca_rate 반환으로 해결됨
        # [PATCH] SOLVER 설정에 따라 적분 방법 선택
        # - "heun": Heun 방법 사용 (더 정확하지만 계산 비용 증가)
        # - 그 외: 기존 ca.step() 사용 (기본 Euler 방법)
        if CONFIG["SOLVER"]["CA"] == "heun":
            # Heun 방법 사용 (predictor-corrector)
            # predictor: Euler step으로 예측
            Ca0 = ca.Ca
            influx0 = sum(ca.A * ca._alpha_kernel(ca.t_ms + ca.dt_ms - ts) for ts in ca.spike_times)
            pump0 = ca.k_c * float(mito.ATP) * max(0.0, (Ca0 - ca.C0))
            dCa0 = (influx0 - pump0)
            Ca_pred = Ca0 + dCa0 * (ca.dt_ms / 1000.0)
            
            # corrector: 예측값을 사용해서 기울기 재계산 후 평균
            influx1 = influx0  # alpha kernel은 t+dt에서 다시 계산하면 더 정확(원하면 갱신)
            pump1 = ca.k_c * float(mito.ATP) * max(0.0, (Ca_pred - ca.C0))
            dCa1 = (influx1 - pump1)
            ca.Ca = Ca0 + 0.5*(dCa0 + dCa1) * (ca.dt_ms / 1000.0)
            
            # 시간 진행
            ca.t_ms += ca.dt_ms
            
            # 스파이크 메모리 관리
            ca._trim_spike_memory()
            
            # 안전: 지나친 음수 방지
            ca.Ca = max(ca.Ca, ca.C0 * 0.1)
            
            # 이후 S/status/J_Ca_rate 계산은 기존 로직 재사용
            # 정규화 및 상태
            denom = max(1e-12, (ca.Cmax - ca.C0))
            S = (ca.Ca - ca.C0) / denom
            status = "under" if S < 0.0 else ("normal" if S <= 1.0 else "alert")
            
            # 이벤트 기록 (메모리 과다 방지: 필요 시 슬라이싱)
            ca_ev = VesicleEvent(t_ms=float(ca.t_ms), Ca=float(ca.Ca), S=float(S), status=status)
            ca.events.append(ca_ev)
            if len(ca.events) > 10000:
                ca.events = ca.events[-5000:]
            
            # J_Ca_rate 계산 (ATP 소비율)
            # [NOTE] k_atp_per_Ca가 있는 경우 사용, 없으면 기본값 1.0
            k_atp_per_Ca = getattr(ca, 'k_atp_per_Ca', 1.0)
            J_Ca_rate = k_atp_per_Ca * ca.k_c * float(mito.ATP) * max(0.0, (ca.Ca - ca.C0))
        else:
            # 기본 Euler 방법 사용 (ca.step() 내부 구현)
            ca_ev, J_Ca_rate = ca.step(ATP=mito.ATP)  # 🔸 변경: J_Ca_rate 함께 받음 [ATP/ms]
        
        # --- (3) Feedback 먼저 ---
        # [PATCH] Feedback을 Mito step 전에 실행하여 Mito 파라미터를 조정
        # 기능: Ca 상태에 따라 Mito의 eta0, Ploss, recover_k 등을 동적으로 조정
        # 효과: Mito step이 조정된 파라미터를 사용하여 ATP, Heat, CO2를 계산
        feedback.update(ca_ev.status)
        
        # Ca 스텝 이후에 PTP와 Resonance 업데이트
        if spiked:
            ptp.on_spike(S=ca_ev.S)
            phi_current = getattr(dtg, 'phi', 0.0)
            resonance.on_spike(ptp.R, phi_current)
            spike_events.append((t, ca_ev.Ca * 1e6, ptp.R))
        ptp.step(dt_bio)
        
        # --- (4) 위상 공명 한 스텝 ---
        # phi는 아직 계산되지 않았으므로 이전 스텝의 phi 사용
        phi_current = getattr(dtg, 'phi', 0.0)
        theta, delta_phi = resonance.step(dt_bio, phi_current, ca_ev.S)
        
        # --- (4.5) DTG에 θ 역피드백 주입 (양방향 결합 완성) ---
        dtg.apply_resonance_feedback(theta, k_back=0.05)

        # --- (5) 이번 bio 스텝 총 소비율 ---
        # [PATCH] Energy leak integral for metabolic accounting
        # 축삭 전위에서 Vrest로부터의 편차를 적분하여 누출 에너지 비용 계산
        # 누출 에너지 = Σ(V - Vrest)² * dx (공간 적분)
        leak_cost = np.sum((axon.V - CONFIG["AXON"]["Vrest"])**2) * axon.dx
        # 총 ATP 소비율 = Na/K 펌프 + Ca 펌프 + 누출 에너지 비용 (0.001 스케일)
        J_use_total = (J_NaK_amount / dt_bio) + J_Ca_rate + 0.001 * leak_cost  # [ATP/ms]

        # --- (6) Mito step ---
        # [PATCH] 섹션 번호 중복 해결: (4) → (6)으로 변경
        # HeatGrid는 Mitochondria 내부에서 자동 관리됨
        # feedback.update()는 (3)에서 이미 호출됨 (ca_ev.status 사용)
        # NOTE: dt_bio ≫ dt_elec 이므로, Mito는 생리학적 시간 상수 기반의
        #       느린(저주파) 통합 계층으로 유지된다. (ATP 갱신은 dt_bio 단위)
        # [PATCH] Mito energy step with full leak correction (누출 에너지 포함)
        out = mito.step(dt_bio, Glu=5.0, O2=5.0, J_use=J_use_total)
        
        # --- (7) DTG step — "이 스텝에서 방금 생산된 ATP" 사용 ---
        # [PATCH] 섹션 번호 중복 해결: (5) → (7)으로 변경
        # [PATCH] Mito step의 반환값에서 ATP를 사용하여 DTG에 전달
        # 기능: 이번 스텝에서 방금 계산된 최신 ATP 값을 DTG에 전달
        # 효과: mito.ATP (객체 속성, 이전 값일 수 있음) 대신 out["ATP"] (이번 스텝의 최신 값) 사용
        # 시간적 일관성: Mito 업데이트 → DTG 업데이트 순서 보장
        # [PATCH] SOLVER 설정에 따라 적분 방법 선택
        # - "rk4": rk4_step 사용 (더 정확하지만 계산 비용 증가)
        # - "euler": 기존 dtg.step() 사용 (기본 Euler 방법)
        if CONFIG["SOLVER"]["DTG"] == "rk4":
            # 4차 Runge-Kutta 방법 사용
            y = np.array([dtg.E, dtg.phi])
            y = rk4_step(dtg_rhs(dtg, out["ATP"]), y, dt_bio)
            dtg.E, dtg.phi = float(np.clip(y[0], 0.0, dtg.E0*2.0)), float(y[1]%(2*np.pi))
            phi = dtg.phi
        else:
            # 기본 Euler 방법 사용 (dtg.step() 내부 구현)
            _, phi, _, _ = dtg.step(out["ATP"], dt_bio)

        # =========================================
        # [PATCH 2] HeatGrid 연동/확산 → feedback.update() 순으로 유지
        # =========================================
        # feedback.update()는 (3)에서 이미 호출됨 (ca_ev.status 사용)
        
        # --- (8) 로깅 ---
        step_idx = int(round(t / dt_bio))
        if step_idx % log_every == 0:
            Ca_um = ca_ev.Ca * 1e6
            phi_display = math.fmod(phi, 2 * math.pi)
            delta_phi_logged = delta_phi if np.isfinite(delta_phi) else 0.0
            table1_data.append(
                (
                    float(t),
                    float(mito.ATP),
                    float(soma.V),
                    float(phi_display),
                    float(Ca_um),
                    float(ptp.R),
                    float(mito.eta),
                    float(delta_phi_logged),
                )
            )

            tailV_curr = float(axon.V[-1])
            active_nodes = int(np.sum(axon.V >= axon.thresh))
            v_snapshot = float(axon.velocity_last())
            table2_data.append(
                (
                    float(t),
                    v_snapshot,
                    tailV_curr,
                    float(mito.Heat),
                    float(mito.CO2),
                    int(spike_count),
                    active_nodes,
                    bool(tailV_curr > axon.thresh),
                )
            )
            Vmap_data.append(axon.V.copy())

    t1 = perf_counter()

    for t_ms, ATP_val, Vm_val, phi_val, Ca_val, R_val, eta_val, delta_phi_val in table1_data:
        print(f"{t_ms:7.1f} | {ATP_val:6.2f} | {Vm_val:8.2f} | {phi_val:7.3f} | "
              f"{Ca_val:8.3f} | {R_val:7.3f} | {eta_val:7.3f} | {delta_phi_val:7.3f}")
        sys.stdout.flush()

    print("=" * 75); sys.stdout.flush()
    if spike_events:
        print("Spikes Timeline"); sys.stdout.flush()
        print("=" * 75); sys.stdout.flush()
        for t_event, ca_event, r_event in spike_events:
            print(f"[{t_event:7.2f} ms] Spike → Ca={ca_event:.2f} μM, PTP R={r_event:.3f}")
            sys.stdout.flush()
        print("=" * 75); sys.stdout.flush()
    print("표 2: 전도 및 환경 파라미터"); sys.stdout.flush()
    print("=" * 75); sys.stdout.flush()
    print(
        f"{'t(ms)':>7} | {'v(m/s)':>7} | {'tailV':>8} | {'Heat':>6} | "
        f"{'CO₂':>6} | {'spikes':>7} | {'active':>7} | {'tail_peak':>9}"
    )
    sys.stdout.flush()
    print("=" * 75); sys.stdout.flush()

    for (
        t_ms,
        v_val,
        tailV_val,
        heat_val,
        co2_val,
        spike_total,
        active_nodes,
        tail_peak,
    ) in table2_data:
        print(
            f"{t_ms:7.1f} | {v_val:7.2f} | {tailV_val:8.2f} | {heat_val:6.2f} | "
            f"{co2_val:6.2f} | {spike_total:7d} | {active_nodes:7d} | {str(tail_peak):>9}"
        )
        sys.stdout.flush()

    print("=" * 75); sys.stdout.flush()

    first_cross_raw = [t_val for t_val in getattr(axon, "first_cross_ms", {}).values() if t_val is not None]
    if first_cross_raw:
        first_cross_raw.sort()
        t0_cross = first_cross_raw[0]
        tN_cross = first_cross_raw[-1]
        TOF_scaled = max(tN_cross - t0_cross, 1e-3)
    else:
        t0_cross = float("nan")
        tN_cross = float("nan")
        TOF_scaled = float("nan")

    axon_length_sim = axon.N * axon.dx
    axon_length_real = axon.N * getattr(axon, "dx_real_m", axon.dx)
    ms_per_sim_ms = R.get("ms_per_sim_ms", 1.0)
    TOF_real_ms = TOF_scaled * ms_per_sim_ms if np.isfinite(TOF_scaled) else float("nan")
    v_scaled = axon_length_sim / (TOF_scaled / 1000.0) if np.isfinite(TOF_scaled) and TOF_scaled > 0 else float("nan")
    v_real = axon_length_real / (TOF_real_ms / 1000.0) if np.isfinite(TOF_real_ms) and TOF_real_ms > 0 else float("nan")

    print("[Transmission Velocity Summary — Scaled vs Real]"); sys.stdout.flush()
    print(f"TOF (ms)              : {TOF_scaled:.2f}"); sys.stdout.flush()
    print(f"TOF_real (ms)         : {TOF_real_ms:.2f}"); sys.stdout.flush()
    print(f"Axon length (sim)     : {axon_length_sim:.6f}"); sys.stdout.flush()
    print(f"Axon length real (m)  : {axon_length_real:.6f}"); sys.stdout.flush()
    print(f"v_scaled (sim units)  : {v_scaled:.2f} m/s"); sys.stdout.flush()
    print(f"v_real   (physical)   : {v_real:.2f} m/s"); sys.stdout.flush()
    print(f"Done. Elapsed {(t1 - t0):.3f} sec"); sys.stdout.flush()

    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    df1 = pd.DataFrame(
        table1_data,
        columns=["t", "ATP", "Vm", "phi", "Ca", "R", "eta", "delta_phi"],
    )
    df2 = pd.DataFrame(
        table2_data,
        columns=["t", "v", "tailV", "Heat", "CO2", "spikes", "active", "tail_peak"],
    )
    df1.to_csv(os.path.join(logs_dir, "table1.csv"), index=False)
    df2.to_csv(os.path.join(logs_dir, "table2.csv"), index=False)
    print("CSV files saved: logs/table1.csv, logs/table2.csv"); sys.stdout.flush()

    if Vmap_data:
        Vmap = np.array(Vmap_data).T
        out_png = os.path.join(logs_dir, "saltatory_conduction.png")
        plt.figure(figsize=(8, 4))
        plt.imshow(Vmap, aspect='auto', cmap='plasma', origin='lower',
                   extent=[0, T_ms, 0, axon.N])
        plt.colorbar(label='Node transient (mV)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Node index (prox→distal)')
        plt.title('Saltatory Conduction — Detailed control panel')
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Visualization saved: {out_png}")
        sys.stdout.flush()
    
# =============================================================
# Entry Point
# =============================================================
if __name__ == "__main__":
    run_pipeline()

# =============================================================
# PHAM-OPEN LICENSE v2.0 (Trust-Based Creative Ledger License)
# (C) 2025 Qquarts Co / GNJz
#
# 🪶 1. 기본 원칙 (Principles)
# 한국어 버전
# 이 라이선스는 법적 강제가 아닌, 신뢰·기록·기여를 바탕으로 한 새로운 오픈 코드 문화의 선언입니다.
# 모든 코드는 인간의 창의적 기여이며, 그 가치는 공개된 Ledger를 통해 투명하게 증명됩니다.
#
# English Version
# This license is a declaration of a new open-code culture founded on trust, record-keeping, and contribution,
# rather than legal compulsion. All code represents human creative contribution,
# and its value is transparently proven through a public Ledger.
#
# ... (이하 전체 PHAM-OPEN LICENSE v2.0 본문)
#
# “Trust as Law. Ledger as Proof. Code as Culture.”
# =============================================================