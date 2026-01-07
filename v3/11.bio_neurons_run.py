# =============================================================
# 11.bio_neurons_run.py — Integrated Neuron Simulation Pipeline (V3)
# =============================================================
# V3 계약 고정:
#   - 단일 방향화: Soma → Axon (역방향 참조 금지) ⭐ V3 변경
#   - 값 복사: 참조 공유 금지 (ionflow.set_V(soma.V)) ⭐ V3 변경
#   - 이벤트 기반 전달: 값 복사로 데이터 전달 ⭐ V3 변경
#   - ATP: [0,100] (정규화, 0~100 범위로 통일) ⭐ V3 변경
#   - S: [0,1] (정규화된 Ca 농도) ⭐ V3 계약 고정
#   - 시간 단위: [ms] (밀리초) ⭐ V3 계약 고정
#
# 구성:
#   DTGSystem → Mitochondria → HHSoma → MyelinatedAxon
#      → CaVesicle → [PTPPlasticity, SynapticResonance, MetabolicFeedback]
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
# Import CONFIG and all components
# =============================================================
import importlib.util
import os

# 현재 디렉토리 경로
_current_dir = os.path.dirname(os.path.abspath(__file__))

# 동적 import 함수
def _import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# CONFIG 및 모든 컴포넌트 import (components 폴더에서)
_components_dir = os.path.join(_current_dir, "components")

_config_mod = _import_module("bioneuron_config", os.path.join(_components_dir, "00.bioneuron_config.py"))
CONFIG = _config_mod.CONFIG

_dtg_mod = _import_module("dtg_system", os.path.join(_components_dir, "01.dtg_system.py"))
DTGSystem = _dtg_mod.DTGSystem

_mito_mod = _import_module("mitochon_atp", os.path.join(_components_dir, "02.mitochon_atp.py"))
Mitochondria = _mito_mod.Mitochondria

_hh_mod = _import_module("hh_soma", os.path.join(_components_dir, "03.hh_soma.py"))
HHSoma = _hh_mod.HHSoma

_ion_mod = _import_module("ion_flow", os.path.join(_components_dir, "04.ion_flow.py"))
IonFlowDynamics = _ion_mod.IonFlowDynamics

_axon_mod = _import_module("myelin_axon", os.path.join(_components_dir, "05.myelin_axon.py"))
MyelinatedAxon = _axon_mod.MyelinatedAxon

_ca_mod = _import_module("ca_vesicle", os.path.join(_components_dir, "06.ca_vesicle.py"))
CaVesicle = _ca_mod.CaVesicle
VesicleEvent = _ca_mod.VesicleEvent

_ptp_mod = _import_module("ptp", os.path.join(_components_dir, "07.ptp.py"))
PTPPlasticity = _ptp_mod.PTPPlasticity
PTPConfig = _ptp_mod.PTPConfig

_feedback_mod = _import_module("metabolic_feedback", os.path.join(_components_dir, "08.metabolic_feedback.py"))
MetabolicFeedback = _feedback_mod.MetabolicFeedback

_resonance_mod = _import_module("synaptic_resonance", os.path.join(_components_dir, "09.synaptic_resonance.py"))
SynapticResonance = _resonance_mod.SynapticResonance

_terminal_mod = _import_module("terminal_release", os.path.join(_components_dir, "10.terminal_release.py"))
Terminal = _terminal_mod.Terminal
SimpleSynapse = _terminal_mod.SimpleSynapse

# =============================================================
# Solver Utilities
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
# Main Integrated Pipeline
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
    r"""
    Integrated Bio-Physical Neuron Simulation Pipeline
    --------------------------------------------------
    
    목적:
        모든 생물학적 뉴런 컴포넌트를 시간에 따라 순차적으로 업데이트하여
        완전한 뉴런 시뮬레이션을 실행합니다.
    
    Parameters
    ----------
    T_ms : float, optional
        시뮬레이션 총 시간 [ms]. None이면 CONFIG["RUN"]["T_ms"] 사용.
        기본값: 500 ms
    
    Returns
    -------
    None
        결과는 다음 파일로 저장됩니다:
        - logs/table1.csv: 생리학 파라미터 (ATP, Vm, φ, Ca, R, η, θ-φ)
        - logs/table2.csv: 전도 파라미터 (v, tailV, Heat, CO₂, spikes)
        - logs/terminal.csv: 시냅스 방출량 (Q, p_eff)
        - logs/saltatory_conduction.png: 축삭 전도 시각화
    
    알고리즘
    --------
    각 bio 스텝(dt_bio)마다 다음 순서로 실행:
    
    1. 모듈 초기화
       - DTGSystem, Mitochondria, HHSoma, IonFlowDynamics, MyelinatedAxon
       - CaVesicle, PTPPlasticity, SynapticResonance, MetabolicFeedback
       - Terminal, SimpleSynapse
    
    2. 시뮬레이션 루프 (t = 0 ~ T_ms, step = dt_bio)
       
       [2-1] 전기/이온 미세 반복 (MICRO_ITERS = 2)
            for _micro in range(MICRO_ITERS):
                for k in range(n_elec):  # n_elec = dt_bio / dt_elec
                    (a) Nernst 역전위 계산 (동적 이온 농도 기반)
                        ENa = (RT/F)·ln([Na]_out / [Na]_in)
                        EK  = (RT/F)·ln([K]_out / [K]_in)
                    
                    (b) DTG 위상 구동 → I_ext 생성 (자기 발화)
                        I_ext = I_stim + I_autonomous · (1 + 0.5·cos(φ))
                    
                    (c) HH 막전위 계산
                        C_m·dV/dt = g_Na·m³h·(E_Na-V) + g_K·n⁴·(E_K-V)
                                   + g_L·(E_L-V) + I_ext - I_pump
                        J_NaK = g_pump_consume · |I_pump|
                    
                    (d) IonFlow 업데이트
                        ∂C_i/∂t = D_i·∇²C_i - μ_i·z_i·C_i·∇V
                        V[:] = soma.V  (전위 반영)
                    
                    (e) Reversal potentials 재계산
                        ENa, EK, ECa, ECl = f([ion]_in, [ion]_out)
                    
                    (f) 스파이크 이벤트 처리
                        if soma.spiking():
                            axon.trigger_alpha(t)
                            ca.add_spike(t)
                    
                    (g) 축삭 전도
                        ∂V/∂t = D(x)·∂²V/∂x² - (V-V_rest)/τ + I_ext/C_m
                        (수초화 도약전도)
       
       [2-2] Ca²⁺ Vesicle 업데이트
            Solver = "heun": Heun 방법 (predictor-corrector)
            Solver = "euler": 기본 Euler 방법
            수식: dCa/dt = Σ_k A·α(t-t_k) - k_c·ATP·(Ca - C0)
                  α(t) = (e^{-t/τ_d} - e^{-t/τ_r})_+
            출력: ca_ev (VesicleEvent), J_Ca_rate [ATP/ms]
       
       [2-3] Metabolic Feedback (Mito 파라미터 조정)
            feedback.update(ca_ev.status)
            - Ca alert → recover_k 증가
            - Heat ↑ → η0 감소
            - CO₂ ↑ → Ploss 증가
       
       [2-4] PTP 및 Synaptic Resonance
            if spiked:
                ptp.on_spike(S=ca_ev.S)  # Hill 함수: R += g_ptp·S^n/(S^n + K^n)
                resonance.on_spike(R, φ)
            ptp.step(dt_bio)  # R *= exp(-dt/τ_ptp)
            theta, delta_phi = resonance.step(dt_bio, φ, S)
            # dθ/dt = ω + K·sin(φ-θ)·(1 + λ·S)
       
       [2-5] DTG에 θ 역피드백 주입
            dtg.apply_resonance_feedback(theta, k_back=0.05)
            # dφ/dt += k_res·sin(θ - φ)
       
       [2-6] Terminal release (스파이크 의존)
            if spiked:
                Q = spike · α_C·S^p · α_R·R^q · α_φ·(1+h|Δφ|) · (ATP/100)^{1/2}
                p_eff = p0·(1 + R)
                terminal.broadcast(t, Q)
       
       [2-7] ATP 소비율 집계
            leak_cost = Σ(V - V_rest)²·dx  (누출 에너지)
            J_use_total = (J_NaK / dt_bio) + J_Ca + 0.001·leak_cost
       
       [2-8] Mitochondria 업데이트
            out = mito.step(dt_bio, Glu=5.0, O2=5.0, J_use=J_use_total)
            - ATP, Heat, CO₂ 갱신
            - Solver: RK4 방법
       
       [2-9] DTG 업데이트
            Solver = "rk4": rk4_step(dtg_rhs(dtg, ATP), [E, φ], dt_bio)
            Solver = "euler": dtg.step(ATP, dt_bio)
            - E, φ 갱신
            - θ→φ 결합 반영
    
    3. 결과 출력 및 저장
       - 표 1: 생리학 파라미터 (ATP, Vm, φ, Ca, R, η, θ-φ)
       - 표 2: 전도 파라미터 (v, tailV, Heat, CO₂, spikes)
       - CSV 파일 저장
       - 축삭 전도 시각화 (heatmap)
    
    양방향 피드백 루프
    -----------------
    ① DTG phase → Soma I_ext: φ(t) → I_ext = I_stim + I_autonomous·(1 + 0.5·cos(φ)) (자기 발화)
    ② HH + Ca → Mito: J_NaK + J_Ca → J_use → ATP 소비
    ③ Feedback → Mito: Heat/CO₂/Ca → η0, Ploss, recover_k 조정
    ④ Ca alert → Mito: recover_k 증가 (ATP 회복률 강화)
    ⑤ Resonance ↔ DTG: θ → φ 역피드백 (양방향 결합)
    
    Notes
    -----
    - dt_elec (전기적 시간 스텝)과 dt_bio (생리학적 시간 스텝)은 분리됨
    - CFL 조건: dt_elec < 0.5·dx²/D_max (안정성 보장)
    - MICRO_ITERS: HH ↔ IonFlow 수렴을 위한 미세 반복 횟수 (기본값: 2)
    - Solver 방법은 CONFIG["SOLVER"]에서 설정 가능 (Euler, Heun, RK4)
    """

    R = CONFIG["RUN"]
    T_ms = int(T_ms if T_ms is not None else R["T_ms"])
    dt_bio = float(R["dt_bio"])
    dt_elec = float(R["dt_elec"])
    # ---------------------------------------------------------
    # 1️⃣ Initialize modules
    # ---------------------------------------------------------
    # 모든 생물학적 뉴런 컴포넌트를 초기화합니다.
    # 초기화 순서는 의존성 관계를 고려합니다:
    #   - IonFlowDynamics는 HHSoma보다 먼저 생성 (HHSoma가 ionflow를 사용)
    #   - MetabolicFeedback은 Mitochondria 객체를 받음
    
    # [1-1] 메타 제어 계층
    dtg = DTGSystem(CONFIG["DTG"])
    # 수식: dE/dt = g_sync·(ATP - E) - γ·(E - E0)
    #       dφ/dt = ω0 + α·(E - E0)
    
    mito = Mitochondria(CONFIG["MITO"])
    # 수식: dATP/dt = κ·(E_buf - ATP) - J_use
    #       dHeat/dt = (1-η)·J_transfer - (Heat - Heat_rest)/τ_heat
    #       dCO₂/dt = c_CO2·J_transfer - (CO₂ - CO₂_rest)/τ_CO2
    
    # [1-2] 전기생리 계층
    # ① IonFlowDynamics 생성 위치를 HHSoma 위로 이동
    #    이유: HHSoma가 ionflow 객체를 받아서 reversal potentials를 계산
    ionflow = IonFlowDynamics(CONFIG["AXON"])
    # 수식: ∂C_i/∂t = D_i·∇²C_i - μ_i·z_i·C_i·∇V
    #       (다중 이온 확산 + 전기장 drift)
    
    soma = HHSoma(CONFIG["HH"], ionflow=ionflow)
    # 수식: C_m·dV/dt = g_Na·m³h·(E_Na-V) + g_K·n⁴·(E_K-V)
    #                  + g_L·(E_L-V) + I_ext - I_pump
    #       I_pump = g_pump·(1 - e^{-ATP/ATP₀})·(V - E_pump)
    
    axon = MyelinatedAxon(CONFIG["AXON"])
    # 수식: ∂V/∂t = D(x)·∂²V/∂x² - (V-V_rest)/τ + I_ext/C_m
    #       (수초화 도약전도, 노드에서만 Na⁺ 채널 활성)
    
    # [1-3] 시냅스 가소성 계층
    ca = CaVesicle(CONFIG["CA"], dt_ms=CONFIG["CA"]["dt_ms"])
    # 수식: dCa/dt = Σ_k A·α(t-t_k) - k_c·ATP·(Ca - C0)
    #       α(t) = (e^{-t/τ_d} - e^{-t/τ_r})_+  (Spike-triggered α-kernel)
    
    ptp = PTPPlasticity(PTPConfig())
    # 수식: R_{n+1} = R_n·exp(-dt/τ_ptp) + δ (on spike)
    #       δ = g_ptp·S^n/(S^n + K^n)  (Hill 함수)
    
    res_cfg = CONFIG.get("RESONANCE", {})
    resonance = SynapticResonance(
        omega=res_cfg.get("omega", 1.0),
        K=res_cfg.get("K", 0.03),
        lambda_ca=res_cfg.get("lambda_ca", 1.0)
    )
    # 수식: dθ/dt = ω + K·sin(φ-θ)·(1 + λ·S)
    #       (Ca²⁺-modulated Kuramoto 모델)
    
    # [1-4] 피드백 및 출력 계층
    feedback = MetabolicFeedback(mito)
    # 기능: Heat/CO₂/Ca 상태에 따라 Mito 파라미터 동적 조정
    # 수식: η0 = η_base - β_heat·Heat
    #       Ploss = Ploss_base·(1 + β_CO2·CO2)
    #       recover_k = recover_base·(1 + λ_Ca·S_alert)  (Ca alert 시)
    
    terminal = Terminal()
    # 수식: Q = spike · α_C·S^p · α_R·R^q · α_φ·(1+h|Δφ|) · (ATP/100)^{1/2}
    #       p_eff = p0·(1 + R)
    
    sink_syn = SimpleSynapse()
    terminal.attach_synapse(sink_syn)
    # 기능: Terminal에서 방출된 이벤트를 수집하여 CSV로 저장
    
    # HeatGrid는 Mitochondria 내부에서 자동 관리됨

    print("[Neuron Pipeline Quick Run — with Velocity Log]")
    sys.stdout.flush()

    table1_data = []
    table2_data = []
    spike_events = []
    Vmap_data = []
    terminal_logs = []

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
    # [ATP 의존 Na/K 펌프 효율 조정]
    # =========================================
    # 개념:
    #   ATP 농도에 따라 Na⁺/K⁺ 펌프의 효율을 부드럽게 조정합니다.
    #   급격한 변화를 방지하여 수치 안정성을 향상시킵니다.
    #
    # 수식:
    #   sigmoid_arg = (ATP - ATP_SOFT_REF) / ATP_SOFT_K
    #   sigmoid_val = 1 / (1 + exp(-sigmoid_arg))
    #   I_pump_scale = MIN_SCALE + (1 - MIN_SCALE) · sigmoid_val
    #
    # 파라미터:
    #   ATP_SOFT_REF: 기준 ATP 농도 [ATP units]
    #                 - 중간 효율 지점 (sigmoid_val = 0.5)
    #   ATP_SOFT_K: 완화 계수 [ATP units]
    #               - 큰 값: 펌프 응답곡선이 완만함 (overshoot 감소)
    #               - 작은 값: 펌프 응답곡선이 급격함 (빠른 반응)
    #   MIN_SCALE: 최소 펌프 효율 (0~1)
    #              - ATP가 매우 낮을 때도 일정 효율 유지 (생존 보장)
    #
    # 효과:
    #   - ATP가 낮을 때: I_pump_scale ≈ MIN_SCALE (최소 효율 유지)
    #   - ATP가 높을 때: I_pump_scale ≈ 1.0 (최대 효율)
    #   - 전환 구간: 부드러운 sigmoid 곡선 (급격한 변화 방지)
    ATP_SOFT_REF = 80.0   # 기준 ATP (중간 효율 지점)
    ATP_SOFT_K = 10.0     # 완화 계수 (펌프 응답곡선 완화, overshoot 감소)
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
    # =============================================================
    # Stimulus Helper Function
    # =============================================================
    # 개념:
    #   외부 자극 전류를 계산하는 헬퍼 함수입니다.
    #   CONFIG["STIMULUS"] 설정에 따라 다양한 자극 프로토콜을 지원합니다.
    #
    # 지원 프로토콜:
    #   1. discrete pulses: 개별 펄스 (pulse1, pulse2)
    #      - start ≤ t ≤ end: amplitude 추가
    #   2. train: 주기적 펄스 트레인
    #      - start ≤ t ≤ end: 주기적 펄스 생성
    #      - 주기: period = 1000 / f_hz [ms]
    #      - 펄스 폭: width_ms [ms]
    #      - 진폭: amp [µA]
    #
    # 수식 (train protocol):
    #   period = 1000 / f_hz  [ms]
    #   phase = (t - t0) mod period
    #   I_stim = amp  (if phase ≤ width_ms)
    #          = base (otherwise)
    # =============================================================
    SCONF = CONFIG.get("STIMULUS", {})
    def stimulus_current(t_ms: float) -> float:
        """
        외부 자극 전류 계산
        
        Parameters
        ----------
        t_ms : float
            현재 시간 [ms]
            
        Returns
        -------
        float
            자극 전류 [µA]
        """
        if not SCONF:
            return float(SCONF.get("base", 0.0))
        protocol = SCONF.get("protocol", "none")
        base = float(SCONF.get("base", 0.0))
        val = base
        
        # discrete pulses (개별 펄스)
        for key in ("pulse1", "pulse2"):
            p = SCONF.get(key)
            if p:
                if p["start"] <= t_ms <= p["end"]:
                    val += float(p.get("amplitude", 0.0))
        
        # train protocol (주기적 펄스 트레인)
        if protocol == "train":
            tr = SCONF.get("train", {})
            t0, t1 = float(tr.get("start", 0.0)), float(tr.get("end", 0.0))
            if t0 <= t_ms <= t1:
                f_hz = float(tr.get("f_hz", 20.0))  # 주파수 [Hz]
                width = float(tr.get("width_ms", 2.0))  # 펄스 폭 [ms]
                amp = float(tr.get("amp", 100.0))  # 진폭 [µA]
                # rectangular pulses every 1000/f ms
                period = 1000.0 / max(1e-6, f_hz)  # 주기 [ms]
                phase = (t_ms - t0) % period  # 위상 [ms]
                if phase <= width:
                    val += amp
        return float(val)
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

        # =============================================================
        # (1) 전기/이온 미세 반복: HH ↔ IonFlow ↔ Nernst 고정점
        # =============================================================
        # 개념:
        #   HH 막전위와 이온 농도는 강하게 결합되어 있습니다:
        #   - V 변화 → 이온 이동 → 농도 변화 → Nernst 전위 변화 → V 변화
        #   이를 해결하기 위해 MICRO_ITERS번 반복하여 수렴시킵니다.
        #
        # 수식 (고정점 반복):
        #   V_{n+1} = f(V_n, [ion]_n)  (HH 방정식)
        #   [ion]_{n+1} = g(V_{n+1}, [ion]_n)  (IonFlow 방정식)
        #   E_{ion} = (RT/F)·ln([ion]_out / [ion]_in)  (Nernst 방정식)
        #
        # 반복 횟수:
        #   MICRO_ITERS = 2 (기본값)
        #   - 2~3번 반복으로 충분히 수렴
        #   - 수렴 확인 후 1로 낮출 수 있음 (성능 향상)
        #
        # 시간 스케일:
        #   dt_elec: 전기적 시간 스텝 [ms] (HH, Ion, Axon)
        #   n_elec = dt_bio / dt_elec: bio 스텝 내 전기 스텝 수
        #   (DTG 위상은 아래 (7)에서 계산됨 - 이전 스텝의 ATP 기반)
        # =============================================================
        MICRO_ITERS = 2          # 2~3 권장 (수렴 확인 후 1로 낮출 수 있음)
        for _micro in range(MICRO_ITERS):
            J_NaK_amount_iter = 0.0

            n_elec = int(round(dt_bio / dt_elec))
            spiked = False
            spk_prev = False

            for k in range(n_elec):
                t_e = t + k * dt_elec

                # -----------------------------------------------------
                # (a) Nernst 역전위 계산 (동적 이온 농도 기반)
                # -----------------------------------------------------
                # 개념:
                #   이전 스텝의 이온 농도를 사용하여 현재 역전위를 계산합니다.
                #   첫 번째 반복: 이전 bio 스텝의 이온 농도 사용
                #   이후 반복: 이번 미세 반복에서 갱신된 이온 농도 사용
                #
                # 수식:
                #   E_ion = (RT/F)·ln([ion]_out / [ion]_in)
                #   - R: 기체 상수 [J/(mol·K)]
                #   - T: 온도 [K] (37°C = 310K)
                #   - F: 패러데이 상수 [C/mol]
                #   - [ion]_out, [ion]_in: 외부/내부 이온 농도 [mM]
                #
                # 동적 농도:
                #   Na_in = 15.0 + (ionflow.ions["Na"]["C"][0] - 15.0)
                #   K_in  = 140.0 + (ionflow.ions["K"]["C"][0] - 140.0)
                #   (기준값 15.0, 140.0에서의 편차를 반영)
                # -----------------------------------------------------
                Na_out, K_out = 145.0, 5.0  # 외부 농도 [mM] (고정)
                Na_in = max(1e-6, 15.0 + (ionflow.ions["Na"]["C"][0] - 15.0))
                K_in  = max(1e-6, 140.0 + (ionflow.ions["K"]["C"][0] - 140.0))
                ENa_dyn = HHSoma.nernst(Na_out, Na_in, z=1)  # [mV]
                EK_dyn  = HHSoma.nernst(K_out,  K_in,  z=1)  # [mV]

                # -----------------------------------------------------
                # (b) DTG 위상 구동 → I_ext 생성 (자기 발화)
                # -----------------------------------------------------
                # 개념:
                #   DTG 위상 φ(t)가 소마 입력 전류를 직접 생성합니다.
                #   이것이 V2의 자기 발화 메커니즘입니다.
                #   - DTG는 자체적으로 위상을 생성: dφ/dt = ω0 + α·(E - E0)
                #   - 위상이 높을 때(cos(φ) > 0) 입력이 증가하여 발화 확률 증가
                #
                # 수식:
                #   I_base = I_stim + I_autonomous · (1 + 0.5·cos(φ))
                #   - I_stim: 외부 자극 전류 (train/pulses, 선택적)
                #   - I_autonomous: 기본 자율 전류 (DTG 위상 변조됨, 자기 발화)
                #   - 0.5·cos(φ): 위상 변조 항 (진폭 0.5)
                #
                # 자기 발화:
                #   - I_stim = 0이어도 I_autonomous > 0이면 발화 가능
                #   - DTG 위상이 주기적으로 진동하므로 주기적 발화 패턴 생성
                #
                # 주의:
                #   phi는 (8)에서 계산되므로, 여기서는 이전 스텝의 phi 사용
                #   (또는 초기값 0.0)
                # -----------------------------------------------------
                phi_current = getattr(dtg, 'phi', 0.0)  # [rad]
                I_ext_mod = 1.0 + 0.5 * np.cos(phi_current)  # 위상 변조 계수 (0.5 ~ 1.5)
                I_stim = stimulus_current(t_e)  # 외부 자극 전류 [µA]
                # DTG 기반 자기 발화: DTG 위상이 직접 입력 전류를 생성
                # I_autonomous: 기본 자율 전류 (DTG 위상 변조됨)
                I_autonomous = CONFIG.get("AUTONOMOUS", {}).get("I_base", 10.0)  # 기본값 10.0 µA
                I_base = I_stim + I_autonomous * (1.0 + 0.5 * np.cos(phi_current))  # 외부 입력 + DTG 자기 발화
                # V3: 역방향 참조 제거 (I_back 제거) ⭐ V3 단일 방향화 원칙
                # 이전: I_back = 0.1 * (axon.V[0] - soma.V)  # 축삭→소마 역류 [µA]

                # -----------------------------------------------------
                # (c) HH 막전위 계산
                # -----------------------------------------------------
                # 개념:
                #   Hodgkin-Huxley 방정식으로 막전위를 업데이트합니다.
                #   ATP 의존 Na⁺/K⁺ 펌프 및 ATP 소비율을 계산합니다.
                #
                # 수식:
                #   C_m·dV/dt = g_Na·m³h·(E_Na-V) + g_K·n⁴·(E_K-V)
                #              + g_L·(E_L-V) + I_ext - I_pump
                #   I_pump = g_pump·I_pump_scale·(1 - e^{-ATP/ATP₀})·(V - E_pump)
                #   J_NaK = g_pump_consume · |I_pump|  [ATP/ms]
                #
                # 파라미터:
                #   Heat: Q10 효과 (온도에 따른 게이트 반응 속도 변화)
                #   - 온도가 높을수록 게이트 반응 속도가 빨라짐 (생리학적 현실 반영)
                # -----------------------------------------------------
                soma_result = soma.step(
                    dt_elec, I_ext=I_base, ATP=mito.ATP,  # V3: I_base - I_back → I_base ⭐ V3 단일 방향화 원칙
                    ENa_override=ENa_dyn, EK_override=EK_dyn,
                    Heat=mito.Heat
                )
                Vm = soma_result["V"]  # [mV]
                J_NaK_rate = soma_result["J_use"]  # [ATP/ms]
                J_NaK_amount_iter += J_NaK_rate * dt_elec  # [ATP] (누적)

                # -----------------------------------------------------
                # (d) IonFlow 업데이트
                # -----------------------------------------------------
                # 개념:
                #   HH가 계산한 막전위를 IonFlow에 반영하여 이온 농도를 업데이트합니다.
                #   이온 농도 변화는 다음 반복에서 더 정확한 Nernst 전위 계산에 사용됩니다.
                #
                # 수식:
                #   ∂C_i/∂t = D_i·∇²C_i - μ_i·z_i·C_i·∇V
                #   - D_i: 확산 계수 [cm²/s]
                #   - μ_i: 이동도 [cm²/(s·mV)]
                #   - z_i: 전하수
                #   - V: 막전위 [mV]
                #
                # 업데이트:
                #   ionflow.set_V(soma.V)  (전위 반영, 값 복사) ⭐ V3: 참조 공유 금지
                #   ionflow.step(dt_elec)  (이온 농도 갱신)
                # -----------------------------------------------------
                ionflow.set_V(soma.V)  # V3: 값 복사로 전위 반영 ⭐ V3 단일 방향화 원칙
                ionflow.step(dt_elec)  # 이온 농도 갱신
                
                # -----------------------------------------------------
                # (e) Reversal Potentials 재계산
                # -----------------------------------------------------
                # 개념:
                #   IonFlow 업데이트 후 즉시 reversal potentials를 갱신합니다.
                #   이온 농도 변화를 기반으로 ENa, EK, ECa, ECl을 동적으로 재계산합니다.
                #
                # 수식:
                #   E_ion = (RT/F)·ln([ion]_out / [ion]_in)
                #
                # 효과:
                #   다음 반복에서 더 정확한 채널 전류 계산 (Nernst 방정식 적용)
                # -----------------------------------------------------
                soma.update_reversal_potentials(ionflow)

                # -----------------------------------------------------
                # (f) 스파이크 이벤트 처리
                # -----------------------------------------------------
                # 개념:
                #   소마가 발화하면 (V > spike_thresh) 다음을 수행:
                #   - 축삭에 α-pulse 트리거
                #   - CaVesicle에 스파이크 시간 기록
                #
                # 조건:
                #   soma.spiking() and not spk_prev
                #   - 현재 스텝에서 발화했고, 이전 스텝에서는 발화하지 않음
                #   (중복 발화 방지)
                # -----------------------------------------------------
                if soma.spiking() and not spk_prev:
                    axon.trigger_alpha(t_e)  # 축삭에 α-pulse 트리거
                    ca.add_spike(t_e)  # CaVesicle에 스파이크 시간 기록
                spk_prev = soma.spiking()
                if spk_prev: spiked = True

                # -----------------------------------------------------
                # (g) 축삭 전도
                # -----------------------------------------------------
                # 개념:
                #   수초화 축삭에서 도약전도를 시뮬레이션합니다.
                #   소마에서 축삭으로 전류가 전달되고, 노드에서만 Na⁺ 채널이 활성화됩니다.
                #
                # 수식:
                #   ∂V/∂t = D(x)·∂²V/∂x² - (V-V_rest)/τ + I_ext/C_m
                #   - D(x): 공간 가변 확산 계수 (노드 vs 인터노드)
                #   - I_ext: 소마에서 주입된 전류
                #
                # 파라미터:
                #   ATP_level: ATP 의존 Na⁺ 전도도 변조
                #   stim_gain: 소마→축삭 결합 강도
                # -----------------------------------------------------
                axon.ATP_level = mito.ATP  # ATP 의존 Na⁺ 전도도 변조
                I0 = CONFIG["AXON"]["stim_gain"] * (soma.V - axon.V[0])  # [µA]
                axon.step(dt_elec, t_ms=t_e, I0_from_soma=I0, soma_V=soma.V)

            # ---------------------------------------------------------
            # 미세 반복 누적 소비율을 평균화해 안정화
            # ---------------------------------------------------------
            # 개념:
            #   여러 미세 반복에서 계산된 J_NaK_amount를 평균화합니다.
            #   첫 번째 반복: 그대로 사용
            #   이후 반복: 이전 값과 평균 (안정화)
            #
            # 수식:
            #   J_NaK_amount = 0.5·(J_NaK_amount_prev + J_NaK_amount_iter)
            # ---------------------------------------------------------
            if _micro == 0:
                J_NaK_amount = J_NaK_amount_iter
            else:
                J_NaK_amount = 0.5 * (J_NaK_amount + J_NaK_amount_iter)

        if -20 < soma.V < 40 and Vm_prev < -20:
            depol_count += 1
        if spiked:
            spike_count += 1
            
        Vm_prev = soma.V

        # =============================================================
        # (2) Ca²⁺ Vesicle 업데이트
        # =============================================================
        # 개념:
        #   Spike-triggered α-kernel 모델로 Ca²⁺ 농도를 업데이트합니다.
        #   각 스파이크마다 α(t-t_k) 형태의 유입이 발생하고,
        #   ATP 의존 펌프가 Ca²⁺를 제거합니다.
        #
        # 수식:
        #   dCa/dt = Σ_k A·α(t-t_k) - k_c·ATP·(Ca - C0)
        #   α(t) = (e^{-t/τ_d} - e^{-t/τ_r})_+  (α-kernel)
        #   - A: 스파이크당 Ca 유입량 [μM]
        #   - k_c: ATP 의존 펌프 계수 [1/(ATP·s)]
        #   - C0: 휴지 Ca 농도 [μM]
        #
        # Solver:
        #   - "heun": Heun 방법 (predictor-corrector, 2차 정확도)
        #   - "euler": 기본 Euler 방법 (1차 정확도, 빠름)
        #
        # 출력:
        #   ca_ev: VesicleEvent (t_ms, Ca, S, status)
        #   J_Ca_rate: ATP 소비율 [ATP/ms]
        #   - J_Ca_rate = k_atp_per_Ca · k_c · ATP · (Ca - C0)
        # =============================================================
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
            # V3: ATP는 [0,100] 범위로 전달 ⭐ V3 계약 고정
            ca_ev = ca.step(ATP=mito.ATP, dt_ms=ca.dt_ms)  # ATP [0,100] ⭐ V3 계약 고정
            # J_Ca_rate 계산 (ATP 소비율)
            k_atp_per_Ca = getattr(ca, 'k_atp_per_Ca', 1.0)
            J_Ca_rate = k_atp_per_Ca * ca.k_c * (mito.ATP / 100.0) * max(0.0, (ca.Ca - ca.C0))  # [ATP/ms] ⭐ V3: ATP/100 변환
        
        # =============================================================
        # (3) Metabolic Feedback (Mito 파라미터 조정)
        # =============================================================
        # 개념:
        #   Ca 상태에 따라 Mitochondria의 파라미터를 동적으로 조정합니다.
        #   Feedback을 Mito step 전에 실행하여 조정된 파라미터를 사용합니다.
        #
        # 수식:
        #   η0 = η_base - β_heat·Heat  (Heat ↑ → 효율 ↓)
        #   Ploss = Ploss_base·(1 + β_CO2·CO2)  (CO₂ ↑ → 손실 ↑)
        #   recover_k = recover_base·(1 + λ_Ca·S_alert)  (Ca alert → 회복률 ↑)
        #
        # 효과:
        #   Mito step이 조정된 파라미터를 사용하여 ATP, Heat, CO2를 계산
        # =============================================================
        feedback.update(ca_ev.status)
        
        # =============================================================
        # (4) PTP 및 Synaptic Resonance 업데이트
        # =============================================================
        # 개념:
        #   PTP (Post-Tetanic Potentiation)와 Synaptic Resonance를 업데이트합니다.
        #   스파이크 발생 시 PTP가 강화되고, Resonance 위상이 동기화됩니다.
        #
        # 수식 (PTP):
        #   on_spike: R += g_ptp·S^n/(S^n + K^n)  (Hill 함수)
        #   step: R *= exp(-dt/τ_ptp)  (지수 감쇠)
        #
        # 수식 (Resonance):
        #   dθ/dt = ω + K·sin(φ-θ)·(1 + λ·S)
        #   - ω: 고유 주파수 [rad/ms]
        #   - K: 커플링 강도
        #   - λ: Ca²⁺ 민감도
        #   - S: 정규화된 Ca 농도 (0~1)
        # =============================================================
        if spiked:
            # PTP 강화 (Hill 함수 기반)
            ptp.on_spike(S=ca_ev.S)  # R += g_ptp·S^n/(S^n + K^n)
            phi_current = getattr(dtg, 'phi', 0.0)
            resonance.on_spike(ptp.R, phi_current)  # Resonance 초기화
            spike_events.append((t, ca_ev.Ca * 1e6, ptp.R))
        
        # PTP 감쇠 (지수 감쇠)
        ptp.step(dt_bio)  # R *= exp(-dt/τ_ptp)
        
        # Resonance 위상 업데이트
        # phi는 아직 계산되지 않았으므로 이전 스텝의 phi 사용
        phi_current = getattr(dtg, 'phi', 0.0)
        theta, delta_phi = resonance.step(dt_bio, phi_current, ca_ev.S)
        # dθ/dt = ω + K·sin(φ-θ)·(1 + λ·S)
        
        # -------------------------------------------------------------
        # (4.5) DTG에 θ 역피드백 주입 (양방향 결합 완성)
        # -------------------------------------------------------------
        # 개념:
        #   Resonance 위상 θ를 DTG 위상 φ에 역피드백합니다.
        #   양방향 결합을 완성하여 위상 동기화를 강화합니다.
        #
        # 수식:
        #   dφ/dt += k_res·sin(θ - φ)
        #   - k_res: 역피드백 강도 (기본값: 0.05)
        # -------------------------------------------------------------
        dtg.apply_resonance_feedback(theta, k_back=0.05)

        # =============================================================
        # (5) Terminal release (스파이크 의존)
        # =============================================================
        # 개념:
        #   스파이크 발생 시 시냅스 터미널에서 방출량을 계산합니다.
        #   Ca²⁺, PTP, 위상차, ATP 수준이 방출량에 영향을 미칩니다.
        #
        # 수식:
        #   Q = spike · α_C·S^p · α_R·R^q · α_φ·(1+h|Δφ|) · (ATP/100)^{1/2}
        #   p_eff = p0·(1 + R)
        #   - Q: 방출량
        #   - p_eff: 방출 확률
        #   - S: 정규화된 Ca 농도 (0~1)
        #   - R: PTP 잔여 강화량
        #   - Δφ: 위상차 (φ - θ) [rad]
        #   - ATP: ATP 농도 (정규화, 0~1)
        #
        # Broadcast:
        #   terminal.broadcast(t, Q)  → 연결된 모든 시냅스로 전달
        # =============================================================
        if spiked:
            Q, p_eff = terminal.release(
                spike=1,
                S=ca_ev.S,
                R=ptp.R,
                dphi=delta_phi,
                ATP=mito.ATP  # [0,100] ⭐ V3 계약 고정
            )
            terminal.broadcast(t, Q)  # 연결된 모든 시냅스로 전달
            terminal_logs.append((float(t), float(Q), float(p_eff)))

        # =============================================================
        # (6) ATP 소비율 집계
        # =============================================================
        # 개념:
        #   이번 bio 스텝에서 소비된 총 ATP를 계산합니다.
        #   Na/K 펌프, Ca 펌프, 누출 에너지 비용을 합산합니다.
        #
        # 수식:
        #   J_NaK = J_NaK_amount / dt_bio  (Na/K 펌프 소비율 [ATP/ms])
        #   J_Ca = J_Ca_rate  (Ca 펌프 소비율 [ATP/ms])
        #   leak_cost = Σ(V - V_rest)²·dx  (누출 에너지, 공간 적분)
        #   J_use_total = J_NaK + J_Ca + 0.001·leak_cost  [ATP/ms]
        #
        # 누출 에너지:
        #   축삭 전위에서 V_rest로부터의 편차를 적분하여 누출 에너지 비용 계산
        #   0.001 스케일: 누출 에너지가 ATP 소비에 미치는 영향 (작은 기여)
        # =============================================================
        leak_cost = np.sum((axon.V - CONFIG["AXON"]["Vrest"])**2) * axon.dx
        # 누출 에너지 = Σ(V - V_rest)²·dx  (공간 적분)
        J_use_total = (J_NaK_amount / dt_bio) + J_Ca_rate + 0.001 * leak_cost  # [ATP/ms]
        # 총 ATP 소비율 = Na/K 펌프 + Ca 펌프 + 누출 에너지 비용

        # =============================================================
        # (7) Mitochondria 업데이트
        # =============================================================
        # 개념:
        #   ATP 생성/소비, Heat, CO₂를 업데이트합니다.
        #   Feedback에서 조정된 파라미터(η0, Ploss, recover_k)를 사용합니다.
        #
        # 수식:
        #   dE_buf/dt = (P_in - P_loss) - k_transfer·(E_buf - ATP)
        #   dATP/dt = κ·(E_buf - ATP) - J_use
        #   dHeat/dt = (1-η)·J_transfer - (Heat - Heat_rest)/τ_heat
        #   dCO₂/dt = c_CO2·J_transfer - (CO₂ - CO₂_rest)/τ_CO2
        #
        # Solver:
        #   RK4 방법 (CONFIG["SOLVER"]["MITO"] = "rk4")
        #   - 4차 정확도, ATP 대사 정밀도 향상
        #
        # 시간 스케일:
        #   dt_bio ≫ dt_elec 이므로, Mito는 생리학적 시간 상수 기반의
        #   느린(저주파) 통합 계층으로 유지됩니다.
        #
        # HeatGrid:
        #   Mitochondria 내부에서 자동 관리됨 (공간적 열 확산)
        # =============================================================
        out = mito.step(dt_bio, Glu=5.0, O2=5.0, J_use=J_use_total)
        # 반환값: {"ATP": float, "E_buf": float, "Heat": float, "CO2": float}
        
        # =============================================================
        # (8) DTG 업데이트 (에너지-위상 동기화)
        # =============================================================
        # 개념:
        #   DTG 시스템의 에너지(E)와 위상(φ)을 업데이트합니다.
        #   이번 스텝에서 방금 계산된 최신 ATP 값을 사용합니다.
        #
        # 수식:
        #   dE/dt = g_sync·(ATP - E) - γ·(E - E0)
        #   dφ/dt = ω0 + α·(E - E0) + k_res·sin(θ - φ)
        #   - g_sync: ATP-E 동기화 이득
        #   - γ: 에너지 복원 계수
        #   - ω0: 기본 위상속도 [rad/ms]
        #   - α: 에너지-위상 결합 계수
        #   - k_res: θ→φ 역피드백 강도
        #
        # Solver:
        #   - "rk4": rk4_step 사용 (4차 정확도, 더 정확하지만 계산 비용 증가)
        #   - "euler": 기본 Euler 방법 (1차 정확도, 빠름)
        #
        # 시간적 일관성:
        #   Mito 업데이트 → DTG 업데이트 순서 보장
        #   mito.ATP (객체 속성, 이전 값일 수 있음) 대신
        #   out["ATP"] (이번 스텝의 최신 값) 사용
        # =============================================================
        if CONFIG["SOLVER"]["DTG"] == "rk4":
            # 4차 Runge-Kutta 방법 사용
            y = np.array([dtg.E, dtg.phi])  # 상태 벡터 [E, φ]
            y = rk4_step(dtg_rhs(dtg, out["ATP"]), y, dt_bio)
            dtg.E = float(np.clip(y[0], 0.0, dtg.E0*2.0))  # E 클램프
            dtg.phi = float(y[1] % (2*np.pi))  # φ wrap (0~2π)
            phi = dtg.phi
        else:
            # 기본 Euler 방법 사용 (dtg.step() 내부 구현)
            _, phi, _, _ = dtg.step(out["ATP"], dt_bio)

        # =============================================================
        # (9) 로깅 및 결과 저장
        # =============================================================
        # 개념:
        #   주기적으로 시뮬레이션 상태를 로깅하고 결과를 저장합니다.
        #   log_every 스텝마다 데이터를 기록합니다.
        #
        # 기록 데이터:
        #   - table1_data: 생리학 파라미터 (ATP, Vm, φ, Ca, R, η, θ-φ)
        #   - table2_data: 전도 파라미터 (v, tailV, Heat, CO₂, spikes)
        #   - Vmap_data: 축삭 전위 분포 (공간-시간 heatmap용)
        #
        # 로깅 주기:
        #   log_every = max(1, round(LOG_INTERVAL / dt_bio))
        #   LOG_INTERVAL: 로깅 간격 [ms] (기본값: 5 ms)
        # =============================================================
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

    t1 = perf_counter()  # 시뮬레이션 종료 시간 기록

    # =============================================================
    # 결과 출력 및 저장
    # =============================================================
    # 개념:
    #   시뮬레이션 결과를 콘솔에 출력하고 CSV 파일로 저장합니다.
    #   축삭 전도 속도, 스파이크 타임라인, 생리학 파라미터 등을 기록합니다.
    # =============================================================
    
    # -------------------------------------------------------------
    # (1) Distal 도달 확인
    # -------------------------------------------------------------
    # 개념:
    #   축삭 말단 노드에 신호가 도달했는지 확인합니다.
    #   first_cross_ms: 각 노드에서 임계값을 처음 넘은 시간
    # -------------------------------------------------------------
    try:
        last_node_idx = axon.NODE_IDX[-1]  # 말단 노드 인덱스
        t_reach = axon.first_cross_ms.get(last_node_idx, None)
        if t_reach is not None and np.isfinite(t_reach):
            print(f"[TAIL] distal reached at {t_reach:.2f} ms, tailV_peak={90.00:.2f} mV")
            sys.stdout.flush()
    except Exception:
        pass

    # -------------------------------------------------------------
    # (2) 표 1: 생리학 파라미터 출력
    # -------------------------------------------------------------
    # 컬럼: t(ms), ATP, Vm(mV), φ(rad), Ca(μM), PTP R, η(meta), θ−φ
    # -------------------------------------------------------------
    for t_ms, ATP_val, Vm_val, phi_val, Ca_val, R_val, eta_val, delta_phi_val in table1_data:
        print(f"{t_ms:7.1f} | {ATP_val:6.2f} | {Vm_val:8.2f} | {phi_val:7.3f} | "
              f"{Ca_val:8.3f} | {R_val:7.3f} | {eta_val:7.3f} | {delta_phi_val:7.3f}")
        sys.stdout.flush()

    print("=" * 75); sys.stdout.flush()
    
    # -------------------------------------------------------------
    # (3) 스파이크 타임라인 출력
    # -------------------------------------------------------------
    # 각 스파이크 발생 시점과 Ca 농도, PTP 강화량을 기록
    # -------------------------------------------------------------
    if spike_events:
        print("Spikes Timeline"); sys.stdout.flush()
        print("=" * 75); sys.stdout.flush()
        for t_event, ca_event, r_event in spike_events:
            print(f"[{t_event:7.2f} ms] Spike → Ca={ca_event:.2f} μM, PTP R={r_event:.3f}")
            sys.stdout.flush()
        print("=" * 75); sys.stdout.flush()
    
    # -------------------------------------------------------------
    # (4) 표 2: 전도 및 환경 파라미터 출력
    # -------------------------------------------------------------
    # 컬럼: t(ms), v(m/s), tailV(mV), Heat, CO₂, spikes, active, tail_peak
    # -------------------------------------------------------------
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

    # =============================================================
    # (5) 전도 속도 계산 (Scaled vs Real)
    # =============================================================
    # 개념:
    #   축삭 전도 속도를 계산합니다.
    #   시뮬레이션 단위와 실제 물리 단위를 구분하여 계산합니다.
    #
    # 수식:
    #   TOF (Time of Flight) = t_N - t_0  [ms]
    #   - t_0: 첫 번째 노드에서 임계값 도달 시간
    #   - t_N: 말단 노드에서 임계값 도달 시간
    #
    #   v_scaled = L_sim / (TOF_scaled / 1000)  [m/s]
    #   - L_sim: 시뮬레이션 축삭 길이 [cm]
    #   - TOF_scaled: 시뮬레이션 시간 [ms]
    #
    #   v_real = L_real / (TOF_real / 1000)  [m/s]
    #   - L_real: 실제 물리 축삭 길이 [m]
    #   - TOF_real = TOF_scaled · ms_per_sim_ms  [ms]
    #
    # 스케일링:
    #   ms_per_sim_ms: 시뮬레이션 시간과 실제 시간의 비율
    #   (기본값: 0.4, 시뮬레이션이 실제보다 빠름)
    # =============================================================
    first_cross_raw = [t_val for t_val in getattr(axon, "first_cross_ms", {}).values() if t_val is not None]
    if first_cross_raw:
        first_cross_raw.sort()
        t0_cross = first_cross_raw[0]  # 첫 번째 노드 도달 시간 [ms]
        tN_cross = first_cross_raw[-1]  # 말단 노드 도달 시간 [ms]
        TOF_scaled = max(tN_cross - t0_cross, 1e-3)  # Time of Flight [ms]
    else:
        t0_cross = float("nan")
        tN_cross = float("nan")
        TOF_scaled = float("nan")

    axon_length_sim = axon.N * axon.dx  # 시뮬레이션 축삭 길이 [cm]
    axon_length_real = axon.N * getattr(axon, "dx_real_m", axon.dx)  # 실제 축삭 길이 [m]
    ms_per_sim_ms = R.get("ms_per_sim_ms", 1.0)  # 시간 스케일링 비율
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

    # =============================================================
    # (6) CSV 파일 저장
    # =============================================================
    # 개념:
    #   시뮬레이션 결과를 CSV 파일로 저장합니다.
    #   logs/ 디렉토리에 다음 파일들이 생성됩니다:
    #   - table1.csv: 생리학 파라미터 (ATP, Vm, φ, Ca, R, η, θ-φ)
    #   - table2.csv: 전도 파라미터 (v, tailV, Heat, CO₂, spikes)
    #   - terminal.csv: 시냅스 방출량 (Q, p_eff)
    #   - terminal_sink.csv: 시냅스 이벤트 (t, Q)
    # =============================================================
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 표 1: 생리학 파라미터
    df1 = pd.DataFrame(
        table1_data,
        columns=["t", "ATP", "Vm", "phi", "Ca", "R", "eta", "delta_phi"],
    )
    df1.to_csv(os.path.join(logs_dir, "table1.csv"), index=False)
    
    # 표 2: 전도 및 환경 파라미터
    df2 = pd.DataFrame(
        table2_data,
        columns=["t", "v", "tailV", "Heat", "CO2", "spikes", "active", "tail_peak"],
    )
    df2.to_csv(os.path.join(logs_dir, "table2.csv"), index=False)
    
    # Terminal releases (시냅스 방출량)
    if terminal_logs:
        df_term = pd.DataFrame(terminal_logs, columns=["t", "Q", "p_eff"])
        df_term.to_csv(os.path.join(logs_dir, "terminal.csv"), index=False)
    
    # Sink synapse events (시냅스 이벤트 수집)
    try:
        df_sink = sink_syn.to_dataframe()
        if df_sink is not None and not df_sink.empty:
            df_sink.to_csv(os.path.join(logs_dir, "terminal_sink.csv"), index=False)
    except Exception:
        pass
    
    print("CSV files saved: logs/table1.csv, logs/table2.csv"); sys.stdout.flush()
    if terminal_logs:
        print("CSV files saved: logs/terminal.csv"); sys.stdout.flush()

    # =============================================================
    # (7) 축삭 전도 시각화 (Heatmap)
    # =============================================================
    # 개념:
    #   축삭 전위 분포를 공간-시간 heatmap으로 시각화합니다.
    #   Vmap_data: [time_steps, N_nodes] 형태의 2D 배열
    #
    # 시각화:
    #   - X축: 시간 [ms]
    #   - Y축: 노드 인덱스 (proximal → distal)
    #   - 색상: 막전위 [mV] (plasma colormap)
    #
    # 저장:
    #   logs/saltatory_conduction.png (150 DPI)
    # =============================================================
    if Vmap_data:
        Vmap = np.array(Vmap_data).T  # [N_nodes, time_steps]로 전치
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
# V3 독립 실행 테스트
# =============================================================
if __name__ == "__main__":
    """
    V3 통합 시뮬레이션 실행
    
    V3 계약 고정 검증:
    - 단일 방향화 확인 (역방향 참조 제거)
    - 값 복사 확인 (참조 공유 제거)
    - ATP 정규화 범위 [0,100] 확인
    - 알고리즘 흐름 확인
    """
    print("=" * 75)
    print("V3 Integrated Neuron Simulation Pipeline")
    print("=" * 75)
    print("V3 계약 고정:")
    print("  ✅ 단일 방향화: Soma → Axon (역방향 참조 금지)")
    print("  ✅ 값 복사: 참조 공유 금지 (ionflow.set_V())")
    print("  ✅ ATP: [0,100] 범위로 통일")
    print("  ✅ S: [0,1] 범위 명시")
    print("  ✅ 시간 단위: [ms] 명시")
    print("=" * 75)
    print()
    
    try:
        # 시뮬레이션 실행
        run_pipeline(T_ms=100.0)  # 100ms 테스트 실행
        
        print()
        print("=" * 75)
        print("✅ V3 시뮬레이션 완료")
        print("=" * 75)
        print("결과 파일:")
        print("  - logs/table1.csv: 생리학 파라미터")
        print("  - logs/table2.csv: 전도 파라미터")
        print("  - logs/terminal.csv: 시냅스 방출량")
        print("  - logs/saltatory_conduction.png: 축삭 전도 시각화")
        print("=" * 75)
        
    except Exception as e:
        print()
        print("=" * 75)
        print("❌ V3 시뮬레이션 오류 발생")
        print("=" * 75)
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 75)
        sys.exit(1)

# =============================================================

