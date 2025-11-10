# =============================================================
# new_run_quick.py — PTP 실험용 (정합성 패치 통합본, 범위가드/안정기본값 포함)
# =============================================================
# 목적:
#   • PTP (Post-Tetanic Potentiation) 실험용 페어 펄스/트레인 자극 프로토콜
#   • 빠른 시뮬레이션 + 도약 전도속도(v) 포함 상세 로그 출력
#   • HH·Ca·PTP 모듈 API 불일치(hh/ca/ptp) 호환 어댑터 내장
#   • ★ 정상작동 범위 가드(물리·생리 범위 클램프/경고) + 보수적 기본값
#   • ★ “폭주” 방지: 외부자극/축삭노드 파라미터/칼슘커널/속도업데이트 안정화
# 사용:
#   $ python3 new_run_quick.py
# 출력:
#   - 표1(생리): t, ATP, Vm, φ, Ca(μM), PTP R, η(meta), θ−φ
#   - 표2(전도): t, v(m/s), tailV, Heat, CO₂, spikes, active
#   - CSV: logs/table1.csv, logs/table2.csv
#   - 도약 전도 시각화: logs/saltatory_conduction.png
# =============================================================

# Qquarts co Present # 지은이 : GNJz

import time
import numpy as np
import os
import sys
import importlib.util
import yaml
import pandas as pd
import inspect
import math

# -------------------------------------------------------------
# 글로벌 상수
# -------------------------------------------------------------
PTP_TAU_DECAY_MS = 800.0
ETA_BETA = 0.4      # ATP 피드백 계수
K_META = 0.8        # metabolic drive gain for soma current
ALPHA_CA = 0.25     # Ca 유입 → ATP 생성 이득
BETA_ATP = 0.02     # ATP 회복 감쇠율
ATP_REST = 100.0    # ATP 평형 레벨

# 자가 순환 모드 설정
AUTONOMOUS_MODE = True
K_CA_FEEDBACK = 0.9          # Ca 농도 → 전류 피드백
CA_FEEDBACK_REST = 1.0       # μM 기준값
K_ATP_FEEDBACK = 0.02        # ATP 편차 → 전류 피드백
K_PHI_FEEDBACK = 8.0         # 위상 편차 → 전류 피드백
PHI_DAMPING = 0.05           # 위상 피드백 감쇠

# =============================================================
# 유틸(범위 가드/경고)
# =============================================================
def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def warn_once(flag_dict, key, msg):
    if key not in flag_dict:
        print(f"[warn] {msg}")
        sys.stdout.flush()
        flag_dict[key] = True

WARNED = {}

# =============================================================
# MyelinAxon — 이벤트 기반 도약 전도 모델 (안정화 버전)
# =============================================================
class MyelinAxon:
    """
    Myelinated Axon (Event-Driven)
    ----------------------------------------
    - 소마 스파이크 시 node0으로 이벤트 입력
    - 각 노드는 delay=dx/v 후 다음 노드로 이벤트 전달(실제 물리식 사용)
    - 각 노드는 짧은 알파형 전위(u_amp, tau_node)로 응답 (과도전위는 수 mV~수십 mV)
    - 휴지전위로 자동 복귀, 불응기(refraction) 존재
    - ★ 변경:
        * scale 제거: 실제 지연 = (dx/v)*1e3 (ms)
        * u_amp 기본 40mV, tau_node=0.8ms, refrac_ms=1.0ms, spike_thresh=-40mV
        * tailV는 말단 노드 전위
        * 속도업데이트: ATP, Vm_soma 민감도 축소(폭주 방지)
    """

    def __init__(self, cfg):
        # 물리 파라미터
        self.N = int(cfg.get("N", 120))
        self.dx = float(cfg.get("dx", 1e-3))     # m
        self.dx_real_m = float(cfg.get("dx_real_m", self.dx))
        self.v = float(getattr(self, "v", 50.0))  # 초기 속도 기본값
        self.v_min = float(cfg.get("v_min", 5.0))
        self.v_max = float(cfg.get("v_max", 120.0))
        self.alpha_v = float(cfg.get("alpha_v", 0.004))  # ATP 민감도
        self.beta_v  = float(cfg.get("beta_v", 0.0001))  # Vm 민감도

        # 전위 파형 파라미터
        self.u_rest = -70.0
        self.u_amp = float(cfg.get("u_amp", 70.0))  # 과도전위 mV
        self.u_amp = max(self.u_amp, 30.0)
        self.tau_node = float(cfg.get("tau_node", 2.2))  # ms
        self.tau_node = max(self.tau_node, 1.5)
        self.spike_thresh = float(cfg.get("spike_thresh", -40.0))
        self.refrac_ms = float(cfg.get("refrac_ms", 1.0))
        self.k_diff = float(cfg.get("k_diff", 0.005))  # coupling 완화로 난발사 억제
        self.coupling_alpha = float(cfg.get("coupling_alpha", 0.04))
        self.alpha_ATP = float(cfg.get("alpha_ATP", 0.5))
        self.beta_ATP = float(cfg.get("beta_ATP", 0.05))

        # 상태
        self.u = np.full(self.N, self.u_rest, dtype=float)
        self.last_spike_t = np.full(self.N, -1e9, dtype=float)
        self.queue = [[] for _ in range(self.N)]
        self.time_ms = 0.0
        self.total_spikes = 0
        self.tailV = self.u_rest
        self.length_m = (self.N - 1) * self.dx
        self.spike_log = [[] for _ in range(self.N)]  # 각 노드 스파이크 시간 기록
        self.first_cross = np.full(self.N, np.nan, dtype=float)
        self.tail_hit = False
        self.tail_hit_t = None
        self.tail_hit_v = None
        self.ATP = 100.0
        self.Ca = 0.05
        self.heat_gain = float(cfg.get("heat_gain", 1.0e-3))
        self.tau_heat_ms = float(cfg.get("tau_heat_ms", 150.0))
        self.Heat_rest = float(cfg.get("heat_rest", 0.2))
        self.k_heat_env = float(cfg.get("k_heat_env", 0.004))
        self.heat_env = float(cfg.get("heat_env", 0.0))
        self.co2_gain = float(cfg.get("co2_gain", 0.08))
        self.tau_co2_ms = float(cfg.get("tau_co2_ms", 400.0))
        self.CO2_rest = float(cfg.get("co2_rest", 0.12))
        self.k_co2_clear = float(cfg.get("k_co2_clear", 0.003))
        self.co2_env = float(cfg.get("co2_env", 0.0))
        self.Heat = float(self.Heat_rest)
        self.CO2 = float(self.CO2_rest)
        self.Vm_tail_hist = [self.u[-1]]
        self.t_ms = 0.0
        self.tail_reached = False
        self.t_reach_ms = None
        self.tail_peak_val = self.u[-1]

    def _node_delay_ms(self):
        return (self.dx / max(self.v_min, min(self.v, self.v_max))) * 1e3

    def on_soma_spike(self, t_ms):
        """소마에서 상승엣지 감지 시 node0으로 이벤트 주입"""
        self.queue[0].append(float(t_ms))

    def _fire_node(self, i):
        # 노드 과도전위
        self.u[i] = max(self.u[i], self.spike_thresh + self.u_amp)
        self.last_spike_t[i] = self.time_ms
        self.total_spikes += 1
        self.spike_log[i].append(self.time_ms)  # 스파이크 시간 기록
        if np.isnan(self.first_cross[i]):
            self.first_cross[i] = float(self.time_ms)
        if i == self.N - 1 and self.u[i] > (self.spike_thresh + 5.0):
            if not self.tail_hit:
                self.tail_hit = True
                self.tail_hit_t = float(self.time_ms)
                self.tail_hit_v = float(self.u[i])
            if np.isnan(self.first_cross[i]):
                self.first_cross[i] = float(self.time_ms)
        # 다음 노드에 이벤트 예약
        if i < self.N - 1:
            delay_sim = self._node_delay_ms()
            self.queue[i + 1].append(self.time_ms + delay_sim + 0.05)
            # 이벤트 수 과도 누적 방지
            if len(self.queue[i + 1]) > 8:
                self.queue[i + 1] = self.queue[i + 1][-8:]

    def step(self, dt_ms, Vm_soma, ATP, soma_fired=False):
        self.time_ms += dt_ms
        # 속도 적응(ATP, 평균 전위 기반)
        self.ATP = float(ATP)
        self.Vm = self.u.copy()
        Vm_mean = float(np.mean(self.Vm))
        dv = self.alpha_v * (self.ATP - ATP_REST) - self.beta_v * (Vm_mean + 65.0)
        self.v = float(np.clip(self.v + dv * dt_ms, 1.0, self.v_max))
        dATP = ALPHA_CA * (self.Ca - 0.05) - BETA_ATP * (self.ATP - ATP_REST)
        self.ATP = float(np.clip(self.ATP + dATP * dt_ms, 80.0, 120.0))
        heat_prod = abs(dATP) * self.heat_gain
        heat_decay = (self.Heat - self.Heat_rest) / max(self.tau_heat_ms, 1e-3)
        heat_env_loss = self.k_heat_env * (self.Heat - self.heat_env)
        self.Heat += (heat_prod - heat_decay - heat_env_loss) * dt_ms
        self.Heat = max(self.Heat_rest, float(self.Heat))

        co2_prod = self.co2_gain * (self.Heat - self.Heat_rest)
        co2_decay = (self.CO2 - self.CO2_rest) / max(self.tau_co2_ms, 1e-3)
        co2_env_loss = self.k_co2_clear * (self.CO2 - self.co2_env)
        self.CO2 += (co2_prod - co2_decay - co2_env_loss) * dt_ms
        self.CO2 = max(self.CO2_rest, float(self.CO2))
        self.t_ms = self.time_ms

        # 1) 이벤트 처리 (전위 감쇠 전에)
        for i in range(self.N):
            arrived = [t for t in self.queue[i] if t <= self.time_ms]
            if arrived and (self.time_ms - self.last_spike_t[i]) >= (self.refrac_ms - 1e-6):
                self._fire_node(i)
            # 남은 예정 이벤트 유지
            self.queue[i] = [t for t in self.queue[i] if t > self.time_ms]

        # 3) 강화된 확산형 전위 전달 (2차 차분)
        d2u = np.zeros(self.N, dtype=float)
        if self.N > 2:
            d2u[1:-1] = self.u[0:-2] - 2.0 * self.u[1:-1] + self.u[2:]
        if self.N > 1:
            d2u[0] = self.u[1] - self.u[0]
            d2u[-1] = self.u[-2] - self.u[-1]
        self.u += self.k_diff * d2u * dt_ms
        # 포화 방지
        self.u = np.minimum(self.u, self.spike_thresh + self.u_amp)

        # 4) 노드 파형 감쇠(알파/지수형 단순화)
        decay = math.exp(-dt_ms / max(self.tau_node, 1e-6))
        for i in range(self.N):
            # 발화 직후 1ms 이내 노드는 유지, 나머지는 복귀
            if (self.time_ms - self.last_spike_t[i]) > 2.0:
                self.u[i] = self.u_rest + (self.u[i] - self.u_rest) * decay

        # tailV 업데이트
        self.tailV = float(self.u[-1])
        self.Vm_tail_hist.append(self.tailV)
        if len(self.Vm_tail_hist) > 512:
            self.Vm_tail_hist = self.Vm_tail_hist[-512:]

        Vm_mean_tail = float(self.Vm_tail_hist[-1])
        prev_tail_V = self.Vm_tail_hist[-2] if len(self.Vm_tail_hist) >= 2 else self.Vm_tail_hist[-1]
        dV_tail = (Vm_mean_tail - prev_tail_V) / max(dt_ms, 1e-6)
        if self.tail_hit and not self.tail_reached:
            self.tail_reached = True
            self.t_reach_ms = float(self.tail_hit_t)
            self.tail_peak_val = float(self.tail_hit_v)
        elif self.tail_reached:
            self.tail_peak_val = max(self.tail_peak_val, self.tailV)

        return float(self.v)

    def get_status(self):
        active_nodes = int(np.sum(self.u > self.spike_thresh))
        return (
            active_nodes,
            int(self.total_spikes),
            float(self.u[-1]),
            self.tail_hit,
            self.tail_hit_t,
            self.tail_hit_v,
        )


# --- 계층 임포트 (숫자 시작 파일 동적 import) ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_bio_neurons_dir = os.path.join(_current_dir, 'bio_neurons')

if not os.path.isdir(_bio_neurons_dir):
    alt_dir = os.path.abspath(os.path.join(_current_dir, '..', 'bio_neurons'))
    if os.path.isdir(alt_dir):
        _bio_neurons_dir = alt_dir
    else:
        raise FileNotFoundError(
            "Could not locate 'bio_neurons' directory. "
            "Checked paths:\n"
            f"  - {_bio_neurons_dir}\n"
            f"  - {alt_dir}"
        )

candidate_file = os.path.join(_bio_neurons_dir, '1.dtg_system.py')
if not os.path.isfile(candidate_file):
    nested_dir = os.path.join(_bio_neurons_dir, 'bio_neurons')
    nested_file = os.path.join(nested_dir, '1.dtg_system.py')
    if os.path.isfile(nested_file):
        _bio_neurons_dir = nested_dir
    else:
        raise FileNotFoundError(
            "Expected module '1.dtg_system.py' not found.\n"
            f"  Checked: {candidate_file}\n"
            f"  Checked: {nested_file}"
        )

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 각 모듈 import (절대 경로 사용)
dtg_module = import_module_from_path('dtg_system', os.path.join(_bio_neurons_dir, '1.dtg_system.py'))
mito_module = import_module_from_path('mitochon_atp', os.path.join(_bio_neurons_dir, '2.mitochon_atp.py'))
hh_module = import_module_from_path('hh_soma', os.path.join(_bio_neurons_dir, '3.hh_soma.py'))
ion_module = import_module_from_path('ion_flow', os.path.join(_bio_neurons_dir, '4.ion_flow.py'))
ca_module = import_module_from_path('ca_vesicle', os.path.join(_bio_neurons_dir, '6.ca_vesicle.py'))
ptp_module = import_module_from_path('ptp', os.path.join(_bio_neurons_dir, '7.ptp.py'))
resonance_module = import_module_from_path('synaptic_resonance', os.path.join(_bio_neurons_dir, '9.synaptic_resonance.py'))

# 클래스 가져오기
DTGSystem = dtg_module.DTGSystem
Mitochondria = mito_module.Mitochondria
HHSoma = hh_module.HHSoma
IonFlowDynamics = ion_module.IonFlowDynamics
CaVesicle = ca_module.CaVesicle
PTPPlasticity = ptp_module.PTPPlasticity
PTPConfig = ptp_module.PTPConfig
SynapticResonance = resonance_module.SynapticResonance

# -------------------------------------------------------------
# CONFIG (안정 기본값)
# -------------------------------------------------------------
RUN_CONFIG = {
    "T_total": 500,      # 총 시뮬레이션 시간 [ms]
    "dt_elec": 0.02,     # 단일 타임스텝 (ms)
    "log_interval": 5,
    "pre_run_ms": 20.0,
    "ms_per_sim_ms": 0.4,  # 1 시뮬레이션 ms가 대응하는 실제 ms
}

NEURON_CFG = {
    "RUN": RUN_CONFIG,
    "DTG": {
        "E0": 100.0,
        "omega0": 1.0,
        "alpha": 0.02,      # 동조 민감도 낮춤
        "gamma": 0.08,
        "sync_gain": 0.15
    },
    "MITO": {
        "ATP0": 105.0,
        "Ebuf0": 70.0,
        "eta": 0.60,
        "k_transfer": 0.3,
        "Ploss": 1.2,
        "Heat0": 0.0,
        "CO2_0": 0.0,
        "D_H": 1e-6,
        "dx_heat": 1e-3,
        "k_heat": 0.01,
        "Heat_env": 0.0,
        "k_co2": 0.01,
        "CO2_env": 0.0
    },
    "HH": {
        "V0": -70.0,
        "gNa": 220.0,
        "gK": 26.0,
        "gL": 0.08,
        "ENa": 50.0,
        "EK": -77.0,
        "EL": -54.4,
        "spike_thresh": 0.0,     # 스파이크 검출용(0mV 교차)
        "use_pump": True,
        "g_pump": 0.05,          # 펌프 기본
        "E_pump": -70.0,
        "ATP0_ref": 100.0,
        "g_pump_consume": 0.02,
        "tau_h_Na_scale": 1.2    # Na inactivation 조금 느리게(폭주 방지)
    },
    "ION": {
        "N": 121,
        "dx": 1e-3,
        "dx_real_m": 1e-3,
        "Vrest": -70.0
    },
    "AXON": {
        "N": 120,
        "dx": 1e-3,
        "v_init": 60.0,          # 보수적 도약속도
        "v_min": 5.0,
        "v_max": 150.0,
        "alpha_v": 0.004,
        "beta_v": 0.0001,
        "spike_thresh": -40.0,   # 노드 활성 임계
        "u_amp": 130.0,         # 약 90 mV 목표
        "tau_node": 2.2,
        "refrac_ms": 2.5,
        "k_diff": 0.8,
        "heat_gain": 1.0e-3,
        "tau_heat_ms": 150.0,
        "heat_rest": 0.2,
        "k_heat_env": 0.004,
        "heat_env": 0.0,
        "co2_gain": 0.08,
        "tau_co2_ms": 400.0,
        "co2_rest": 0.12,
        "k_co2_clear": 0.003,
        "co2_env": 0.0
    },
    "CA": {
        # 내부 단위는 SI(mol/L); 표 출력 시 μM로 변환
        "C0": 1e-7,         # 0.1 μM
        "Cmax": 5e-6,       # 5 μM
        "A": 0.25e-6,       # 스파이크 유입 커널 스케일
        "tau_r": 0.5,       # ms
        "tau_d": 80.0,      # ms — 느린 세척
        "k_c": 0.02,        # Ca 펌프 약화(너무 강하면 0으로 붙음)
        "max_spike_memory_ms": 2000.0
    },
    "RESONANCE": {
        "omega": 1.0,
        "K": 0.02,
        "lambda_ca": 1.0
    },
    "STIMULUS": {
        "protocol": "train",
        "pulse1": {"start": 10, "end": 20, "amplitude": 120.0},  # legacy
        "pulse2": {"start": 100, "end": 110, "amplitude": 120.0},
        "base": 0.0,
        "train": {
            "start": 20.0,
            "end": 320.0,
            "f_hz": 30.0,
            "width_ms": 4.0,
            "amp": 260.0,
            "base": 0.0,
        },
    }
}

# YAML 설정 파일 (로그 검사용)
with open("neuron_config.yaml", "w") as f:
    yaml.dump(NEURON_CFG, f, default_flow_style=False, allow_unicode=True)

# -------------------------------------------------------------
# 객체 초기화 유틸
# -------------------------------------------------------------
ca_cfg = {
    "C0": 0.1e-6,       # 0.1 μM
    "Cmax": 5e-6,       # 5 μM
    "A": 0.15e-6,       # 유입커널 스케일 (μM/스파이크 상당)
    "tau_r": 0.5,       # ms
    "tau_d": 120.0,     # ms
    "k_c": 0.02,        # 펌프 적당
    "max_spike_memory_ms": 2000.0,
}

def reset_state():
    global dtg, mito, hh, ion, axon, ca, ptp, res
    dtg = DTGSystem(NEURON_CFG["DTG"])
    mito = Mitochondria({
        "ATP0": NEURON_CFG["MITO"]["ATP0"],
        "Ebuf0": NEURON_CFG["MITO"]["Ebuf0"],
        "eta": NEURON_CFG["MITO"]["eta"],
        "k_transfer": NEURON_CFG["MITO"]["k_transfer"],
        "Ploss": NEURON_CFG["MITO"]["Ploss"],
        "Heat0": NEURON_CFG["MITO"]["Heat0"],
        "CO2_0": NEURON_CFG["MITO"]["CO2_0"],
        "D_H": NEURON_CFG["MITO"]["D_H"],
        "dx_heat": NEURON_CFG["MITO"]["dx_heat"],
        "k_heat": NEURON_CFG["MITO"]["k_heat"],
        "Heat_env": NEURON_CFG["MITO"]["Heat_env"],
        "k_co2": NEURON_CFG["MITO"]["k_co2"],
        "CO2_env": NEURON_CFG["MITO"]["CO2_env"],
    })
    hh = HHSoma({
        "V0": NEURON_CFG["HH"]["V0"],
        "gNa": NEURON_CFG["HH"]["gNa"],
        "gK": NEURON_CFG["HH"]["gK"],
        "gL": NEURON_CFG["HH"]["gL"],
        "ENa": NEURON_CFG["HH"]["ENa"],
        "EK": NEURON_CFG["HH"]["EK"],
        "EL": NEURON_CFG["HH"]["EL"],
        "spike_thresh": NEURON_CFG["HH"]["spike_thresh"],
        "use_pump": NEURON_CFG["HH"]["use_pump"],
        "g_pump": NEURON_CFG["HH"]["g_pump"],
        "E_pump": NEURON_CFG["HH"]["E_pump"],
        "ATP0_ref": NEURON_CFG["HH"]["ATP0_ref"],
        "g_pump_consume": NEURON_CFG["HH"]["g_pump_consume"],
        "tau_h_Na_scale": NEURON_CFG["HH"]["tau_h_Na_scale"],
    })
    ion = IonFlowDynamics({"N": NEURON_CFG["ION"]["N"], "dx": NEURON_CFG["ION"]["dx"], "Vrest": NEURON_CFG["ION"]["Vrest"]})
    axon = MyelinAxon(NEURON_CFG["AXON"])
    ca = CaVesicle(ca_cfg, dt_ms=RUN_CONFIG["dt_elec"])
    ptp = PTPPlasticity(PTPConfig())
    res = SynapticResonance(omega=NEURON_CFG["RESONANCE"]["omega"], K=NEURON_CFG["RESONANCE"]["K"], lambda_ca=NEURON_CFG["RESONANCE"]["lambda_ca"])
    if hasattr(ptp, "step"):
        orig_step = ptp.step

        def _ptp_step_with_decay(*, dt_ms):
            R_val = orig_step(dt_ms=dt_ms)
            decay = math.exp(-dt_ms / max(1e-6, PTP_TAU_DECAY_MS))
            ptp.R = float(ptp.R * decay)
            return ptp.R

        ptp.step = _ptp_step_with_decay

reset_state()

# -------------------------------------------------------------
# 어댑터: 모듈 API 불일치 보정 (HHSoma, CaVesicle)
# -------------------------------------------------------------
def hh_step(hh_obj, dt, I_ext, ATP):
    """
    HHSoma.step 반환 형식(dict/tuple)을 통일해 (Vm, INa, IK, IL, I_pump, J_use)로 돌려줌.
    ★ 안정장치: Vm NaN/Inf 방지/클램프
    """
    out = hh_obj.step(dt=dt, I_ext=I_ext, ATP=ATP)
    if isinstance(out, dict):
        Vm = float(out.get("V"))
        INa = float(out.get("INa", 0.0))
        IK = float(out.get("IK", 0.0))
        IL = float(out.get("IL", 0.0))
        I_pump = float(out.get("I_pump", 0.0))
        J_use = float(out.get("J_use", 0.0))
    else:
        Vm, INa, IK, IL, I_pump, J_use = out

    if not np.isfinite(Vm):
        warn_once(WARNED, "hh_vm_nan", "Vm is not finite; clamping to [-120, 80] mV")
        Vm = -70.0
    Vm = clamp(Vm, -80.0, 70.0)
    return (Vm, INa, IK, IL, I_pump, J_use)

def ca_step(ca_obj, Vm, dt_ms, ATP):
    """
    CaVesicle.step 시그니처 차이를 흡수하고 통일된 출력 반환.
    반환: (event_or_none, J_Ca_rate, Ca_um, S[0..1])
    ★ 안정장치: Ca 범위(0~Cmax), S=[0,1]로 강제
    """
    event = None
    J_Ca_rate = 0.0
    Ca_val = None
    S = None

    kwargs = {}
    try:
        sig = inspect.signature(ca_obj.step)
        params = sig.parameters
        if "Vm" in params:
            kwargs["Vm"] = Vm
        if "dt_ms" in params:
            kwargs["dt_ms"] = dt_ms
        elif "dt" in params:
            kwargs["dt"] = dt_ms
        if "ATP" in params:
            kwargs["ATP"] = ATP
        result = ca_obj.step(**kwargs)
    except (TypeError, ValueError):
        try:
            result = ca_obj.step(Vm, dt_ms)
        except TypeError:
            result = ca_obj.step(dt_ms)

    if isinstance(result, tuple):
        maybe_event = result[0]
        if hasattr(maybe_event, "Ca") and hasattr(maybe_event, "S"):
            event = maybe_event
            if len(result) > 1 and isinstance(result[1], (int, float, np.floating)):
                J_Ca_rate = float(result[1])
        elif len(result) >= 2 and isinstance(result[0], (int, float, np.floating)):
            Ca_val = float(result[0])
            if isinstance(result[1], (int, float, np.floating)):
                J_Ca_rate = float(result[1])
        elif len(result) == 1 and isinstance(result[0], (int, float, np.floating)):
            Ca_val = float(result[0])
    elif hasattr(result, "Ca") and hasattr(result, "S"):
        event = result
    elif isinstance(result, dict):
        Ca_val = result.get("Ca")
        S = result.get("S")
        if isinstance(result.get("J_Ca_rate"), (int, float, np.floating)):
            J_Ca_rate = float(result["J_Ca_rate"])
    elif isinstance(result, (int, float, np.floating)):
        Ca_val = float(result)

    if event is not None:
        Ca_val = getattr(event, "Ca", Ca_val)
        S = getattr(event, "S", S)

    state = {}
    if Ca_val is None or S is None:
        if hasattr(ca_obj, "get_state"):
            try:
                state = ca_obj.get_state()
            except Exception:
                state = {}
    if Ca_val is None:
        Ca_val = state.get("Ca", getattr(ca_obj, "Ca", 0.0))
    if S is None:
        S = state.get("S")

    # 내부 기본 단위는 mol/L (SI)
    C0 = getattr(ca_obj, "C0", 0.1e-6)
    Cmax = getattr(ca_obj, "Cmax", 5e-6)
    Ca_val = float(clamp(Ca_val, 0.0, max(1e-12, Cmax)))
    # 정규화 S
    denom = max(1e-12, (Cmax - C0))
    S_calc = (Ca_val - C0) / denom
    if S is None or not np.isfinite(S):
        S = S_calc
    S = float(clamp(S, 0.0, 1.0))

    # μM 변환
    Ca_um = float(Ca_val * 1e6)

    # Ca 펌프 사용량이 음수/비정상일 경우 0
    if not np.isfinite(J_Ca_rate) or J_Ca_rate < 0.0:
        J_Ca_rate = 0.0

    return event, float(J_Ca_rate), float(Ca_um), float(S)

# -------------------------------------------------------------
# 자극 프로토콜 (안정 기본: pair-pulse)
# -------------------------------------------------------------
def pair_pulse_protocol(t):
    """페어 펄스 자극 프로토콜"""
    if 10 <= t < 20:
        return 300.0     # 1st pulse
    elif 100 <= t < 110:
        return 120.0     # 2nd pulse
    else:
        return 0.0       # baseline

def train_protocol(t, start=0, end=500, f_hz=100, width_ms=1, amp=300.0, base=0.0):
    if t < start or t >= end:
        return base
    period = 1000.0 / float(f_hz)
    phase = (t - start) % period
    return amp if phase < width_ms else base

def update_env(axon, dt_ms):
    if axon is None:
        return
    axon.heat_env += 0.001 * dt_ms
    axon.co2_env += 0.0005 * dt_ms

# -------------------------------------------------------------
# 실행
# -------------------------------------------------------------
print("[Neuron Pipeline Quick Run — with Velocity Log]")
sys.stdout.flush()

# ------------------------- #
# 표 1 헤더
# ------------------------- #
start = time.time()
ATP = float(getattr(mito, "ATP", NEURON_CFG["MITO"]["ATP0"]))
soma_ready = True
soma_refrac_ms = 1.0
last_soma_spike_ms = -1e9

table1_data = []
table2_data = []
Vmap_data = []
spike_events = []
prev_Ca_um = 2.0  # μM 기준값
phi_fb_state = 0.0

proto_name = NEURON_CFG["STIMULUS"]["protocol"].lower().strip()
use_pair = (proto_name == "pairpulse")
train_cfg = NEURON_CFG["STIMULUS"].get("train", {})

DT_STEP = RUN_CONFIG["dt_elec"]
LOG_INTERVAL = RUN_CONFIG["log_interval"]
PRE_RUN_MS = float(RUN_CONFIG.get("pre_run_ms", 0.0))

reported_tail = False
total_steps = int(round(RUN_CONFIG["T_total"] / DT_STEP)) + 1
pre_steps = max(0, int(round(PRE_RUN_MS / DT_STEP)))
total_steps_with_warmup = pre_steps + total_steps
log_every = max(1, int(round(LOG_INTERVAL / DT_STEP)))

print("=" * 95); sys.stdout.flush()
print("표 1: 생리학 파라미터"); sys.stdout.flush()
print("=" * 95); sys.stdout.flush()
print(f"{'t(ms)':>7} | {'ATP':>6} | {'Vm(mV)':>8} | {'φ(rad)':>7} | "
      f"{'Ca(μM)':>8} | {'PTP R':>7} | {'η(meta)':>7} | {'θ−φ':>7}")
sys.stdout.flush()
print("=" * 95); sys.stdout.flush()

for abs_step in range(total_steps_with_warmup):
    t_ms_raw = abs_step * DT_STEP
    sim_t_ms = t_ms_raw - PRE_RUN_MS
    warmup_phase = sim_t_ms < 0.0
    stim_t = sim_t_ms

    # 1) Mito
    out_m = mito.step(dt=DT_STEP, Glu=5.0, O2=5.0)
    ATP = float(out_m.get("ATP", ATP))
    if not np.isfinite(ATP):
        warn_once(WARNED, "mito_atp_nan", "ATP is not finite; resetting to 100")
        ATP = 100.0
        out_m["ATP"] = ATP

    # 2) DTG
    E, phi, dE, dphi = dtg.step(ATP, dt=DT_STEP)
    if not np.isfinite(phi):
        phi = 0.0

    # 3) Soma(HH)
    if AUTONOMOUS_MODE:
        I_ext = 0.0
    else:
        if use_pair:
            I_ext = pair_pulse_protocol(stim_t)
        else:
            I_ext = train_protocol(
                stim_t,
                start=train_cfg.get("start", 0.0),
                end=train_cfg.get("end", RUN_CONFIG["T_total"]),
                f_hz=train_cfg.get("f_hz", 20.0),
                width_ms=train_cfg.get("width_ms", 2.0),
                amp=train_cfg.get("amp", 10.0),
                base=train_cfg.get("base", 0.0),
            )

    # --- 내부 피드백 전류 구성 ---
    ATP_ref = NEURON_CFG["HH"].get("ATP0_ref", 100.0)
    eta_meta = 1.0 - ETA_BETA * max(0.0, (ATP_REST - ATP) / 50.0)
    eta_meta = clamp(eta_meta, 0.0, 1.25)
    I_meta = K_META * (ATP - ATP_ref) * eta_meta

    I_ca_fb = K_CA_FEEDBACK * (prev_Ca_um - CA_FEEDBACK_REST)
    I_atp_fb = K_ATP_FEEDBACK * (ATP - ATP_REST)
    I_phi_fb = K_PHI_FEEDBACK * phi_fb_state

    I_ext += I_meta + I_ca_fb + I_atp_fb + I_phi_fb
    Vm, INa, IK, IL, I_pump, J_use = hh_step(hh, DT_STEP, I_ext, ATP)
    if hasattr(hh, "tau_h_Na_scale"):
        hh.tau_h_Na_scale = float(np.clip(1.2 - 0.004 * (ATP - 100.0), 0.8, 1.4))

    # 4) Ca 업데이트(ATP 기반) + S, J_Ca_rate
    ev, J_Ca_rate, Ca_um, S = ca_step(ca, Vm, DT_STEP, ATP)
    axon.Ca = Ca_um
    prev_Ca_um = Ca_um

    # 스파이크 검출 (히스테리시스: spike_lo → spike_hi)
    spike_hi = -55.0
    spike_lo = -65.0
    if Vm <= spike_lo and (stim_t - last_soma_spike_ms) >= soma_refrac_ms:
        soma_ready = True
    soma_fired = False
    if soma_ready and Vm > spike_hi and (stim_t - last_soma_spike_ms) >= soma_refrac_ms:
        soma_fired = True
        soma_ready = False
        last_soma_spike_ms = stim_t

    if soma_fired:
        try:
            ca.add_spike(stim_t)
        except Exception:
            pass
        axon.on_soma_spike(stim_t)
        ptp.on_spike(S=S)
        if not warmup_phase:
            spike_events.append((stim_t, Ca_um, ptp.R))

    # ATP 소비 회계(펌프 + Ca펌프)
    ATP_use = 0.02 * abs(I_pump) + 0.5 * J_Ca_rate
    out_m["ATP"] = max(0.0, ATP - ATP_use * DT_STEP)
    ATP = float(out_m["ATP"])
    eta_feedback = 1.0 - ETA_BETA * max(0.0, NEURON_CFG["MITO"]["ATP0"] - ATP)
    out_m["eta"] = clamp(eta_feedback, 0.0, 1.0)

    # 5) PTP / 공명
    R = ptp.step(dt_ms=DT_STEP)
    if not np.isfinite(R):
        R = 1.0
    R = float(clamp(R, 0.0, 3.0))
    ptp.R = R
    theta, delta_phi = res.step(dt=DT_STEP, phi=phi, S=S)
    if not np.isfinite(delta_phi):
        delta_phi = 0.0
    phi_fb_state = (1.0 - PHI_DAMPING) * phi_fb_state + delta_phi

    # 6) 환경 갱신 + 축삭 스텝(속도 업데이트)
    update_env(axon, DT_STEP)
    v = axon.step(dt_ms=DT_STEP, Vm_soma=Vm, ATP=ATP, soma_fired=soma_fired)

    # 범위 가드 경고(한 번만)
    if Vm > 60.0 or Vm < -120.0:
        warn_once(WARNED, "vm_out", f"Vm out of range ({Vm:.1f} mV)")
    if Ca_um < 0.0 or Ca_um > 50.0:
        warn_once(WARNED, "ca_out", f"Ca out of range ({Ca_um:.2f} μM)")
    if v < 1.0 or v > 120.0:
        warn_once(WARNED, "v_out", f"v out of range ({v:.1f} m/s)")

    heat_val = float(getattr(axon, "Heat", out_m.get("Heat", 0.0)))
    co2_val = float(getattr(axon, "CO2", out_m.get("CO2", 0.0)))
    out_m["Heat"] = heat_val
    out_m["CO2"] = co2_val
    eta_val = float(out_m.get("eta", 0.0))

    if warmup_phase:
        if pre_steps > 0 and abs_step == pre_steps - 1:
            axon.first_cross[:] = np.nan
            axon.tail_hit = False
            axon.tail_hit_t = None
            axon.tail_hit_v = None
        continue

    sim_step_idx = abs_step - pre_steps
    if sim_step_idx < 0 or (sim_step_idx % log_every) != 0:
        continue

    (
        active_nodes,
        total_spikes,
        tailV_curr,
        tail_hit,
        tail_t,
        tail_v,
    ) = axon.get_status()

    if tail_hit and not reported_tail:
        tail_t_adj = (tail_t - PRE_RUN_MS) if tail_t is not None else None
        print(f"[TAIL] distal reached at {tail_t_adj:.2f} ms, tailV_peak={tail_v:.2f} mV")
        sys.stdout.flush()
        reported_tail = True

    tail_peak = bool(axon.u[-1] > axon.spike_thresh)

    table1_data.append((stim_t, ATP, Vm, phi, Ca_um, R, eta_val, delta_phi))
    table2_data.append(
        (
            stim_t,
            float(axon.v),
            tailV_curr,
            heat_val,
            co2_val,
            total_spikes,
            active_nodes,
            tail_peak,
        )
    )
    Vmap_data.append(axon.u.copy())

# 표 1 출력
for t, ATP_val, Vm_val, phi_val, Ca_val, R_val, eta_val, delta_phi_val in table1_data:
    print(f"{t:7.1f} | {ATP_val:6.2f} | {Vm_val:8.2f} | {phi_val:7.3f} | "
          f"{Ca_val:8.3f} | {R_val:7.3f} | {eta_val:7.3f} | {delta_phi_val:7.3f}")
    sys.stdout.flush()

# ------------------------- #
# 표 2
# ------------------------- #
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
    t,
    v_val,
    tailV_val,
    heat_val,
    co2_val,
    spike_count,
    active_nodes,
    tail_peak,
) in table2_data:
    print(
        f"{t:7.1f} | {v_val:7.2f} | {tailV_val:8.2f} | {heat_val:6.2f} | "
        f"{co2_val:6.2f} | {spike_count:7d} | {active_nodes:7d} | {str(tail_peak):>9}"
    )
    sys.stdout.flush()

print("=" * 75); sys.stdout.flush()
print(f"Done. Elapsed {time.time() - start:.3f} sec"); sys.stdout.flush()

# ==============================================================
# 도약 전도 속도(비행시간) 및 실제 전송속도 출력 — Scaled + Real
# ==============================================================

axon_length_m = axon.N * axon.dx
axon_length_real_m = getattr(axon, "dx_real_m", axon.dx) * (axon.N - 1)
ms_per_sim_ms = float(RUN_CONFIG.get("ms_per_sim_ms", 1.0))

TOF_scaled = None
t0_cross = None
tN_cross = None

# TOF 계산 (첫 노드 → 마지막 노드 첫 스파이크 시간 차)
first_cross_times = [t for t in getattr(axon, "first_cross", []) if np.isfinite(t)]
if len(first_cross_times) >= 2:
    t0 = min(first_cross_times)
    tN = max(first_cross_times)
    TOF_scaled = max(tN - t0, 0.001)
    t0_cross = t0 - PRE_RUN_MS
    tN_cross = tN - PRE_RUN_MS
else:
    TOF_scaled = float("nan")

# fallback: spike_log 전체 범위
if (TOF_scaled is None) or (not np.isfinite(TOF_scaled)) or TOF_scaled <= 0.0:
    if hasattr(axon, "spike_log"):
        spike_times = [t for node_log in axon.spike_log if node_log for t in node_log if np.isfinite(t)]
        if spike_times:
            first = min(spike_times)
            last = max(spike_times)
            t0_cross = first - PRE_RUN_MS
            tN_cross = last - PRE_RUN_MS
            TOF_scaled = max(tN_cross - t0_cross, 0.0)

if TOF_scaled is None or TOF_scaled <= 0.0:
    tail_times = [row[0] for row in table2_data if row[-1]]
    if len(tail_times) >= 2:
        t0_cross = tail_times[0]
        tN_cross = tail_times[-1]
        TOF_scaled = tN_cross - t0_cross

if TOF_scaled is None or TOF_scaled <= 0.0:
    t0_cross = 0.0
    tN_cross = RUN_CONFIG["T_total"]
    TOF_scaled = tN_cross - t0_cross

# --- ① 시뮬레이터 내부 스케일 속도 ---
v_scaled = axon_length_m / (TOF_scaled / 1000.0)  # [simulation length units per simulation second]

# --- ② 실제 물리 스케일 변환 (Config 기반) ---
TOF_real_ms = TOF_scaled * ms_per_sim_ms
v_real = axon_length_real_m / (TOF_real_ms / 1000.0) if TOF_real_ms > 0 else float("nan")

print("=" * 75)
print("[Transmission Velocity Summary — Scaled vs Real]")
print(f"TOF (ms)              : {TOF_scaled:.2f}")
print(f"TOF_real (ms)         : {TOF_real_ms:.2f}")
print(f"Axon length (m)       : {axon_length_m:.6f}")
print(f"Axon length real (m)  : {axon_length_real_m:.6f}")
print(f"v_scaled (sim units)  : {v_scaled:.2f} m/s")
print(f"v_real   (physical)   : {v_real:.2f} m/s")
print("CSV files saved : logs/table1.csv, logs/table2.csv")
print("Figure saved    : logs/saltatory_conduction.png")
print("=" * 75)
sys.stdout.flush()

# CSV 저장
os.makedirs("logs", exist_ok=True)
df1 = pd.DataFrame(table1_data, columns=["t", "ATP", "Vm", "phi", "Ca", "R", "eta", "delta_phi"])
df2 = pd.DataFrame(
    table2_data,
    columns=["t", "v", "tailV", "Heat", "CO2", "spikes", "active", "tail_peak"],
)
df1.to_csv("logs/table1.csv", index=False)
df2.to_csv("logs/table2.csv", index=False)
print("CSV files saved: logs/table1.csv, logs/table2.csv"); sys.stdout.flush()

# 시각화
if Vmap_data:
    import matplotlib.pyplot as plt
    Vmap = np.array(Vmap_data).T  # (N, time_steps)
    out_png = os.path.join("logs", "saltatory_conduction.png")
    plt.figure(figsize=(8, 4))
    plt.imshow(Vmap, aspect='auto', cmap='plasma', origin='lower',
               extent=[0, RUN_CONFIG["T_total"], 0, axon.N])
    plt.colorbar(label='Node transient (mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Node index (0→distal)')
    plt.title('Saltatory Conduction — Event-driven node transients')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()  # 그래프 창 표시
    print(f"Visualization saved: {out_png}")
    sys.stdout.flush()

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