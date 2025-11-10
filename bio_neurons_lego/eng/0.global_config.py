# =============================================================
# bioneuron_config_v1.py — Bio-Neuron Global Configuration
# =============================================================
#
# 🧠 PHAM Neural System (Qquarts Co.)
# Smart Neurons : New_Run.py — CONFIG ONLY
#
# Purpose
# -------------------------------------------------------------
#  • Defines the **Single Source of Truth (SSOT)** parameter set
#    for the unified neuron simulator.
#  • Each layer (DTG, Mito, HH, Axon, Ca, Feedback, Resonance)
#    corresponds to one biological process:
#        Energy ↔ Electricity ↔ Chemical ↔ Feedback
#  • Contains *no execution logic* — only parameters, units,
#    formulas, and biological meanings.
#
# Verification
# -------------------------------------------------------------
#  • CFL stability:
#       dt_elec ≤ 0.9 × dx² / (2 × max(D_node, D_internode))
#  • With default values, conduction velocity v ≈ 50–80 m/s
#    (depends on simulator implementation).
# -------------------------------------------------------------

from copy import deepcopy
from math import pi

CONFIG = {
    # =========================================================
    # [L0] DTG Layer — Energy–Phase Dynamics
    # =========================================================
    # Governing Equations
    #   dE/dt = g_sync·(ATP − E) − γ·(E − E₀)
    #   dφ/dt = ω₀ + α·(E − E₀)
    #
    # Meaning
    #   • φ (rad): intrinsic phase of neuron oscillator
    #   • E (arb): meta-energy scalar (macroscopic state)
    #   • Couples metabolic energy (ATP) with temporal phase φ.
    #   • Drives rhythmic synchronization across neuron modules.
    #
    # Units: t[ms], φ[rad], E[arb]
    # ---------------------------------------------------------
    "DTG": {
        "E0": 100.0,      # steady-state energy
        "omega0": 1.0,    # base angular frequency [rad/ms]
        "alpha": 0.03,    # energy–phase coupling gain
        "gamma": 0.10,    # relaxation rate of E
        "sync_gain": 0.20 # ATP–E synchronization factor
    },

    # =========================================================
    # [L1] Mitochondria Layer — Energy Metabolism
    # =========================================================
    # Governing Equations
    #   dE_buf/dt = (P_in − P_loss) − k_tr·(E_buf − ATP)
    #   dATP/dt   =  k_tr·(E_buf − ATP) − J_use
    #
    #   Heat↑ = (1−η)·k_tr·(E_buf−ATP)_+
    #   CO₂ ↑ = c_CO₂·k_tr·(E_buf−ATP)_+
    #
    # Meaning
    #   • E_buf: internal energy buffer (NADH pool)
    #   • ATP : available biochemical energy
    #   • η   : conversion efficiency (heat = 1−η)
    #   • J_use: energy consumption by Na/K & Ca pumps
    #
    # Units: energy[arb], power[arb/ms]
    # ---------------------------------------------------------
    "MITO": {
        "ATP0": 100.0, "Ebuf0": 80.0,
        "Pin": 10.0, "Ploss": 1.5,
        "recover_k": 8.0,          # recovery gain when ATP low
        "recover_thresh": 60.0,    # recovery threshold
        "delta_transfer": 5.0,     # minimal E_buf−ATP gap for transfer
        "ATP_clip": (1.0, 110.0),
        "Ebuf_clip": (15.0, 100.0),
        "k_transfer": 0.4,         # E_buf→ATP transfer rate
        "eta": 0.60,               # efficiency (0–1)
        "c_CO2": 0.80,             # CO₂ yield ratio
        "Heat0": 0.0, "CO2_0": 0.0
    },

    # =========================================================
    # [L2] HH Soma Layer — Membrane Potential
    # =========================================================
    # Hodgkin–Huxley Formalism
    #   C_m dV/dt = g_Na m³h(E_Na−V)
    #              + g_K n⁴(E_K−V)
    #              + g_L(E_L−V)
    #              + I_ext − I_pump(ATP)
    #
    #   dm/dt = α_m(V)(1−m) − β_m(V)m    (same for h,n)
    #   I_pump = g_pump·(1−exp[−ATP/ATP₀])·(V−E_pump)
    #
    # Meaning
    #   • Generates electrical spikes from ionic conductances.
    #   • Can include ATP-dependent Na/K pump (optional).
    #
    # Units: V[mV], t[ms], g[mS/cm²], I[µA/cm²]
    # ---------------------------------------------------------
    "HH": {
        "V0": -70.0,
        "gNa": 120.0, "gK": 36.0, "gL": 0.3,
        "ENa": 50.0, "EK": -77.0, "EL": -54.4,
        "spike_thresh": 0.0,  # spike if Vm > 0 mV
        "use_pump": False, "g_pump": 0.0,
        "E_pump": -70.0, "ATP0_ref": 100.0
    },

    # =========================================================
    # [L3] Myelinated Axon Layer — Saltatory Conduction
    # =========================================================
    # PDE Form
    #   ∂V/∂t = D(x)∂²V/∂x² − g_L(x)(V−E_L)/C_m(x)
    #            + [I_ext + I_Na_node]/C_m(x)
    #
    # Node (active): D=D_node, Cm=Cm_node, gL=gL_node
    # Internode (passive): D=D_internode≪node, Cm,gL ≪ node
    #
    # Node-specific Na current:
    #   I_Na_node = gNa_node·m³h(E_Na_node−V)
    #   ḿ = (m_inf(V)−m)/τ_m,  ḣ = (h_inf(V)−h)/τ_h
    #   m_inf=σ((V−Vh_m)/k_m),  h_inf=σ((V−Vh_h)/k_h)
    #
    # Conduction velocity estimated from node-to-node crossing.
    # Units: x[cm], t[ms], V[mV]
    # ---------------------------------------------------------
    "AXON": {
        "N": 121, "node_period": 5, "dx": 1.0e-3,
        "Vrest": -70.0, "EL": -54.4, "tau": 1.2,
        "D_node": 1.5e-3, "D_internode": 1.5e-5,
        "Cm_node": 1.0, "Cm_myelin": 0.005,
        "gL_node": 0.25, "gL_myelin": 1.0e-4,
        "node_gNa": 1200.0, "node_ENa": 50.0,
        "node_m_tau": 0.03, "node_h_tau": 0.40,
        "node_m_inf_k": 6.0, "node_m_inf_Vh": -37.0,
        "node_h_inf_k": -6.0, "node_h_inf_Vh": -58.0,
        "thresh": -50.0, "cfl_safety": 0.9,
        "coupling": 0.0, "stim_gain": 15.0,
        "c0": 1.0, "Lambda": 0.0, "gamma_decay": 0.0
    },

    # =========================================================
    # [L4] Ca²⁺ Vesicle Layer — Synaptic Release
    # =========================================================
    # Differential Equation
    #   d[Ca]/dt = Σ_k A·α(t−t_k) − k_c·ATP·([Ca]−[Ca]_0)
    #   α(t) = (e^{−t/τ_d} − e^{−t/τ_r})_+
    # Normalization:
    #   S = ([Ca]−[Ca]_0)/([Ca]_max−[Ca]_0)
    #
    # Meaning
    #   • Spike-triggered calcium influx and ATP-dependent removal.
    #   • Generates normalized signal S for plasticity and feedback.
    # ---------------------------------------------------------
    "CA": {
        "C0": 1e-4, "Cmax": 2e-3,
        "A": 1.2e-3, "tau_r": 0.01, "tau_d": 0.20,
        "k_c": 0.20,
        "max_spike_memory_ms": 2000.0, "dt_ms": 1.0
    },

    # =========================================================
    # [Integrator / Run Parameters]
    # =========================================================
    # • dt_bio : biological-scale timestep [ms]
    # • dt_elec: electrical-scale timestep (must satisfy CFL)
    # CFL Check:
    #   dt_elec ≤ 0.9·dx²/(2·max(D_node,D_internode))
    # ---------------------------------------------------------
    "RUN": {
        "T_ms": 300, "dt_bio": 1.0,
        "dt_elec": 0.01, "print_every_ms": 2,
        "color": True
    },

    # =========================================================
    # [Alpha Pulse] — Optional Stimulus Kernel
    # =========================================================
    #   Iα(t) = I₀·(e^{−t/τ_d} − e^{−t/τ_r})_+
    # ---------------------------------------------------------
    "ALPHA": {"I0": 50.0, "tau_r": 0.5, "tau_d": 3.0},

    # =========================================================
    # [Energy Ledger] — Optional Accounting
    # =========================================================
    # xi_prod : external production (+/ms)
    # chi_spike: energy cost per spike
    # zeta_leak: leakage cost coefficient
    # ---------------------------------------------------------
    "LEDGER": {"xi_prod": 0.0, "chi_spike": 0.0, "zeta_leak": 0.0},

    # =========================================================
    # [Metabolic Feedback] — Heat / CO₂ / Ca Adjustment
    # =========================================================
    # η(t+) = η₀ − β_heat·Heat
    # P_loss(t+) = P_loss₀·(1 + β_CO₂·CO₂)
    # Ca(alert) → recover_k↑ , Ca(under) → recover_k↓
    # ---------------------------------------------------------
    "FEEDBACK": {
        "beta_heat": 0.0015, "beta_co2": 0.0010,
        "lambda_ca": 0.3, "lambda_under": 0.1
    },

    # =========================================================
    # [Synaptic Resonance] — Phase–Calcium Coupling
    # =========================================================
    #   θ̇ = ω + K·sin(φ − θ)·(1 + λ·S)
    # ---------------------------------------------------------
    "RESONANCE": {"omega": 1.0, "K": 0.05, "lambda_ca": 1.0}
}

# =============================================================
# Helper Functions
# =============================================================

def get_config():
    """Return a deep-copied CONFIG to avoid in-place modification."""
    return deepcopy(CONFIG)

def cfl_dt_ax(config=None):
    """
    Compute CFL upper bound for axon timestep.
    dt_cfl = cfl * dx² / (2 * D_max)  [ms]
    """
    cfg = config or CONFIG
    ax = cfg["AXON"]
    dx = ax["dx"]
    Dmax = max(ax["D_node"], ax["D_internode"])
    return ax["cfl_safety"] * (dx**2) / (2.0 * Dmax)

def print_cfl_report(config=None):
    """
    Print a simple CFL-stability report for quick verification.
    """
    cfg = config or CONFIG
    dt_cfl = cfl_dt_ax(cfg)
    dt_elec = cfg["RUN"]["dt_elec"]
    ok = dt_elec <= dt_cfl
    status = "OK" if ok else "VIOLATION"
    print(f"[CFL] dt_elec={dt_elec:.5f} ms,  dt_ax_cfl={dt_cfl:.5f} ms  →  {status}")

# Stand-alone check
if __name__ == "__main__":
    print_cfl_report()