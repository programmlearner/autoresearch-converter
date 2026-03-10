"""
Synthetic data generator for a simplified two-level three-phase VSC with LC filter.

Simulates various operating scenarios (steady-state, load step, voltage sag,
frequency deviation, harmonic injection) and saves time-domain waveforms as CSV.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pathlib import Path

# ── Circuit parameters ──────────────────────────────────────────────────────
L = 2e-3          # Filter inductance [H]
C = 50e-6         # Filter capacitance [F]
R_LOAD = 10.0     # Nominal load resistance [Ohm]
F_SW = 5000       # Switching frequency [Hz]
F_FUND = 50       # Fundamental frequency [Hz]
V_DC = 800        # DC-link voltage [V]
SAMPLE_RATE = 50000   # Output sampling rate [Hz]
SIM_DURATION = 0.1    # Duration per scenario [s] (5 fundamental cycles)
NUM_SCENARIOS = 200   # Total number of scenarios
DATA_DIR = Path.home() / ".cache" / "autoresearch-converter" / "data"


def pwm_modulation(t, v_ref_abc, f_sw=F_SW, v_dc=V_DC):
    """Determine switch states via sine-triangle PWM for three phases."""
    # Triangular carrier: period = 1/f_sw, amplitude [-1, 1]
    carrier = 2 * np.abs(2 * (t * f_sw - np.floor(t * f_sw + 0.5))) - 1
    # Modulation index: v_ref is in volts, normalize by v_dc/2
    m_abc = v_ref_abc / (v_dc / 2)
    # Switch states: +1 if ref > carrier, else -1
    s_abc = np.where(m_abc > carrier, 1.0, -1.0)
    # Converter output voltages
    v_conv_abc = s_abc * (v_dc / 2)
    return v_conv_abc


def vsc_ode(t, y, scenario_params):
    """
    State equations for three-phase VSC + LC filter.

    States: y = [i_a, i_b, i_c, v_ca, v_cb, v_cc]
      i_a/b/c  : inductor currents (converter side)
      v_ca/b/c : capacitor voltages (PCC voltages)
    """
    i_abc = y[0:3]
    v_c_abc = y[3:6]

    f_fund = scenario_params["f_fund"]
    v_mag = scenario_params["v_mag"]
    m_index = scenario_params["m_index"]
    r_load = scenario_params["r_load"]
    harmonics = scenario_params.get("harmonics", [])

    # Reference voltage for PWM (open-loop control for simplicity)
    phases = np.array([0, -2 * np.pi / 3, 2 * np.pi / 3])
    v_ref_abc = m_index * (V_DC / 2) * np.sin(2 * np.pi * f_fund * t + phases)

    # Add harmonics to reference if specified
    for h_order, h_mag, h_phase in harmonics:
        v_ref_abc += h_mag * (V_DC / 2) * np.sin(
            2 * np.pi * h_order * f_fund * t + phases + h_phase
        )

    # Converter output voltage via PWM
    v_conv_abc = pwm_modulation(t, v_ref_abc)

    # Grid-side voltage (voltage at load)
    v_grid_abc = v_mag * np.sin(2 * np.pi * f_fund * t + phases)

    # Apply voltage sag if specified
    sag = scenario_params.get("voltage_sag", None)
    if sag is not None:
        t_start, t_end, sag_depth, sag_phases = sag
        if t_start <= t <= t_end:
            for ph_idx in sag_phases:
                v_grid_abc[ph_idx] *= (1 - sag_depth)

    # Load current (resistive load on capacitor)
    i_load_abc = v_c_abc / r_load

    # di/dt = (v_conv - v_c) / L
    di_dt = (v_conv_abc - v_c_abc) / L

    # dv_c/dt = (i - i_load) / C
    dv_c_dt = (i_abc - i_load_abc) / C

    return np.concatenate([di_dt, dv_c_dt])


def simulate_scenario(scenario_params, dt_sim=1e-6):
    """Run ODE simulation for one scenario and resample to SAMPLE_RATE."""
    t_span = (0, SIM_DURATION)
    t_eval = np.arange(0, SIM_DURATION, 1.0 / SAMPLE_RATE)

    y0 = np.zeros(6)

    sol = solve_ivp(
        vsc_ode,
        t_span,
        y0,
        method="RK45",
        t_eval=t_eval,
        args=(scenario_params,),
        rtol=1e-6,
        atol=1e-8,
        max_step=1.0 / F_SW / 10,  # At least 10 steps per switching period
    )

    if sol.status != 0:
        print(f"Warning: solver did not converge: {sol.message}")

    # Extract voltages (capacitor voltage = PCC voltage) and currents
    t_out = sol.t
    i_abc = sol.y[0:3, :]   # Inductor currents
    v_abc = sol.y[3:6, :]   # Capacitor voltages

    return t_out, v_abc, i_abc


def generate_scenario_params(idx, rng):
    """Generate parameters for scenario idx."""
    params = {
        "f_fund": F_FUND,
        "v_mag": 325.0,    # 230V RMS * sqrt(2)
        "m_index": 0.85,
        "r_load": R_LOAD,
    }

    if idx < 60:
        # Steady-state at various power levels
        power_factor = rng.uniform(0.3, 1.0)
        params["r_load"] = R_LOAD / power_factor
        params["m_index"] = rng.uniform(0.6, 0.95)

    elif idx < 100:
        # Load step
        params["r_load"] = R_LOAD * rng.uniform(0.5, 2.0)
        # Load step is implicitly handled by starting from zero IC
        params["m_index"] = rng.uniform(0.7, 0.95)

    elif idx < 140:
        # Voltage sag
        sag_depth = rng.uniform(0.1, 0.9)
        t_start = rng.uniform(0.02, 0.04)
        t_end = min(t_start + rng.uniform(0.02, 0.05), SIM_DURATION - 0.005)
        if idx < 120:
            # Single-phase sag
            sag_phase = [rng.integers(0, 3)]
        else:
            # Three-phase sag
            sag_phase = [0, 1, 2]
        params["voltage_sag"] = (t_start, t_end, sag_depth, sag_phase)
        params["m_index"] = rng.uniform(0.75, 0.9)

    elif idx < 170:
        # Frequency deviation
        params["f_fund"] = F_FUND + rng.uniform(-2.0, 2.0)
        params["m_index"] = rng.uniform(0.7, 0.95)

    else:
        # Harmonic injection
        num_harmonics = rng.integers(1, 4)
        harmonics = []
        for _ in range(num_harmonics):
            h_order = rng.choice([3, 5, 7, 11, 13])
            h_mag = rng.uniform(0.01, 0.08)
            h_phase = rng.uniform(0, 2 * np.pi)
            harmonics.append((h_order, h_mag, h_phase))
        params["harmonics"] = harmonics
        params["m_index"] = rng.uniform(0.75, 0.9)

    return params


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print(f"Generating {NUM_SCENARIOS} scenarios...")
    print(f"Data directory: {DATA_DIR}")

    for idx in range(NUM_SCENARIOS):
        params = generate_scenario_params(idx, rng)
        t, v_abc, i_abc = simulate_scenario(params)

        df = pd.DataFrame({
            "time": t,
            "va": v_abc[0],
            "vb": v_abc[1],
            "vc": v_abc[2],
            "ia": i_abc[0],
            "ib": i_abc[1],
            "ic": i_abc[2],
        })

        csv_path = DATA_DIR / f"scenario_{idx:04d}.csv"
        df.to_csv(csv_path, index=False)

        if (idx + 1) % 20 == 0:
            print(f"  Generated {idx + 1}/{NUM_SCENARIOS} scenarios")

    print(f"Done. {NUM_SCENARIOS} CSV files saved to {DATA_DIR}")
    print(f"  Train: scenario_0000 to scenario_0159")
    print(f"  Val:   scenario_0160 to scenario_0199")


if __name__ == "__main__":
    main()
