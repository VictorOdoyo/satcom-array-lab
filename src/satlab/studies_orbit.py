import numpy as np

from .results import curve, make


def invariants(scale=1, seed=42):
    from . import orbit
    from .earth import Earth

    p = Earth()
    a = p.radius_m + 900000
    x = orbit.from_elements(a, 0.08, 0.9, 0.3, 0.4, 0.2)
    period = 2 * np.pi * np.sqrt(a**3 / p.mu_m3_s2)
    t = np.linspace(0, period, 181 * scale)
    states = np.array([orbit.propagate(x, v) for v in t])
    inv = [orbit.invariants(v) for v in states]
    e = np.array([v["specific_energy_j_kg"] for v in inv])
    h = np.array([v["angular_momentum_m2_s"] for v in inv])
    er = (e - e[0]) / abs(e[0])
    hr = np.linalg.norm(h - h[0], axis=1) / np.linalg.norm(h[0])
    return make(
        "kepler-invariants",
        {"semimajor_m": a, "eccentricity": 0.08},
        {"max_energy_relative": float(max(abs(er))), "max_h_relative": float(max(hr))},
        {"time_s": t, "state_eci": states, "energy_error": er, "h_error": hr},
        [
            curve(
                "Two-body conservation",
                "time_s",
                [("energy_error", "Energy"), ("h_error", "Angular momentum")],
                "Time (s)",
                "Relative error",
                ylim=(-1e-10, 1e-10),
            )
        ],
    )


def convergence(scale=1, seed=42):
    from . import orbit
    from .earth import Earth

    x = orbit.from_elements(Earth().radius_m + 600000, 0.02, 0.9)
    duration = 1800.0
    reference = orbit.propagate(x, duration)
    steps = np.array([60.0, 30.0, 15.0, 7.5]) / scale
    errors = []
    for dt in steps:
        state = x.copy()
        for _ in range(round(duration / dt)):
            state = orbit.step(state, dt)
        errors.append(np.linalg.norm(state[:3] - reference[:3]))
    order = float(np.log2(errors[0] / errors[1]))
    return make(
        "rk4-convergence",
        {"duration_s": duration},
        {"observed_order": order, "finest_position_error_m": errors[-1]},
        {"step_s": steps, "error_m": np.array(errors)},
        [
            curve(
                "Independent propagation convergence",
                "step_s",
                [("error_m", "Position error")],
                "Step (s)",
                "Position error (m)",
                xscale="log",
                yscale="log",
            )
        ],
    )
