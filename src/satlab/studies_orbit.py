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
