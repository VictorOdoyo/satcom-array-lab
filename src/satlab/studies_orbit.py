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


def precession(scale=1, seed=42):
    from . import orbit
    from .earth import Earth

    p = Earth()
    a = p.radius_m + 600000
    inc = 0.9
    dt = 15 / scale
    t = np.arange(round(18000 / dt) + 1) * dt
    state = orbit.from_elements(a, 0.01, inc)
    nodes = []
    energy = []
    hz = []
    for _ in t:
        nodes.append(orbit.elements(state)["raan_rad"])
        inv = orbit.invariants(state, include_j2=True)
        energy.append(inv["specific_energy_j_kg"])
        hz.append(inv["angular_momentum_m2_s"][2])
        state = orbit.step(state, dt, include_j2=True)
    node = np.unwrap(nodes)
    predicted = (
        -1.5 * p.j2 * np.sqrt(p.mu_m3_s2 / a**3) * (p.radius_m / (a * (1 - 0.01**2))) ** 2 * np.cos(inc)
    )
    measured = np.polyfit(t, node, 1)[0]
    er = (np.array(energy) - energy[0]) / abs(energy[0])
    hr = (np.array(hz) - hz[0]) / abs(hz[0])
    return make(
        "j2-precession",
        {"step_s": dt, "duration_s": float(t[-1])},
        {
            "measured_raan_rad_s": float(measured),
            "secular_raan_rad_s": float(predicted),
            "max_energy_relative": float(max(abs(er))),
            "max_hz_relative": float(max(abs(hr))),
        },
        {
            "time_s": t,
            "node_deg": np.rad2deg(node - node[0]),
            "reference_deg": np.rad2deg(predicted * t),
            "energy_error": er,
            "hz_error": hr,
        },
        [
            curve(
                "J2 node regression",
                "time_s",
                [("node_deg", "Osculating RK4"), ("reference_deg", "First-order secular")],
                "Time (s)",
                "RAAN change (deg)",
            ),
            curve(
                "Axisymmetric gravity conservation",
                "time_s",
                [("energy_error", "Energy"), ("hz_error", "Axial angular momentum")],
                "Time (s)",
                "Relative error",
                ylim=(-1e-7, 1e-7),
            ),
        ],
        assumptions=["First-order secular formula is approximate; osculating node includes periodic terms."],
    )


def doppler(scale=1, seed=42):
    from .earth import C
    from .passmodel import Config, geometry

    t, s = geometry(Config(duration_s=600, step_s=0.25 / scale))
    ranges = np.array([x["range_m"] for x in s[0]["true"]])
    truth = np.array([x["doppler_hz"] for x in s[0]["true"]])
    derived = -np.gradient(ranges, t, edge_order=2) * 20e9 / C
    interior = np.zeros(len(t), bool)
    interior[1:-1] = True
    error = derived - truth
    return make(
        "doppler-range-derivative",
        {"carrier_hz": 20e9},
        {"interior_max_error_hz": float(max(abs(error[interior])))},
        {
            "time_s": t,
            "analytic_hz": truth,
            "difference_hz": derived,
            "error_hz": error,
            "interior": interior,
        },
        [
            curve(
                "Geometric Doppler",
                "time_s",
                [("analytic_hz", "Velocity projection"), ("difference_hz", "Range derivative")],
                "Time (s)",
                "Frequency (Hz)",
            ),
            curve(
                "Independent derivative discrepancy",
                "time_s",
                [("error_hz", "Finite difference", "interior")],
                "Time (s)",
                "Error (Hz)",
            ),
        ],
    )
