import numpy as np
import pytest

from satlab import orbit as m


def test_classical_elements_recover_circular_speed_and_orbit_inclination():
    from satlab.earth import Earth

    p = Earth()
    x = m.from_elements(7e6, 0, 0.7, 0.3)
    assert np.linalg.norm(x[:3]) == pytest.approx(7e6)
    assert np.linalg.norm(x[3:]) == pytest.approx(np.sqrt(p.mu_m3_s2 / 7e6))
    h = np.cross(x[:3], x[3:])
    assert np.arccos(h[2] / np.linalg.norm(h)) == pytest.approx(0.7)


def test_osculating_elements_roundtrip_non_degenerate_orbit_and_label_circular_case():
    x = m.from_elements(8e6, 0.2, 0.8, 0.3, 0.4, 0.7)
    e = m.elements(x)
    restored = m.from_elements(
        **{
            k: e[k]
            for k in (
                "semimajor_m",
                "eccentricity",
                "inclination_rad",
                "raan_rad",
                "argument_rad",
                "true_anomaly_rad",
            )
        }
    )
    assert np.allclose(restored, x, rtol=1e-12, atol=1e-6)
    assert m.elements(m.from_elements(7e6, 0, 0))["circular"]


def test_stumpff_zero_limits_and_positive_negative_identities():
    assert np.allclose(m.stumpff(0), [0.5, 1 / 6])
    c, s = m.stumpff(1.0)
    assert c == pytest.approx(1 - np.cos(1)) and s == pytest.approx(1 - np.sin(1))
    c, s = m.stumpff(-1.0)
    assert c == pytest.approx(np.cosh(1) - 1) and s == pytest.approx(np.sinh(1) - 1)


def test_universal_propagator_is_reversible_and_matches_circular_quarter_period():
    from satlab.earth import Earth

    p = Earth()
    radius = 7e6
    x = m.from_elements(radius, 0, 0)
    period = 2 * np.pi * np.sqrt(radius**3 / p.mu_m3_s2)
    y = m.propagate(x, period / 4)
    assert np.allclose(y[:3], [0, radius, 0], atol=1e-6)
    assert np.allclose(m.propagate(y, -period / 4), x, atol=1e-6)


def test_j2_acceleration_matches_independent_potential_gradient():
    r = np.array([7e6, 2e6, 1e6])
    h = 10.0
    numerical = np.array(
        [
            -(
                m.potential(r + np.eye(3)[k] * h, include_j2=True)
                - m.potential(r - np.eye(3)[k] * h, include_j2=True)
            )
            / (2 * h)
            for k in range(3)
        ]
    )
    assert np.allclose(m.acceleration(r, include_j2=True), numerical, rtol=1e-9)


def test_cartesian_rk4_matches_independent_universal_two_body_solution():
    x = m.from_elements(7e6, 0.05, 0.7, 0.3, 0.2, 0.1)
    assert np.allclose(m.step(x, 1), m.propagate(x, 1), atol=1e-6)


def test_two_body_propagation_conserves_energy_and_full_angular_momentum():
    x = m.from_elements(9e6, 0.3, 0.8, 0.2, 0.3, 0.7)
    a = m.invariants(x)
    b = m.invariants(m.propagate(x, 1200))
    assert b["specific_energy_j_kg"] == pytest.approx(a["specific_energy_j_kg"], rel=1e-12)
    assert np.allclose(a["angular_momentum_m2_s"], b["angular_momentum_m2_s"], rtol=1e-12)


def test_analytic_circular_catalog_matches_independent_universal_propagation():
    x = m.from_elements(7e6, 0, 0.7, 0.2)
    times = np.array([-300, 0, 300])
    a = m.circular_from_state(x, times)
    reference = np.array([m.propagate(x, t) for t in times])
    assert np.allclose(a, reference, atol=1e-6)


def test_overhead_pass_setup_has_declared_range_and_orbit_inclination():
    from satlab.geometry import Station, look

    s = Station()
    state = m.overhead_state(s, 0.4)
    assert look(state, s, 0.4)["range_m"] == pytest.approx(600000)
    assert m.elements(state)["inclination_rad"] == pytest.approx(np.deg2rad(53))
