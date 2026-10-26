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
