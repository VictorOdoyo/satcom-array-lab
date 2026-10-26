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
