import numpy as np
import pytest

from satlab import array as m


def test_planar_aperture_has_centered_half_wavelength_physical_geometry():
    from satlab.earth import C

    p = m.Aperture()
    r = p.positions()
    assert r.shape == (64, 3) and np.allclose(r.mean(axis=0), 0)
    assert r[1, 0] - r[0, 0] == pytest.approx(C / (2 * p.design_hz))


def test_broadside_receive_manifold_has_equal_phase():
    a = m.steering([0, 0, 1])
    assert np.allclose(a, 1)
    with pytest.raises(ValueError):
        m.steering([0, 0, 2])


def test_direction_cosines_avoid_azimuth_singularity_at_zenith():
    assert np.allclose(m.from_direction_cosines([0, 0]), [0, 0, 1])
    a = m.from_direction_cosines([0.3, 0.4])
    assert np.linalg.norm(a) == pytest.approx(1)
