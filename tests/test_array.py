import numpy as np
import pytest

from satlab import array as m


def test_planar_aperture_has_centered_half_wavelength_physical_geometry():
    from satlab.earth import C

    p = m.Aperture()
    r = p.positions()
    assert r.shape == (64, 3) and np.allclose(r.mean(axis=0), 0)
    assert r[1, 0] - r[0, 0] == pytest.approx(C / (2 * p.design_hz))
