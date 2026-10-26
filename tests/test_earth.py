import numpy as np
import pytest

from satlab import earth as m


def test_reference_ellipsoid_matches_declared_defining_constants():
    p = m.Earth()
    assert p.minor_m == pytest.approx(6356752.314245, abs=1e-6)
    assert p.eccentricity_squared == pytest.approx(0.00669437999014)
    with pytest.raises(ValueError):
        m.Earth(flattening=1)


def test_geodetic_coordinates_match_equator_and_pole_reference_points():
    p = m.Earth()
    assert np.allclose(m.geodetic_to_ecef(0, 0), [p.radius_m, 0, 0])
    assert np.allclose(m.geodetic_to_ecef(np.pi / 2, 0), [0, 0, p.minor_m], atol=1e-8)
