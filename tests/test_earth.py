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


def test_geodetic_inverse_recovers_arbitrary_altitude_and_polar_height():
    target = np.array([-0.4, 1.2, 600000.0])
    a = m.ecef_to_geodetic(m.geodetic_to_ecef(*target))
    assert np.allclose(a[:2], target[:2], atol=1e-12) and a[2] == pytest.approx(target[2], abs=1e-6)
    p = m.Earth()
    assert m.ecef_to_geodetic([0, 0, p.minor_m + 100])[2] == pytest.approx(100)


def test_enu_basis_is_orthonormal_right_handed_and_matches_equator():
    r = m.enu_matrix(0.4, 0.7)
    assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1)
    assert np.allclose(m.enu_matrix(0, 0), [[0, 1, 0], [0, 0, 1], [1, 0, 0]])


def test_julian_date_matches_j2000_and_rejects_implicit_host_timezone():
    assert m.julian_date("2000-01-01T12:00:00Z") == 2451545.0
    with pytest.raises(ValueError):
        m.julian_date("2000-01-01T12:00:00")
