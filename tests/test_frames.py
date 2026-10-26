import numpy as np

from satlab import frames as m


def test_inertial_fixed_rotations_roundtrip_and_preserve_distance():
    r = np.array([[1.0, 2.0, 3.0], [-4, 5, 6]])
    assert np.allclose(m.ecef_to_eci(m.eci_to_ecef(r, 0.7), 0.7), r)
    assert np.allclose(np.linalg.norm(m.eci_to_ecef(r, 0.7), axis=1), np.linalg.norm(r, axis=1))


def test_stationary_earth_point_has_inertial_rotation_velocity_and_zero_fixed_velocity():
    p = np.array([6378137.0, 0, 0, 0, 0, 0])
    eci = m.state_to_eci(p, 0.7)
    assert np.linalg.norm(eci[3:]) > 400
    assert np.allclose(m.state_to_ecef(eci, 0.7), p, atol=1e-9)


def test_rtn_basis_matches_circular_equatorial_orbit():
    matrix = m.rtn_matrix([7e6, 0, 0, 0, 7500, 0])
    assert np.allclose(matrix, np.eye(3))
