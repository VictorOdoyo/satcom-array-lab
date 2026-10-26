import numpy as np

from satlab import frames as m


def test_inertial_fixed_rotations_roundtrip_and_preserve_distance():
    r = np.array([[1.0, 2.0, 3.0], [-4, 5, 6]])
    assert np.allclose(m.ecef_to_eci(m.eci_to_ecef(r, 0.7), 0.7), r)
    assert np.allclose(np.linalg.norm(m.eci_to_ecef(r, 0.7), axis=1), np.linalg.norm(r, axis=1))
