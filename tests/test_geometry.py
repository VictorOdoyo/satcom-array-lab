import numpy as np

from satlab import geometry as m


def test_default_station_is_synthetic_geodetic_reference_not_an_implicit_survey():
    s = m.Station()
    assert s.height_m == 0
    assert np.allclose(s.enu() @ s.enu().T, np.eye(3))
