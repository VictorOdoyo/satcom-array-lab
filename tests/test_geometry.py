import numpy as np

from satlab import geometry as m


def test_default_station_is_synthetic_geodetic_reference_not_an_implicit_survey():
    s = m.Station()
    assert s.height_m == 0
    assert np.allclose(s.enu() @ s.enu().T, np.eye(3))


def test_ellipsoid_occultation_distinguishes_overhead_and_through_earth_paths():
    from satlab.earth import Earth

    r = m.Station(latitude_rad=0, longitude_rad=0).ecef()
    p = Earth()
    assert not m.occulted(r, [p.radius_m + 600000, 0, 0])
    assert m.occulted(r, [-p.radius_m - 600000, 0, 0])
