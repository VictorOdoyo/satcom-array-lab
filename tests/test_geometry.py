import numpy as np
import pytest

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


def test_overhead_look_has_valid_elevation_without_fictitious_azimuth_observability():
    from satlab.earth import C
    from satlab.frames import state_to_eci

    s = m.Station(latitude_rad=0, longitude_rad=0)
    r = s.ecef()
    fixed = np.r_[r + np.array([600000, 0, 0]), [100.0, 0, 0]]
    a = m.look(state_to_eci(fixed, 0), s, 0)
    assert a["range_m"] == pytest.approx(600000) and a["elevation_rad"] == pytest.approx(np.pi / 2)
    assert not a["azimuth_valid"] and a["visible"]
    assert a["doppler_hz"] == pytest.approx(-100 * 20e9 / C)
