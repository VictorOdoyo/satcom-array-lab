import pytest

from satlab import earth as m


def test_reference_ellipsoid_matches_declared_defining_constants():
    p = m.Earth()
    assert p.minor_m == pytest.approx(6356752.314245, abs=1e-6)
    assert p.eccentricity_squared == pytest.approx(0.00669437999014)
    with pytest.raises(ValueError):
        m.Earth(flattening=1)
