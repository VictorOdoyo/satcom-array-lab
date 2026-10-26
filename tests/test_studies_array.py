from satlab import studies_array as m


def test_footprint():
    r = m.footprint()
    assert 20 < r["summary"]["peak_gain_dbi"] < 23
    assert r["arrays"]["valid"].dtype == bool


def test_squint():
    r = m.squint()
    assert abs(r["summary"][str(16e9)] - 0.625) < 0.003
    assert abs(r["summary"][str(24e9)] - 5 / 12) < 0.003
