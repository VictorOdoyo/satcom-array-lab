from satlab import studies_array as m


def test_footprint():
    r = m.footprint()
    assert 20 < r["summary"]["peak_gain_dbi"] < 23
    assert r["arrays"]["valid"].dtype == bool
