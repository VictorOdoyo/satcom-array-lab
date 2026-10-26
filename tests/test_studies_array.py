import numpy as np

from satlab import studies_array as m


def test_footprint():
    r = m.footprint()
    assert 20 < r["summary"]["peak_gain_dbi"] < 23
    assert r["arrays"]["valid"].dtype == bool


def test_squint():
    r = m.squint()
    assert abs(r["summary"][str(16e9)] - 0.625) < 0.003
    assert abs(r["summary"][str(24e9)] - 5 / 12) < 0.003


def test_tapers():
    r = m.tapers()
    assert r["summary"]["uniform_peak_dbi"] > r["summary"]["hann_peak_dbi"]


def test_null():
    r = m.null()
    assert r["summary"]["null_sinr"] > r["summary"]["conventional_sinr"] * 10


def test_quantization():
    r = m.quantization()
    assert min(r["arrays"]["bits_8"]) > -0.001
    assert min(r["arrays"]["bits_1"]) < -2


def test_footprint_masks_exterior_not_computed_beam():
    r = m.footprint()
    a = r["arrays"]
    assert np.array_equal(a["outside"], ~a["valid"])
    assert a["outside"][0, 0]
    assert not a["outside"][50, 50]
    assert r["plots"][0]["mask"] == "outside"
