import numpy as np

from satlab import studies_rf as m


def test_ber():
    r = m.ber()
    assert r["summary"]["max_absolute_error"] < 0.003


def test_burst():
    r = m.burst()
    assert r["summary"]["ber"] < 0.01


def test_coherent():
    r = m.coherent()
    assert r["summary"]["max_power_difference"] < 1e-5


def test_atmosphere():
    r = m.atmosphere()
    assert np.all(r["arrays"]["rain_k"] > r["arrays"]["clear_k"])
    assert np.all(r["arrays"]["rain_k"] < 280)
