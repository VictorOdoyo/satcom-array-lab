import numpy as np

from satlab import link as m


def test_power_db_conversion_uses_ten_log_and_declared_zero_floor():
    assert np.allclose(m.db([0, 0.01, 1, 100]), [-200, -20, 0, 20])
    assert np.allclose(m.linear([-20, 0, 20]), [0.01, 1, 100])
