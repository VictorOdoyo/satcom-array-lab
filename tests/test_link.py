import numpy as np
import pytest

from satlab import link as m


def test_power_db_conversion_uses_ten_log_and_declared_zero_floor():
    assert np.allclose(m.db([0, 0.01, 1, 100]), [-200, -20, 0, 20])
    assert np.allclose(m.linear([-20, 0, 20]), [0.01, 1, 100])


def test_free_space_loss_matches_independent_kilometer_megahertz_reference():
    value = m.free_space_loss(1000000, 20e9)
    expected = 32.4477832219 + 20 * np.log10(1000) + 20 * np.log10(20000)
    assert value == pytest.approx(expected, abs=1e-9)
    assert m.free_space_loss(2000000, 20e9) - value == pytest.approx(20 * np.log10(2))
