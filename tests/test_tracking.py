import numpy as np
import pytest

from satlab import tracking as m


def test_bias_prediction_uses_actual_elapsed_time_without_oracle_correction():
    r = m.BiasTracker()
    r.state[:] = [10, 2]
    r.predict(0)
    assert r.predict(3)[0] == pytest.approx(16)
    assert np.linalg.eigvalsh(r.covariance).min() > 0
    with pytest.raises(ValueError):
        r.predict(2)


def test_bias_filter_recovers_frequency_offset_and_rejects_large_outlier():
    r = m.BiasTracker(initial_std=1000, rate_std=10)
    for t in range(30):
        r.predict(t)
        m.correct_bias(r, [300 + 0.2 * t], 10)
    assert abs(r.state[0] - (300 + 0.2 * 29)) < 1 and abs(r.state[1] - 0.2) < 0.1
    assert not m.correct_bias(r, [100000], 10)["accepted"]
