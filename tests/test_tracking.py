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
