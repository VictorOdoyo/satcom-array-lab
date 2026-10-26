import numpy as np
import pytest

from satlab import modem as m


def test_qpsk_mapping():
    symbols = m.qpsk([0, 0, 0, 1, 1, 1, 1, 0])
    assert np.allclose(abs(symbols), 1)
    assert np.allclose(symbols * np.sqrt(2), [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])
    with pytest.raises(ValueError):
        m.qpsk([0, 2])
