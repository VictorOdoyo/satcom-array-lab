import numpy as np
import pytest

from satlab import array as m


def test_planar_aperture_has_centered_half_wavelength_physical_geometry():
    from satlab.earth import C

    p = m.Aperture()
    r = p.positions()
    assert r.shape == (64, 3) and np.allclose(r.mean(axis=0), 0)
    assert r[1, 0] - r[0, 0] == pytest.approx(C / (2 * p.design_hz))


def test_broadside_receive_manifold_has_equal_phase():
    a = m.steering([0, 0, 1])
    assert np.allclose(a, 1)
    with pytest.raises(ValueError):
        m.steering([0, 0, 2])


def test_direction_cosines_avoid_azimuth_singularity_at_zenith():
    assert np.allclose(m.from_direction_cosines([0, 0]), [0, 0, 1])
    a = m.from_direction_cosines([0.3, 0.4])
    assert np.linalg.norm(a) == pytest.approx(1)


def test_uniform_receiver_weights_have_unit_noise_gain_and_coherent_signal_gain():
    w = m.weights([0, 0, 1])
    a = m.steering([0, 0, 1])
    assert np.vdot(w, w).real == pytest.approx(1)
    assert abs(np.vdot(w, a)) ** 2 == pytest.approx(64)


def test_receiver_snr_gain_is_invariant_to_common_weight_scaling():
    w = m.weights([0, 0, 1])
    a = m.response(w, [0, 0, 1])
    b = m.response(3 * w, [0, 0, 1])
    assert a["snr_gain"] == pytest.approx(b["snr_gain"])
    assert b["noise_gain"] == pytest.approx(9 * a["noise_gain"])


def test_phase_quantization_preserves_amplitude_and_respects_half_step_error():
    w = m.weights(m.from_direction_cosines([0.3, 0.2]))
    q = m.quantize_phase(w, 3)
    error = np.angle(q * np.conj(w))
    assert np.allclose(abs(w), abs(q)) and max(abs(error)) <= np.pi / 8 + 1e-12


def test_element_failure_reduces_receive_gain_by_active_count():
    p = m.Aperture()
    w = m.weights([0, 0, 1])
    health = np.arange(p.count) < 32
    q = m.impair(w, healthy=health)
    assert m.response(q, [0, 0, 1])["snr_gain"] == pytest.approx(32)
    assert not m.response(m.impair(w, healthy=np.zeros(64, bool)), [0, 0, 1])["available"]
