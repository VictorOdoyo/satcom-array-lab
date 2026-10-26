from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Aperture:
    east_count: int = 8
    north_count: int = 8
    design_hz: float = 20e9
    spacing_wavelength: float = 0.5
    element_peak_dbi: float = 4.0
    element_exponent: float = 2.0

    def __post_init__(self):
        from .core import finite, integer, positive

        integer(self.east_count, 2)
        integer(self.north_count, 2)
        positive(self.design_hz)
        positive(self.spacing_wavelength)
        positive(self.element_exponent, allow_zero=True)
        finite(self.element_peak_dbi, real=True)

    @property
    def count(self):
        return self.east_count * self.north_count

    def positions(self):
        from .earth import C

        spacing = self.spacing_wavelength * C / self.design_hz
        x = (np.arange(self.east_count) - (self.east_count - 1) / 2) * spacing
        y = (np.arange(self.north_count) - (self.north_count - 1) / 2) * spacing
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(self.count)])


def steering(unit_enu, model=None, frequency_hz=None):
    from .core import finite, positive
    from .earth import C

    p = Aperture() if model is None else model
    u = finite(unit_enu, real=True)
    frequency = p.design_hz if frequency_hz is None else positive(frequency_hz)
    if u.shape[-1] != 3 or not np.allclose(np.linalg.norm(u, axis=-1), 1, atol=1e-8):
        raise ValueError("unit ENU propagation direction required")
    return np.exp(2j * np.pi * frequency / C * (u @ p.positions().T))


def from_direction_cosines(east_north):
    from .core import finite

    u = finite(east_north, real=True)
    if u.shape[-1] != 2 or np.any(np.sum(u * u, axis=-1) > 1 + 1e-12):
        raise ValueError("front-hemisphere direction cosine pair required")
    return np.concatenate([u, np.sqrt(np.maximum(0, 1 - np.sum(u * u, axis=-1)))[..., None]], axis=-1)


def normalize(weights, allow_zero=False):
    from .core import finite

    w = finite(weights, ndim=1)
    norm = np.linalg.norm(w)
    if norm < 1e-15:
        if allow_zero:
            return np.zeros_like(w)
        raise ValueError("nonzero receive aperture required")
    return w / norm


def weights(unit_enu, model=None, taper="uniform"):
    p = Aperture() if model is None else model
    if taper == "uniform":
        amplitude = np.ones(p.count)
    elif taper in ("hann", "hamming"):
        window = np.hanning if taper == "hann" else np.hamming
        amplitude = np.outer(window(p.north_count), window(p.east_count)).ravel()
    else:
        raise ValueError("known aperture taper required")
    a = steering(unit_enu, p)
    if a.ndim != 1:
        raise ValueError("one commanded beam direction required")
    return normalize(a * amplitude)
