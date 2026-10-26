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
