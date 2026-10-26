from dataclasses import dataclass

import numpy as np

C = 299792458.0
K_B = 1.380649e-23


@dataclass(frozen=True)
class Earth:
    radius_m: float = 6378137.0
    flattening: float = 1 / 298.257223563
    mu_m3_s2: float = 3.986004418e14
    rotation_rad_s: float = 7.292115e-5
    j2: float = 1.08262668e-3

    def __post_init__(self):
        from .core import positive

        positive(self.radius_m)
        positive(self.mu_m3_s2)
        positive(self.rotation_rad_s, allow_zero=True)
        positive(self.j2, allow_zero=True)
        if not np.isfinite(self.flattening) or not 0 <= self.flattening < 1:
            raise ValueError("flattening in [0,1) required")

    @property
    def minor_m(self):
        return self.radius_m * (1 - self.flattening)

    @property
    def eccentricity_squared(self):
        return self.flattening * (2 - self.flattening)
