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


def geodetic_to_ecef(latitude_rad, longitude_rad, height_m=0, model=None):
    from .core import finite

    p = Earth() if model is None else model
    lat = finite(latitude_rad, real=True)
    lon = finite(longitude_rad, real=True)
    height = finite(height_m, real=True)
    if np.any(abs(lat) > np.pi / 2):
        raise ValueError("geodetic latitude inside poles required")
    lat, lon, height = np.broadcast_arrays(lat, lon, height)
    n = p.radius_m / np.sqrt(1 - p.eccentricity_squared * np.sin(lat) ** 2)
    return np.stack(
        (
            (n + height) * np.cos(lat) * np.cos(lon),
            (n + height) * np.cos(lat) * np.sin(lon),
            (n * (1 - p.eccentricity_squared) + height) * np.sin(lat),
        ),
        axis=-1,
    )
