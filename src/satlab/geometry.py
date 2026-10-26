from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Station:
    latitude_rad: float = np.deg2rad(-1.2864)
    longitude_rad: float = np.deg2rad(36.8172)
    height_m: float = 0.0
    minimum_elevation_rad: float = np.deg2rad(10.0)

    def __post_init__(self):
        from .core import finite

        finite([self.latitude_rad, self.longitude_rad, self.height_m, self.minimum_elevation_rad], real=True)
        if abs(self.latitude_rad) > np.pi / 2 or not 0 <= self.minimum_elevation_rad < np.pi / 2:
            raise ValueError("geodetic station and interior elevation mask required")

    def ecef(self, model=None):
        from .earth import geodetic_to_ecef

        return geodetic_to_ecef(self.latitude_rad, self.longitude_rad, self.height_m, model)

    def enu(self):
        from .earth import enu_matrix

        return enu_matrix(self.latitude_rad, self.longitude_rad)
