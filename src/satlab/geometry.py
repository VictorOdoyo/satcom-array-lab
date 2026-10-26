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


def occulted(start_ecef, end_ecef, model=None):
    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    a = finite(start_ecef, real=True)
    b = finite(end_ecef, real=True)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Cartesian segment endpoints required")
    axes = np.array([p.radius_m, p.radius_m, p.minor_m])
    origin = a / axes
    delta = (b - a) / axes
    norm = delta @ delta
    if norm < 1e-30:
        raise ValueError("nonzero propagation segment required")
    time = np.clip(-(origin @ delta) / norm, 0, 1)
    closest = origin + time * delta
    return bool(closest @ closest < 1 - 1e-12)


def look(state_eci, station, angle_rad, carrier_hz=20e9, model=None):
    from .core import positive
    from .earth import C
    from .frames import state_to_ecef

    fixed = state_to_ecef(state_eci, angle_rad, model)
    delta = fixed[:3] - station.ecef(model)
    distance = np.linalg.norm(delta)
    if distance < 1:
        raise ValueError("nonzero satellite-station separation required")
    line = station.enu() @ delta
    unit = line / distance
    horizontal = np.hypot(unit[0], unit[1])
    azimuth = np.arctan2(unit[0], unit[1]) % (2 * np.pi) if horizontal > 1e-12 else 0.0
    elevation = float(np.arcsin(np.clip(unit[2], -1, 1)))
    rate = float(delta @ fixed[3:] / distance)
    blocked = occulted(station.ecef(model), fixed[:3], model)
    return {
        "range_m": float(distance),
        "range_rate_m_s": rate,
        "delay_s": float(distance / C),
        "doppler_hz": float(-rate * positive(carrier_hz) / C),
        "azimuth_rad": float(azimuth),
        "elevation_rad": elevation,
        "azimuth_valid": bool(horizontal > 1e-12),
        "unit_enu": unit,
        "visible": bool(not blocked and elevation >= station.minimum_elevation_rad),
        "occulted": blocked,
    }
