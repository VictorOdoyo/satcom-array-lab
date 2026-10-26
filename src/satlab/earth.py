from dataclasses import dataclass
from datetime import UTC

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


def ecef_to_geodetic(position, model=None):
    from .core import finite

    p = Earth() if model is None else model
    r = finite(position, real=True)
    if r.shape != (3,) or np.linalg.norm(r) < 1:
        raise ValueError("noncentral Cartesian Earth position required")
    x, y, z = r
    horizontal = np.hypot(x, y)
    longitude = np.arctan2(y, x)
    if horizontal < 1e-8:
        return np.array([np.sign(z) * np.pi / 2, longitude, abs(z) - p.minor_m])
    latitude = np.arctan2(z, horizontal * (1 - p.eccentricity_squared))
    for _ in range(15):
        n = p.radius_m / np.sqrt(1 - p.eccentricity_squared * np.sin(latitude) ** 2)
        height = horizontal / np.cos(latitude) - n
        new = np.arctan2(z, horizontal * (1 - p.eccentricity_squared * n / (n + height)))
        if abs(new - latitude) < 1e-14:
            latitude = new
            break
        latitude = new
    n = p.radius_m / np.sqrt(1 - p.eccentricity_squared * np.sin(latitude) ** 2)
    height = horizontal / np.cos(latitude) - n
    return np.array([latitude, longitude, height])


def enu_matrix(latitude_rad, longitude_rad):
    from .core import finite

    lat, lon = finite([latitude_rad, longitude_rad], real=True)
    if abs(lat) > np.pi / 2:
        raise ValueError("geodetic station latitude required")
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    return np.array([[-sl, cl, 0], [-sp * cl, -sp * sl, cp], [cp * cl, cp * sl, sp]])


def julian_date(epoch):
    from datetime import datetime

    value = datetime.fromisoformat(epoch.replace("Z", "+00:00")) if isinstance(epoch, str) else epoch
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timezone-aware civil epoch required")
    return value.astimezone(UTC).timestamp() / 86400 + 2440587.5


def gmst(julian_ut1, julian_tt=None):
    from .core import finite

    jd = finite(julian_ut1, real=True)
    tt = jd if julian_tt is None else finite(julian_tt, real=True)
    midnight = np.floor(jd - 0.5) + 0.5
    hours = 24 * (jd - midnight)
    dut = midnight - 2451545.0
    centuries = (tt - 2451545.0) / 36525
    if julian_tt is None:
        mean = 6.697375 + 0.065709824279 * dut + 1.0027379 * hours + 0.0000258 * centuries**2
    else:
        mean = (
            6.697375
            + 0.065707485828 * dut
            + 1.0027379 * hours
            + 0.0854103 * centuries
            + 0.0000258 * centuries**2
        )
    return np.mod(mean, 24) * np.pi / 12
