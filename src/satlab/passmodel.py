from dataclasses import asdict, dataclass

import numpy as np

from .core import integer, positive


@dataclass(frozen=True)
class Config:
    duration_s: float = 900
    step_s: float = 1
    epoch: str = "2025-01-01T00:00:00Z"
    altitude_m: float = 600000
    ephemeris_altitude_bias_m: float = 10000
    carrier_hz: float = 20e9
    symbol_rate_s: float = 2e6
    eirp_dbw: float = 30
    receiver_k: float = 300
    rain_mm_h: float = 0
    phase_bits: int = 8
    phase_std_rad: float = 0
    failed_fraction: float = 0
    closed_loop: bool = True
    pass_centers_s: tuple = (0.0,)
    beacon_gap_s: tuple = ()
    inject_outlier: bool = False
    interference_inr: float = 0
    null_interference: bool = False
    seed: int = 0

    def __post_init__(self):
        from .earth import julian_date

        for key in ("duration_s", "step_s", "altitude_m", "carrier_hz", "symbol_rate_s", "receiver_k"):
            positive(getattr(self, key))
        for key in ("rain_mm_h", "phase_std_rad", "interference_inr"):
            positive(getattr(self, key), allow_zero=True)
        integer(self.phase_bits)
        integer(self.seed, 0)
        if self.phase_bits > 16 or not 0 <= self.failed_fraction <= 1 or not self.pass_centers_s:
            raise ValueError("bounded phase precision and failure fraction required")
        if self.altitude_m + self.ephemeris_altitude_bias_m <= 0:
            raise ValueError("positive nominal altitude required")
        if (
            type(self.closed_loop) is not bool
            or type(self.null_interference) is not bool
            or type(self.inject_outlier) is not bool
        ):
            raise ValueError("boolean receiver switches required")
        from .core import finite

        finite([self.eirp_dbw, self.ephemeris_altitude_bias_m, *self.pass_centers_s], real=True)
        if self.beacon_gap_s and (
            len(self.beacon_gap_s) != 2 or self.beacon_gap_s[1] <= self.beacon_gap_s[0]
        ):
            raise ValueError("ordered beacon gap required")
        julian_date(self.epoch)

    def to_dict(self):
        return asdict(self)


def geometry(config):
    from . import earth, orbit
    from . import geometry as geo
    from .core import timeline

    station = geo.Station()
    t = timeline(config.duration_s, config.step_s) - config.duration_s / 2
    theta = earth.gmst(earth.julian_date(config.epoch)) + earth.Earth().rotation_rad_s * t
    satellites = []
    for center in config.pass_centers_s:
        anchor = theta[0] + earth.Earth().rotation_rad_s * (center - t[0])
        true = orbit.overhead_state(station, anchor, config.altitude_m)
        nominal = orbit.overhead_state(station, anchor, config.altitude_m + config.ephemeris_altitude_bias_m)
        ts = orbit.circular_from_state(true, t - center)
        ns = orbit.circular_from_state(nominal, t - center)
        satellites.append(
            {
                "true_state": ts,
                "nominal_state": ns,
                "true": [geo.look(s, station, a, config.carrier_hz) for s, a in zip(ts, theta, strict=True)],
                "nominal": [
                    geo.look(s, station, a, config.carrier_hz) for s, a in zip(ns, theta, strict=True)
                ],
            }
        )
    return t, satellites


def panel_rotation(roll_rad=-0.03, pitch_rad=0.05, yaw_rad=0):
    from .frames import rz

    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    return (
        rz(yaw_rad)
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def command_direction(nominal, estimated_uv):
    from .array import from_direction_cosines

    uv = np.asarray(nominal)[:2] + np.asarray(estimated_uv)
    norm = np.linalg.norm(uv)
    if norm >= 0.999999:
        uv = uv * 0.999999 / norm
    return from_direction_cosines(uv)


def forecast(look, config, aperture):
    from . import array, link

    if not look["visible"]:
        return None
    a = link.atmosphere(look["elevation_rad"], rain_rate_mm_h=config.rain_mm_h)
    gain = aperture.count * array.element_gain(look["unit_enu"], aperture)
    return float(
        link.budget(
            config.eirp_dbw,
            link.db(gain),
            look["range_m"],
            config.carrier_hz,
            config.receiver_k + a["sky_temperature_k"],
            a["attenuation_db"] + 1,
        )["cn0_dbhz"]
    )
