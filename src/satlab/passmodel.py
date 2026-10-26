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
        if self.seed >= 2**32:
            raise ValueError("32-bit seed required")
        if self.beacon_gap_s:
            from .core import finite

            finite(self.beacon_gap_s, real=True)
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


def channel(look, command, config, aperture, phase_errors, healthy):
    from . import array, link

    actual = panel_rotation().T @ look["unit_enu"]
    interferer = array.from_direction_cosines([0.5, -0.2])
    if config.null_interference:
        w = array.constrained_weights(np.array([command, interferer]), [1, 0], aperture)
    else:
        w = array.weights(command, aperture)
    w = array.impair(
        array.quantize_phase(w, config.phase_bits), phase_error_rad=phase_errors, healthy=healthy
    )
    response = array.response(w, actual, aperture, config.carrier_hz)
    gain = float(response["snr_gain"] * array.element_gain(actual, aperture))
    if not look["visible"] or gain <= 0:
        return {
            "cn0_dbhz": -200.0,
            "gain_dbi": float(link.db(gain)),
            "available": False,
            "weights": w,
            "actual": actual,
        }
    atmos = link.atmosphere(look["elevation_rad"], rain_rate_mm_h=config.rain_mm_h)
    budget = link.budget(
        config.eirp_dbw,
        link.db(gain),
        look["range_m"],
        config.carrier_hz,
        config.receiver_k + atmos["sky_temperature_k"],
        atmos["attenuation_db"] + 1,
    )
    a = array.steering(interferer, aperture)
    ratio = 1 + config.interference_inr * abs(np.vdot(w, a)) ** 2 / response["noise_gain"]
    return {
        "cn0_dbhz": float(budget["cn0_dbhz"] - link.db(ratio)),
        "gain_dbi": float(link.db(gain)),
        "available": True,
        "weights": w,
        "actual": actual,
    }


def simulate(config=None):
    from . import array, link, service, tracking
    from .core import stream

    c = Config() if config is None else config
    t, satellites = geometry(c)
    p = array.Aperture(design_hz=c.carrier_hz)
    rng = stream(c.seed, "beacon")
    hardware = stream(c.seed, "hardware")
    phase = hardware.normal(0, c.phase_std_rad, p.count)
    healthy = hardware.random(p.count) >= c.failed_fraction
    receivers = [
        {
            "point": tracking.BiasTracker(2, 0.05, 0.001, 1e-8),
            "carrier": tracking.BiasTracker(1, 10000, 1000, 100),
            "lock": tracking.Lock(),
            "snr": -200.0,
            "acm": service.AdaptiveMode(),
        }
        for _ in satellites
    ]
    handover = service.Handover()
    n = len(t)
    scalar = (
        "satellite",
        "range_m",
        "elevation_deg",
        "doppler_hz",
        "cn0_dbhz",
        "esn0_db",
        "estimated_esn0_db",
        "residual_hz",
        "point_error_deg",
        "offered_bit_s",
        "delivered_bit_s",
        "mode",
        "beacon",
        "accepted",
        "locked",
        "nis",
    )
    out = {key: np.zeros(n) for key in scalar}
    out["satellite"].fill(-1)
    out["mode"].fill(-1)
    out["command_enu"] = np.zeros((n, 3))
    out["actual_enu"] = np.zeros((n, 3))
    out["weights"] = np.zeros((n, p.count), complex)
    out["measured_uv"] = np.zeros((n, 2))
    out["measured_carrier_hz"] = np.zeros(n)
    out["true_state_eci"] = np.zeros((n, 6))
    out["nominal_state_eci"] = np.zeros((n, 6))
    out["commanded_carrier_hz"] = np.zeros(n)
    outlier_used = False
    for k, time in enumerate(t):
        predictions = []
        for obs in receivers:
            predictions.append((obs["point"].predict(time), obs["carrier"].predict(time)[0]))
        scores = {}
        for j, sat in enumerate(satellites):
            score = forecast(sat["nominal"][k], c, p)
            if score is not None and score >= 60:
                scores[str(j)] = score
        selected = handover.update(time, scores)
        if selected is None:
            for obs in receivers:
                obs["lock"].update(time, False)
            continue
        j = int(selected)
        obs = receivers[j]
        real = satellites[j]["true"][k]
        nominal = satellites[j]["nominal"][k]
        pb, fb = predictions[j]
        command = command_direction(nominal["unit_enu"], pb if c.closed_loop else np.zeros(2))
        physical = channel(real, command, c, p, phase, healthy)
        true_cfo = real["doppler_hz"] + 300 + 0.2 * (time - t[0])
        commanded_cfo = nominal["doppler_hz"] + (fb if c.closed_loop else 0)
        residual = true_cfo - commanded_cfo
        esn0 = (
            physical["cn0_dbhz"]
            - 10 * np.log10(c.symbol_rate_s)
            + float(link.db(link.coherent_gain(residual, 0.001)))
        )
        locked = obs["lock"].available_at(time)
        mode = obs["acm"].update(time, obs["snr"], available=locked)
        offered = 0 if mode is None else service.information_rate(mode, c.symbol_rate_s)
        delivered = (
            offered if mode is not None and physical["available"] and esn0 >= mode.threshold_esn0_db else 0
        )
        old_snr = obs["snr"]
        gap = bool(c.beacon_gap_s and c.beacon_gap_s[0] <= time < c.beacon_gap_s[1])
        beacon = bool(
            physical["available"] and physical["cn0_dbhz"] >= 60 and abs(residual) < 6000 and not gap
        )
        accepted = False
        nis = 0
        if beacon:
            factor = float(np.clip(10 ** ((70 - physical["cn0_dbhz"]) / 20), 0.3, 3))
            uv = physical["actual"][:2] + rng.normal(0, 0.001 * factor, 2)
            measured_cfo = true_cfo + rng.normal(0, 20 * factor)
            if c.inject_outlier and not outlier_used and time >= 0:
                uv += np.array([0.2, -0.2])
                measured_cfo += 2000
                outlier_used = True
            out["measured_uv"][k] = uv
            out["measured_carrier_hz"][k] = measured_cfo
            pr = tracking.correct_bias(obs["point"], uv - nominal["unit_enu"][:2], 0.001 * factor)
            fr = tracking.correct_bias(obs["carrier"], measured_cfo - nominal["doppler_hz"], 20 * factor)
            accepted = bool(pr["accepted"] and fr["accepted"])
            nis = max(pr["nis"], fr["nis"])
            if accepted:
                obs["snr"] = esn0 + rng.normal(0, 0.3)
        obs["lock"].update(time, accepted)
        for other in range(len(receivers)):
            if other != j:
                receivers[other]["lock"].update(time, False)
        values = [
            j,
            real["range_m"],
            np.rad2deg(real["elevation_rad"]),
            real["doppler_hz"],
            physical["cn0_dbhz"],
            esn0,
            old_snr,
            residual,
            np.rad2deg(np.arccos(np.clip(command @ physical["actual"], -1, 1))),
            offered,
            delivered,
            -1 if mode is None else service.MODES.index(mode),
            int(beacon),
            int(accepted),
            int(locked),
            nis,
        ]
        for key, value in zip(scalar, values, strict=True):
            out[key][k] = value
        out["command_enu"][k] = command
        out["actual_enu"][k] = physical["actual"]
        out["weights"][k] = physical["weights"]
        out["true_state_eci"][k] = satellites[j]["true_state"][k]
        out["nominal_state_eci"][k] = satellites[j]["nominal_state"][k]
        out["commanded_carrier_hz"][k] = commanded_cfo
    out.update(service.accumulate(t, out["offered_bit_s"], out["delivered_bit_s"]))
    valid = out["satellite"] >= 0
    return {
        "time_s": t,
        **out,
        "selected": valid,
        "handovers": handover.handovers,
        "configuration": c.to_dict(),
    }
