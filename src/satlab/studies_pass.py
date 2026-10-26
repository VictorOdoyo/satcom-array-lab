import numpy as np

from .passmodel import Config, simulate
from .results import curve, make


def solve(name, scale, seed, **changes):
    c = Config(step_s=1 / scale, seed=seed, **changes)
    r = simulate(c)
    arrays = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
    valid = r["selected"]
    locked = valid & r["locked"].astype(bool)
    summary = {
        "delivered_mbit": r["delivered_bits"][-1] / 1e6,
        "offered_mbit": r["offered_bits"][-1] / 1e6,
        "handovers": r["handovers"],
        "locked_samples": int(locked.sum()),
        "locked_carrier_rms_hz": float(np.sqrt(np.mean(r["residual_hz"][locked] ** 2)))
        if locked.any()
        else 0,
        "locked_point_rms_deg": float(np.sqrt(np.mean(r["point_error_deg"][locked] ** 2)))
        if locked.any()
        else 0,
    }
    plots = [
        curve(
            "Selected physical pass",
            "time_s",
            [("elevation_deg", "Elevation", "selected")],
            "Time from epoch (s)",
            "Elevation (deg)",
        ),
        curve(
            "Carrier correction residual",
            "time_s",
            [("residual_hz", "True minus commanded", "selected")],
            "Time (s)",
            "Frequency (Hz)",
        ),
        curve(
            "Measured receiver and actual channel",
            "time_s",
            [
                ("esn0_db", "Actual Es/N0", "selected"),
                ("estimated_esn0_db", "Previous pilot estimate", "selected"),
            ],
            "Time (s)",
            "Es/N0 (dB)",
            ylim=(-30, 25),
        ),
        curve(
            "Cumulative channel-delivered service",
            "time_s",
            [("offered_bits", "Offered"), ("delivered_bits", "Delivered proxy")],
            "Time (s)",
            "Information (bit)",
        ),
        curve(
            "Panel pointing error",
            "time_s",
            [("point_error_deg", "Physical error", "selected")],
            "Time (s)",
            "Angle (deg)",
        ),
        curve(
            "Causal satellite selection",
            "time_s",
            [("satellite", "Selected index", "selected")],
            "Time (s)",
            "Satellite index",
            style=".",
        ),
    ]
    return make(
        name,
        c.to_dict(),
        summary,
        arrays,
        plots,
        assumptions=[
            "Synthetic circular ephemerides, UTC approximates UT1, fixed Earth rotation after GMST epoch.",
            "Station uses Nairobi coordinates at zero ellipsoidal height; no surveyed terminal or live orbit data.",
            "Idealized noisy beacon observations update bias filters only after current service decisions.",
            "Illustrative ACM thresholds and rain law; no FEC decoder or ITU availability claim.",
        ],
    )


def nominal(scale=1, seed=42):
    return solve("closed-loop-pass", scale, seed)


def open_loop(scale=1, seed=42):
    return solve("ephemeris-only-pass", scale, seed, closed_loop=False)


def rain(scale=1, seed=42):
    return solve("rain-fade-pass", scale, seed, rain_mm_h=20)


def one_bit(scale=1, seed=42):
    return solve("one-bit-phase-pass", scale, seed, phase_bits=1)


def calibration(scale=1, seed=42):
    return solve("calibration-error-pass", scale, seed, phase_std_rad=0.8)


def failures(scale=1, seed=42):
    return solve("element-failure-pass", scale, seed, failed_fraction=0.5)
