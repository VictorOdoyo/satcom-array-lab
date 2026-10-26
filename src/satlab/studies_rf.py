import numpy as np

from .results import curve, make


def ber(scale=1, seed=42):
    from . import link, modem
    from .core import stream

    rng = stream(seed, "ber")
    snr = np.arange(0.0, 13.0, 2.0)
    trials = 200000 * scale
    bits = rng.integers(0, 2, trials)
    s = modem.qpsk(bits)
    measured = []
    lower = []
    upper = []
    rows = []
    for value in snr:
        rx = modem.awgn(s, value, rng)
        errors = int(np.count_nonzero(modem.detect(rx) != bits))
        lo, hi = modem.wilson(errors, trials)
        measured.append(errors / trials)
        lower.append(lo)
        upper.append(hi)
        rows.append({"esn0_db": value, "errors": errors, "bits": trials, "lower_95": lo, "upper_95": hi})
    theoretical = link.qpsk_ber(snr)
    valid = np.array(measured) > 0
    return make(
        "qpsk-ber",
        {"bits_per_point": trials, "seed": seed},
        {"max_absolute_error": float(max(abs(np.array(measured) - theoretical)))},
        {
            "esn0_db": snr,
            "measured": np.array(measured),
            "theory": theoretical,
            "lower": np.maximum(lower, 1e-12),
            "upper": np.array(upper),
            "nonzero": valid,
        },
        [
            curve(
                "Uncoded QPSK in complex AWGN",
                "esn0_db",
                [
                    ("measured", "Measured", "nonzero"),
                    ("theory", "Analytic"),
                    ("lower", "Wilson lower"),
                    ("upper", "Wilson upper"),
                ],
                "Es/N0 (dB)",
                "Bit-error probability",
                yscale="log",
            )
        ],
        {"confidence": rows},
        ["No channel coding; independent bits and unit-energy complex symbols."],
    )


def burst(scale=1, seed=42):
    from . import modem

    result = modem.burst(15, 1234.5, seed, 2048 * scale)
    rx = result.pop("received")
    ref = result.pop("reference")
    n = min(2000, len(rx))
    plots = []
    for label, z in (("Before equalization reference", ref), ("Recovered QPSK", rx)):
        plots.append(
            curve(
                label,
                "ref_i" if label.startswith("Before") else "rx_i",
                [("ref_q" if label.startswith("Before") else "rx_q", "Symbols")],
                "In-phase",
                "Quadrature",
                style=".",
                ylim=(-1.2, 1.2),
            )
        )
        plots[-1]["aspect"] = "equal"
    return make(
        "burst-acquisition",
        {"esn0_db": 15, "offset_hz": 1234.5, "sample_rate_hz": 20000, "seed": seed},
        result,
        {"ref_i": ref[:n].real, "ref_q": ref[:n].imag, "rx_i": rx[:n].real, "rx_q": rx[:n].imag},
        plots,
        assumptions=[
            "A separate unmodulated tone preamble estimates CFO; known pilots fit one complex gain."
        ],
    )


def coherent(scale=1, seed=42):
    from . import link

    offset = np.linspace(-3000, 3000, 601 * scale)
    duration = 0.001
    n = 1000
    sampled = abs(np.mean(np.exp(2j * np.pi * offset[:, None] * np.arange(n) * duration / n), axis=1)) ** 2
    analytic = link.coherent_gain(offset, duration)
    return make(
        "coherent-frequency-loss",
        {"integration_s": duration, "samples": n},
        {"max_power_difference": float(max(abs(sampled - analytic)))},
        {"offset_hz": offset, "analytic_db": link.db(analytic), "discrete_db": link.db(sampled)},
        [
            curve(
                "Residual carrier coherent loss",
                "offset_hz",
                [("analytic_db", "Continuous sinc"), ("discrete_db", "Discrete tone sum")],
                "Residual frequency (Hz)",
                "Coherent power gain (dB)",
                ylim=(-70, 1),
            )
        ],
    )


def atmosphere(scale=1, seed=42):
    from . import link

    el = np.deg2rad(np.linspace(10, 90, 161 * scale))
    clear = link.atmosphere(el)
    rain = link.atmosphere(el, rain_rate_mm_h=20)
    return make(
        "atmosphere-noise",
        {"rain_rate_mm_h": 20, "receiver_k": 300},
        {"zenith_rain_attenuation_db": float(rain["attenuation_db"][-1])},
        {
            "elevation_deg": np.rad2deg(el),
            "clear_db": clear["attenuation_db"],
            "rain_db": rain["attenuation_db"],
            "clear_k": clear["sky_temperature_k"],
            "rain_k": rain["sky_temperature_k"],
        },
        [
            curve(
                "Illustrative slant loss",
                "elevation_deg",
                [("clear_db", "Clear"), ("rain_db", "Rain")],
                "Elevation (deg)",
                "Attenuation (dB)",
            ),
            curve(
                "Atmospheric noise brightness",
                "elevation_deg",
                [("clear_k", "Clear"), ("rain_k", "Rain")],
                "Elevation (deg)",
                "Sky temperature (K)",
            ),
        ],
        assumptions=["Illustrative power-law rain layer, not a complete ITU-R P.618 availability model."],
    )


def budget(scale=1, seed=42):
    from . import link

    ranges = np.linspace(600e3, 2500e3, 101 * scale)
    arrays = {"range_km": ranges / 1000}
    for target in (3, 7.5, 11):
        arrays["target_" + str(target).replace(".", "_")] = link.required_eirp(
            target, 2e6, 22, ranges, 20e9, 350, 2
        )
    return make(
        "link-budget-tradeoff",
        {"symbol_rate_s": 2e6, "receive_gain_dbi": 22, "temperature_k": 350, "loss_db": 2},
        {"near_range_eirp_dbw": float(arrays["target_3"][0])},
        arrays,
        [
            curve(
                "Inverse physical link budget",
                "range_km",
                [(k, k.replace("target_", "") + " dB Es/N0") for k in arrays if k != "range_km"],
                "Slant range (km)",
                "Required EIRP (dBW)",
            )
        ],
    )
