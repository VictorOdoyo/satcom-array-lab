import numpy as np

from .results import curve, make


def solve(name, scale, seed, **changes):
    from .link import db
    from .waveform import burst, periodogram

    r = burst(seed=seed, **changes)
    rx = r["received_symbols"]
    reference = r["reference_symbols"]
    summary = {
        key: r[key]
        for key in (
            "ber",
            "bit_errors",
            "bits",
            "evm_rms",
            "estimated_offset_hz",
            "estimated_lag_samples",
            "clipped_samples",
        )
    }
    spectrum = periodogram(r["receiver_waveform"], r["sample_rate_hz"])
    n = min(len(rx), 512)
    filtered = r["matched_waveform"]
    start = int(r["estimated_lag_samples"])
    segment = filtered[start : start + 320]
    arrays = {
        "rx_i": rx[:n].real,
        "rx_q": rx[:n].imag,
        "reference_i": reference[:n].real,
        "reference_q": reference[:n].imag,
        "timing_lags": r["timing_lags"],
        "timing_score": r["timing_score"],
        "wave_sample": np.arange(len(segment)),
        "matched_i": segment.real,
        "matched_q": segment.imag,
        "frequency_khz": spectrum["frequency_hz"] / 1000,
        "power_db_hz": db(spectrum["power_per_hz"], -160),
        "transmit_waveform": r["transmit_waveform"],
        "receiver_waveform": r["receiver_waveform"],
        "recovered_symbols": rx,
        "reference_symbols": reference,
    }
    plots = [
        curve(
            "Recovered uncoded QPSK payload",
            "rx_i",
            [("rx_q", "Payload symbols")],
            "In-phase",
            "Quadrature",
            style=".",
            ylim=(-1.8, 1.8),
        ),
        curve(
            "Known payload used only for scoring",
            "reference_i",
            [("reference_q", "Reference symbols")],
            "In-phase",
            "Quadrature",
            style=".",
            ylim=(-1.8, 1.8),
        ),
        curve(
            "Pilot-only timing acquisition",
            "timing_lags",
            [("timing_score", "Normalized correlation")],
            "Pilot lag (sample)",
            "Correlation power ratio",
        ),
        curve(
            "Matched receiver waveform",
            "wave_sample",
            [("matched_i", "In-phase"), ("matched_q", "Quadrature")],
            "Sample from acquired pilot",
            "Amplitude",
        ),
        curve(
            "Sample-domain receiver spectrum",
            "frequency_khz",
            [("power_db_hz", "Periodogram")],
            "Frequency (kHz)",
            "Power density (dB/Hz)",
        ),
    ]
    for plot in plots[:2]:
        plot["aspect"] = "equal"
    return make(
        name,
        {
            "samples_per_symbol": 8,
            "symbol_rate_s": 10000,
            "rrc_rolloff": 0.25,
            "pilot_symbols": 128,
            "seed": seed,
            **changes,
        },
        summary,
        arrays,
        plots,
        assumptions=[
            "Standalone oversampled QPSK waveform; no FEC or DVB-S2 conformance claim.",
            "Receiver uses only the known pilot prefix for timing, carrier and widely-linear IQ/DC calibration.",
            "Payload references are used for BER/EVM scoring only. Fixed burst calibration does not track later clock or phase drift.",
        ],
    )


def nominal(scale=1, seed=42):
    return solve("oversampled-qpsk-receiver", scale, seed)


def uncalibrated(scale=1, seed=42):
    return solve("uncalibrated-waveform", scale, seed, calibrated=False)


def timing(scale=1, seed=42):
    return solve("fractional-timing-acquisition", scale, seed, fractional_samples=3.5)


def iq(scale=1, seed=42):
    return solve("severe-iq-calibration", scale, seed, quadrature_gain=0.4, iq_phase_rad=0.3, offset_hz=0)


def clipping(scale=1, seed=42):
    return solve("receiver-adc-clipping", scale, seed, adc_bits=5, adc_rail=0.12)


def amplifier(scale=1, seed=42):
    return solve("amplifier-waveform-compression", scale, seed, amplifier_saturation=0.18)
