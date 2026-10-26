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


def phase(scale=1, seed=42):
    return solve("receiver-phase-noise", scale, seed, linewidth_hz=30)


def clock(scale=1, seed=42):
    return solve("receiver-clock-drift", scale, seed, clock_ppm=1500)


def nyquist(scale=1, seed=42):
    from .waveform import rrc

    h = rrc(8, 10, 0.25)
    combined = np.convolve(h, h[::-1])
    center = len(h) - 1
    time = (np.arange(len(combined)) - center) / 8
    sample_index = np.arange(-5, 6) * 8 + center
    isi = combined[sample_index].copy()
    isi[5] = 0
    return make(
        "rrc-nyquist-pulse",
        {"rolloff": 0.25, "span_symbols": 10, "samples_per_symbol": 8},
        {"maximum_off_center_isi": float(max(abs(isi)))},
        {
            "symbol_time": time,
            "combined": combined,
            "sample_time": (sample_index - center) / 8,
            "nyquist_samples": combined[sample_index],
        },
        [
            curve(
                "Finite-span matched pulse",
                "symbol_time",
                [("combined", "Tx RRC convolved with Rx RRC")],
                "Symbol intervals from pulse center",
                "Pulse amplitude",
            ),
            curve(
                "Symbol-spaced matched samples",
                "sample_time",
                [("nyquist_samples", "Samples")],
                "Symbol offset",
                "Amplitude",
                style="o",
            ),
        ],
        assumptions=[
            "Finite pulse truncation retains small residual ISI rather than asserting ideal infinite-span zeros."
        ],
    )


def image_rejection(scale=1, seed=42):
    from .link import db
    from .waveform import correct_iq, fit_iq, iq_imbalance, periodogram

    count = 4096
    fs = 80000.0
    tone = np.exp(2j * np.pi * np.arange(count) / 8)
    rx = iq_imbalance(tone, 0.6, 0.25, 0.03j)
    fit = fit_iq(rx[:256], tone[:256])
    corrected = correct_iq(rx, fit["coefficients"])
    before = periodogram(rx, fs)
    after = periodogram(corrected, fs)
    positive = np.argmin(abs(before["frequency_hz"] - 10000))
    negative = np.argmin(abs(before["frequency_hz"] + 10000))
    ratio_before = float(db(before["power_per_hz"][positive] / before["power_per_hz"][negative]))
    ratio_after = float(db(after["power_per_hz"][positive] / max(after["power_per_hz"][negative], 1e-30)))
    return make(
        "iq-image-rejection",
        {"tone_hz": 10000, "quadrature_gain": 0.6, "phase_error_rad": 0.25},
        {"image_rejection_before_db": ratio_before, "image_rejection_after_db": ratio_after},
        {
            "frequency_khz": before["frequency_hz"] / 1000,
            "before_db": db(before["power_per_hz"], -160),
            "after_db": db(after["power_per_hz"], -160),
        },
        [
            curve(
                "Conjugate image and DC calibration",
                "frequency_khz",
                [("before_db", "Impaired"), ("after_db", "Calibrated")],
                "Frequency (kHz)",
                "Power density (dB/Hz)",
            )
        ],
        assumptions=[
            "Noiseless synthetic calibration benchmark; extremely small residual image is numerical cancellation, not hardware performance."
        ],
    )
