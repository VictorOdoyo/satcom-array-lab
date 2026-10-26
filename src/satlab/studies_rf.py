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
