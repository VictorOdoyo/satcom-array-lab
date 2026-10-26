import numpy as np

from .results import curve, make


def footprint(scale=1, seed=42):
    from . import array, link

    p = array.Aperture()
    axis = np.linspace(-1, 1, 101 * scale)
    east, north = np.meshgrid(axis, axis)
    valid = east**2 + north**2 <= 1
    uv = np.stack([east, north], axis=-1)
    directions = array.from_direction_cosines(uv[valid])
    w = array.weights(array.from_direction_cosines([0.3, -0.2]), p)
    gain = np.full(valid.shape, -80.0)
    gain[valid] = link.db(array.response(w, directions, p)["snr_gain"] * array.element_gain(directions, p))
    plot = {
        "kind": "image",
        "title": "Receive pattern including element factor",
        "array": "gain_dbi",
        "x": axis.tolist(),
        "y": axis.tolist(),
        "xlabel": "East direction cosine",
        "ylabel": "North direction cosine",
        "component": "real",
        "mask": "valid",
        "colorbar": "Gain (dBi)",
        "aspect": "equal",
        "cmap": "viridis",
    }
    return make(
        "beam-footprint",
        {"elements": p.count, "command_uv": [0.3, -0.2]},
        {"peak_gain_dbi": float(max(gain[valid]))},
        {"gain_dbi": gain, "valid": valid, "east_uv": east, "north_uv": north},
        [plot],
    )


def squint(scale=1, seed=42):
    from . import array, link

    p = array.Aperture()
    u = np.linspace(-0.2, 0.9, 501 * scale)
    directions = array.from_direction_cosines(np.column_stack([u, np.zeros(len(u))]))
    w = array.weights(array.from_direction_cosines([0.5, 0]), p)
    arrays = {"east_uv": u}
    plots = []
    peaks = {}
    for f in (16e9, 20e9, 24e9):
        key = "gain_" + str(int(f / 1e9))
        gain = link.db(array.response(w, directions, p, f)["snr_gain"])
        arrays[key] = gain
        peaks[str(f)] = float(u[np.argmax(gain)])
    plots = [
        curve(
            "Fixed phase-shifter beam squint",
            "east_uv",
            [(k, k.replace("gain_", "") + " GHz") for k in arrays if k != "east_uv"],
            "East direction cosine",
            "Array SNR gain (dB)",
            ylim=(-50, 20),
        )
    ]
    return make("beam-squint", {"design_hz": 20e9, "command_uv": 0.5}, peaks, arrays, plots)
