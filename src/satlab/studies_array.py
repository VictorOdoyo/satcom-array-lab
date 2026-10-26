import numpy as np

from .results import make


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
