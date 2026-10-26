import numpy as np


def db(value, floor_db=-200):
    from .core import finite

    x = finite(value, real=True)
    if np.any(x < 0) or not np.isfinite(floor_db):
        raise ValueError("nonnegative physical ratio and finite display floor required")
    return 10 * np.log10(np.maximum(x, 10 ** (floor_db / 10)))


def linear(decibels):
    from .core import finite

    value = 10 ** (finite(decibels, real=True) / 10)
    if not np.isfinite(value).all():
        raise ValueError("decibel conversion exceeds finite range")
    return value


def free_space_loss(range_m, frequency_hz):
    from .core import finite, positive
    from .earth import C

    r = finite(range_m, real=True)
    if np.any(r <= 0):
        raise ValueError("positive propagation ranges required")
    return 20 * np.log10(4 * np.pi * r * positive(frequency_hz) / C)


def atmosphere(
    elevation_rad,
    zenith_clear_db=0.5,
    rain_rate_mm_h=0,
    rain_coefficient=0.08,
    rain_exponent=1.1,
    rain_height_km=3,
    temperature_k=280,
):
    from .core import finite, positive

    el = finite(elevation_rad, real=True)
    if np.any((el <= 0) | (el > np.pi / 2)):
        raise ValueError("positive visible elevation required")
    clear = positive(zenith_clear_db, allow_zero=True)
    rain = positive(rain_rate_mm_h, allow_zero=True)
    specific = positive(rain_coefficient, allow_zero=True) * rain ** positive(rain_exponent)
    attenuation = (clear + specific * positive(rain_height_km)) / np.sin(el)
    transmission = 10 ** (-attenuation / 10)
    sky = 2.725 * transmission + positive(temperature_k) * (1 - transmission)
    return {"attenuation_db": attenuation, "sky_temperature_k": sky}
