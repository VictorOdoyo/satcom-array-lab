import numpy as np


def from_elements(
    semimajor_m, eccentricity, inclination_rad, raan_rad=0, argument_rad=0, true_anomaly_rad=0, model=None
):
    from .core import finite, positive
    from .earth import Earth
    from .frames import rz

    p = Earth() if model is None else model
    a = positive(semimajor_m)
    e = float(finite(eccentricity, real=True))
    inc, raan, arg, nu = finite([inclination_rad, raan_rad, argument_rad, true_anomaly_rad], real=True)
    if not 0 <= e < 1 or not 0 <= inc <= np.pi:
        raise ValueError("elliptic eccentricity and orbital inclination required")
    parameter = a * (1 - e * e)
    radius = parameter / (1 + e * np.cos(nu))
    r = np.array([radius * np.cos(nu), radius * np.sin(nu), 0.0])
    v = np.sqrt(p.mu_m3_s2 / parameter) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    c, s = np.cos(inc), np.sin(inc)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    rotation = rz(raan) @ rx @ rz(arg)
    return np.r_[rotation @ r, rotation @ v]
