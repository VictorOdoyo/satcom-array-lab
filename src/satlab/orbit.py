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


def elements(state, model=None):
    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    x = finite(state, real=True)
    if x.shape != (6,):
        raise ValueError("six-state orbit required")
    r, v = x[:3], x[3:]
    rn = np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    if min(rn, hn) < 1:
        raise ValueError("nondegenerate orbital position and momentum required")
    energy = float(v @ v / 2 - p.mu_m3_s2 / rn)
    if energy >= 0:
        raise ValueError("bound elliptic orbit required")
    ev = np.cross(v, h) / p.mu_m3_s2 - r / rn
    e = np.linalg.norm(ev)
    node = np.cross([0, 0, 1], h)
    nn = np.linalg.norm(node)
    inc = np.arccos(np.clip(h[2] / hn, -1, 1))
    circular = e < 1e-10
    equatorial = nn / hn < 1e-10
    sign = 1 if h[2] >= 0 else -1
    raan = np.arctan2(node[1], node[0]) % (2 * np.pi) if not equatorial else 0.0

    def signed_angle(a, b):
        return np.arctan2(np.cross(a, b) @ h / hn, a @ b) % (2 * np.pi)

    if not circular:
        argument = signed_angle(node, ev) if not equatorial else np.arctan2(sign * ev[1], ev[0]) % (2 * np.pi)
        anomaly = signed_angle(ev, r)
    else:
        argument = 0.0
        anomaly = signed_angle(node, r) if not equatorial else np.arctan2(sign * r[1], r[0]) % (2 * np.pi)
    return {
        "semimajor_m": float(-p.mu_m3_s2 / (2 * energy)),
        "eccentricity": float(e),
        "inclination_rad": float(inc),
        "raan_rad": float(raan),
        "argument_rad": float(argument),
        "true_anomaly_rad": float(anomaly),
        "circular": bool(circular),
        "equatorial": bool(equatorial),
    }
