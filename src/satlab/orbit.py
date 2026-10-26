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


def stumpff(z):
    from .core import finite

    value = float(finite(z, real=True))
    if abs(value) < 1e-5:
        return (
            0.5 - value / 24 + value**2 / 720 - value**3 / 40320,
            1 / 6 - value / 120 + value**2 / 5040 - value**3 / 362880,
        )
    if value > 0:
        root = np.sqrt(value)
        return (1 - np.cos(root)) / value, (root - np.sin(root)) / root**3
    root = np.sqrt(-value)
    if root > 500:
        raise ValueError("hyperbolic universal argument exceeds supported finite range")
    return (np.cosh(root) - 1) / (-value), (np.sinh(root) - root) / root**3


def propagate(state, dt_s, model=None):
    from scipy.optimize import brentq

    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    x = finite(state, real=True)
    dt = float(finite(dt_s, real=True))
    if x.shape != (6,) or np.linalg.norm(x[:3]) < 1:
        raise ValueError("noncentral Cartesian orbit state required")
    if dt == 0:
        return x.copy()
    r0, v0 = x[:3], x[3:]
    radius = np.linalg.norm(r0)
    root_mu = np.sqrt(p.mu_m3_s2)
    alpha = 2 / radius - (v0 @ v0) / p.mu_m3_s2
    coefficient = (r0 @ v0) / root_mu

    def equation(chi):
        c, s = stumpff(alpha * chi * chi)
        return coefficient * chi * chi * c + (1 - alpha * radius) * chi**3 * s + radius * chi - root_mu * dt

    direction = np.sign(dt)
    bound = direction * max(1, abs(root_mu * dt / radius))
    for _ in range(60):
        if direction * equation(bound) >= 0:
            break
        bound *= 2
    else:
        raise ValueError("universal anomaly could not be bracketed")
    low, high = sorted([0.0, bound])
    chi = brentq(equation, low, high, xtol=1e-10, rtol=1e-14)
    c, s = stumpff(alpha * chi * chi)
    f = 1 - chi * chi * c / radius
    g = dt - chi**3 * s / root_mu
    r = f * r0 + g * v0
    rn = np.linalg.norm(r)
    fdot = root_mu / (rn * radius) * (alpha * chi**3 * s - chi)
    gdot = 1 - chi * chi * c / rn
    return np.r_[r, fdot * r0 + gdot * v0]


def acceleration(position, model=None, include_j2=False):
    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    r = finite(position, real=True)
    if r.shape != (3,) or np.linalg.norm(r) < 1 or type(include_j2) is not bool:
        raise ValueError("noncentral position and boolean gravity option required")
    radius = np.linalg.norm(r)
    result = -p.mu_m3_s2 * r / radius**3
    if include_j2:
        ratio = r[2] ** 2 / radius**2
        factor = 1.5 * p.j2 * p.mu_m3_s2 * p.radius_m**2 / radius**5
        result += factor * r * np.array([5 * ratio - 1, 5 * ratio - 1, 5 * ratio - 3])
    return result


def potential(position, model=None, include_j2=False):
    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    r = finite(position, real=True)
    radius = np.linalg.norm(r)
    if r.shape != (3,) or radius < 1:
        raise ValueError("noncentral gravity position required")
    value = -p.mu_m3_s2 / radius
    if include_j2:
        value += p.mu_m3_s2 * p.j2 * p.radius_m**2 / (2 * radius**3) * (3 * r[2] ** 2 / radius**2 - 1)
    return float(value)


def rhs(state, model=None, include_j2=False):
    from .core import finite

    x = finite(state, real=True)
    if x.shape != (6,):
        raise ValueError("Cartesian orbital state required")
    return np.r_[x[3:], acceleration(x[:3], model, include_j2)]


def step(state, dt_s, model=None, include_j2=False):
    from .core import finite

    x = finite(state, real=True)
    dt = float(finite(dt_s, real=True))
    k1 = rhs(x, model, include_j2)
    k2 = rhs(x + dt * k1 / 2, model, include_j2)
    k3 = rhs(x + dt * k2 / 2, model, include_j2)
    k4 = rhs(x + dt * k3, model, include_j2)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def invariants(state, model=None, include_j2=False):
    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    x = finite(state, real=True)
    if x.shape != (6,) or np.linalg.norm(x[:3]) < 1:
        raise ValueError("noncentral orbital state required")
    h = np.cross(x[:3], x[3:])
    ev = np.cross(x[3:], h) / p.mu_m3_s2 - x[:3] / np.linalg.norm(x[:3])
    return {
        "specific_energy_j_kg": float(x[3:] @ x[3:] / 2 + potential(x[:3], p, include_j2)),
        "angular_momentum_m2_s": h,
        "eccentricity_vector": ev,
    }


def circular_from_state(state, time_s, model=None):
    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    x = finite(state, real=True)
    t = finite(time_s, real=True)
    if x.shape != (6,) or np.linalg.norm(x[:3]) < 1:
        raise ValueError("circular Cartesian state required")
    radius = np.linalg.norm(x[:3])
    speed = np.sqrt(p.mu_m3_s2 / radius)
    rate = speed / radius
    if abs(x[:3] @ x[3:]) > 1e-8 * radius * speed or not np.isclose(np.linalg.norm(x[3:]), speed, rtol=1e-8):
        raise ValueError("state is not circular")
    phase = rate * t
    position = np.cos(phase)[..., None] * x[:3] + np.sin(phase)[..., None] * x[3:] / rate
    velocity = -rate * np.sin(phase)[..., None] * x[:3] + np.cos(phase)[..., None] * x[3:]
    return np.concatenate([position, velocity], axis=-1)


def overhead_state(station, angle_rad, altitude_m=600000, inclination_rad=np.deg2rad(53), model=None):
    from .core import finite, positive
    from .earth import Earth
    from .frames import ecef_to_eci

    p = Earth() if model is None else model
    height = positive(altitude_m)
    inc = float(finite(inclination_rad, real=True))
    ground = ecef_to_eci(station.ecef(p), angle_rad)
    radial = ground / np.linalg.norm(ground)
    longitude = np.arctan2(radial[1], radial[0])
    latitude = np.arcsin(radial[2])
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.cross(radial, east)
    ratio = np.cos(inc) / max(1e-15, np.cos(latitude))
    if not 0 <= inc <= np.pi or abs(ratio) > 1:
        raise ValueError("inclination cannot reach station latitude")
    direction = ratio * east + np.sqrt(max(0, 1 - ratio**2)) * north
    radius = np.linalg.norm(ground) + height
    position = radial * radius
    velocity = direction * np.sqrt(p.mu_m3_s2 / radius)
    return np.r_[position, velocity]
