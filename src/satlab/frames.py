import numpy as np


def rz(angle_rad):
    from .core import finite

    angle = float(finite(angle_rad, real=True))
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def eci_to_ecef(position, angle_rad):
    from .core import finite

    r = finite(position, real=True)
    if r.shape[-1] != 3:
        raise ValueError("Cartesian position triples required")
    return r @ rz(angle_rad)


def ecef_to_eci(position, angle_rad):
    from .core import finite

    r = finite(position, real=True)
    if r.shape[-1] != 3:
        raise ValueError("Cartesian position triples required")
    return r @ rz(angle_rad).T


def state_to_ecef(state, angle_rad, model=None):
    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    x = finite(state, real=True)
    if x.shape != (6,):
        raise ValueError("inertial position and velocity required")
    r = eci_to_ecef(x[:3], angle_rad)
    velocity = eci_to_ecef(x[3:] - np.cross([0, 0, p.rotation_rad_s], x[:3]), angle_rad)
    return np.r_[r, velocity]


def state_to_eci(state, angle_rad, model=None):
    from .core import finite
    from .earth import Earth

    p = Earth() if model is None else model
    x = finite(state, real=True)
    if x.shape != (6,):
        raise ValueError("Earth-fixed position and velocity required")
    r = ecef_to_eci(x[:3], angle_rad)
    velocity = ecef_to_eci(x[3:], angle_rad) + np.cross([0, 0, p.rotation_rad_s], r)
    return np.r_[r, velocity]


def rtn_matrix(state):
    from .core import finite

    x = finite(state, real=True)
    if x.shape != (6,):
        raise ValueError("orbital Cartesian state required")
    r = x[:3]
    h = np.cross(r, x[3:])
    if min(np.linalg.norm(r), np.linalg.norm(h)) < 1e-10:
        raise ValueError("orbital plane unobservable")
    radial = r / np.linalg.norm(r)
    normal = h / np.linalg.norm(h)
    transverse = np.cross(normal, radial)
    return np.vstack([radial, transverse, normal])
