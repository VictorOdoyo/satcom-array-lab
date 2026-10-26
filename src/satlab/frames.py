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
