import numpy as np


class BiasTracker:
    def __init__(self, dimension=1, initial_std=1000, rate_std=10, process_density=1):
        from .core import integer, positive

        self.dimension = integer(dimension)
        self.process = positive(process_density, allow_zero=True)
        self.state = np.zeros(2 * dimension)
        self.covariance = np.diag(
            [positive(initial_std) ** 2] * dimension + [positive(rate_std) ** 2] * dimension
        )
        self.time = None

    def predict(self, time_s):
        from .core import finite

        time = float(finite(time_s, real=True))
        if self.time is not None and time < self.time:
            raise ValueError("monotone bias-tracker clock required")
        dt = 0 if self.time is None else time - self.time
        self.time = time
        eye = np.eye(self.dimension)
        f = np.block([[eye, dt * eye], [np.zeros_like(eye), eye]])
        q = self.process * np.block([[eye * dt**3 / 3, eye * dt**2 / 2], [eye * dt**2 / 2, eye * dt]])
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q
        return self.state[: self.dimension].copy()


def correct_bias(tracker, measured_bias, standard_deviation, gate=9):
    from .core import finite, positive

    d = tracker.dimension
    z = finite(measured_bias, real=True)
    std = finite(standard_deviation, real=True)
    z = np.atleast_1d(z)
    std = np.broadcast_to(std, (d,))
    if z.shape != (d,) or np.any(std <= 0):
        raise ValueError("one measured offset and positive uncertainty per tracked dimension required")
    p = tracker.covariance
    variance = np.diag(std * std)
    residual = z - tracker.state[:d]
    s = p[:d, :d] + variance
    nis = float(residual @ np.linalg.solve(s, residual))
    if nis > positive(gate):
        return {"accepted": False, "nis": nis}
    gain = np.linalg.solve(s, p[:, :d].T).T
    h = np.column_stack([np.eye(d), np.zeros((d, d))])
    j = np.eye(2 * d) - gain @ h
    tracker.state += gain @ residual
    tracker.covariance = j @ p @ j.T + gain @ variance @ gain.T
    tracker.covariance = (tracker.covariance + tracker.covariance.T) / 2
    return {"accepted": True, "nis": nis}


class Lock:
    def __init__(self, acquisition_dwell_s=2, coast_s=5):
        from .core import positive

        self.dwell = positive(acquisition_dwell_s, allow_zero=True)
        self.coast = positive(coast_s)
        self.first = None
        self.last_good = None
        self.time = None
        self.locked = False

    def available_at(self, time_s):
        return bool(self.locked and self.last_good is not None and time_s - self.last_good < self.coast)

    def update(self, time_s, accepted):
        from .core import finite

        time = float(finite(time_s, real=True))
        if type(accepted) is not bool or (self.time is not None and time < self.time):
            raise ValueError("boolean tracking observation and monotone clock required")
        self.time = time
        if self.last_good is not None and time - self.last_good >= self.coast:
            self.locked = False
            self.first = None
        if accepted:
            if self.first is None:
                self.first = time
            self.last_good = time
            if time - self.first >= self.dwell:
                self.locked = True
        else:
            self.first = None
        return self.available_at(time)
