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
