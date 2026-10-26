from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mode:
    name: str
    bits_per_symbol: int
    code_rate: float
    threshold_esn0_db: float

    def __post_init__(self):
        from .core import finite, integer

        integer(self.bits_per_symbol)
        if not isinstance(self.name, str) or not self.name or not 0 < self.code_rate <= 1:
            raise ValueError("named modulation and bounded code rate required")
        finite(self.threshold_esn0_db, real=True)


MODES = (Mode("QPSK 1/2", 2, 0.5, 3), Mode("8PSK 2/3", 3, 2 / 3, 7.5), Mode("16QAM 3/4", 4, 0.75, 11))


def information_rate(mode, symbol_rate_s, framing_efficiency=0.95):
    from .core import positive

    if not isinstance(mode, Mode) or not 0 < framing_efficiency <= 1:
        raise ValueError("declared service mode and framing efficiency required")
    return positive(symbol_rate_s) * mode.bits_per_symbol * mode.code_rate * framing_efficiency


class AdaptiveMode:
    def __init__(self, modes=MODES, upgrade_margin_db=1, dwell_s=5):
        from .core import positive

        self.modes = tuple(modes)
        self.margin = positive(upgrade_margin_db, allow_zero=True)
        self.dwell = positive(dwell_s, allow_zero=True)
        self.index = None
        self.changed = None
        self.time = None
        if not self.modes or any(
            self.modes[k].threshold_esn0_db >= self.modes[k + 1].threshold_esn0_db
            for k in range(len(self.modes) - 1)
        ):
            raise ValueError("strictly ordered thresholds required")

    def update(self, time_s, esn0_db, available=True):
        from .core import finite

        time = float(finite(time_s, real=True))
        snr = float(finite(esn0_db, real=True))
        if self.time is not None and time < self.time:
            raise ValueError("monotone adaptive-mode clock required")
        self.time = time
        eligible = (
            [k for k, mode in enumerate(self.modes) if snr >= mode.threshold_esn0_db] if available else []
        )
        if not eligible:
            self.index = None
            self.changed = time
            return None
        best = max(eligible)
        if self.index is None or self.index not in eligible:
            self.index = best
            self.changed = time
        elif (
            best > self.index
            and time - self.changed >= self.dwell
            and snr >= self.modes[best].threshold_esn0_db + self.margin
        ):
            self.index = best
            self.changed = time
        return self.modes[self.index]


class Handover:
    def __init__(self, hysteresis_db=2, minimum_dwell_s=30, confirmation_s=5):
        from .core import positive

        self.hysteresis = positive(hysteresis_db, allow_zero=True)
        self.dwell = positive(minimum_dwell_s, allow_zero=True)
        self.confirmation = positive(confirmation_s, allow_zero=True)
        self.current = None
        self.changed = None
        self.candidate = None
        self.candidate_since = None
        self.time = None
        self.handovers = 0

    def update(self, time_s, predicted_scores):
        from .core import finite

        time = float(finite(time_s, real=True))
        if self.time is not None and time < self.time:
            raise ValueError("monotone handover clock required")
        self.time = time
        scores = dict(predicted_scores)
        for name, value in scores.items():
            if not isinstance(name, str) or not name:
                raise ValueError("named predicted satellite scores required")
            finite(value, real=True)
        if not scores:
            self.current = None
            self.candidate = None
            return None
        best = max(scores, key=scores.get)
        if self.current not in scores:
            if self.current is not None:
                self.handovers += 1
            self.current = best
            self.changed = time
            self.candidate = None
            return best
        better = best != self.current and scores[best] > scores[self.current] + self.hysteresis
        if not better:
            self.candidate = None
            return self.current
        if self.candidate != best:
            self.candidate = best
            self.candidate_since = time
        if time - self.changed >= self.dwell and time - self.candidate_since >= self.confirmation:
            self.current = best
            self.changed = time
            self.candidate = None
            self.handovers += 1
        return self.current


def accumulate(time_s, offered_bit_s, delivered_bit_s):
    from .core import finite, timebase

    t = timebase(time_s)
    offered = finite(offered_bit_s, real=True, ndim=1)
    delivered = finite(delivered_bit_s, real=True, ndim=1)
    if (
        offered.shape != t.shape
        or delivered.shape != t.shape
        or np.any(delivered < 0)
        or np.any(offered < delivered)
    ):
        raise ValueError("aligned nonnegative offered and delivered service rates required")
    dt = np.diff(t)
    return {
        "offered_bits": np.r_[0, np.cumsum(offered[:-1] * dt)],
        "delivered_bits": np.r_[0, np.cumsum(delivered[:-1] * dt)],
    }
