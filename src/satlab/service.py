from dataclasses import dataclass


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
