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
