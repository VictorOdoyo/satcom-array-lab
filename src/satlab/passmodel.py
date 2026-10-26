from dataclasses import asdict, dataclass

from .core import integer, positive


@dataclass(frozen=True)
class Config:
    duration_s: float = 900
    step_s: float = 1
    epoch: str = "2025-01-01T00:00:00Z"
    altitude_m: float = 600000
    ephemeris_altitude_bias_m: float = 10000
    carrier_hz: float = 20e9
    symbol_rate_s: float = 2e6
    eirp_dbw: float = 30
    receiver_k: float = 300
    rain_mm_h: float = 0
    phase_bits: int = 8
    phase_std_rad: float = 0
    failed_fraction: float = 0
    closed_loop: bool = True
    pass_centers_s: tuple = (0.0,)
    beacon_gap_s: tuple = ()
    inject_outlier: bool = False
    interference_inr: float = 0
    null_interference: bool = False
    seed: int = 0

    def __post_init__(self):
        from .earth import julian_date

        for key in ("duration_s", "step_s", "altitude_m", "carrier_hz", "symbol_rate_s", "receiver_k"):
            positive(getattr(self, key))
        for key in ("rain_mm_h", "phase_std_rad", "interference_inr"):
            positive(getattr(self, key), allow_zero=True)
        integer(self.phase_bits)
        integer(self.seed, 0)
        if self.phase_bits > 16 or not 0 <= self.failed_fraction <= 1 or not self.pass_centers_s:
            raise ValueError("bounded phase precision and failure fraction required")
        if self.altitude_m + self.ephemeris_altitude_bias_m <= 0:
            raise ValueError("positive nominal altitude required")
        if (
            type(self.closed_loop) is not bool
            or type(self.null_interference) is not bool
            or type(self.inject_outlier) is not bool
        ):
            raise ValueError("boolean receiver switches required")
        from .core import finite

        finite([self.eirp_dbw, self.ephemeris_altitude_bias_m, *self.pass_centers_s], real=True)
        if self.beacon_gap_s and (
            len(self.beacon_gap_s) != 2 or self.beacon_gap_s[1] <= self.beacon_gap_s[0]
        ):
            raise ValueError("ordered beacon gap required")
        julian_date(self.epoch)

    def to_dict(self):
        return asdict(self)
