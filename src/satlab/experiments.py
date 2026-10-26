"""Validated reproducible run requests and lazy study dispatch."""

import json
from dataclasses import dataclass
from pathlib import Path

CASES = {
    "closed-loop-pass": [
        "studies_pass",
        "nominal",
        "Compute a complete ephemeris-to-beacon closed-loop satellite pass",
    ],
    "ephemeris-only-pass": [
        "studies_pass",
        "open_loop",
        "Expose physical pointing and frequency losses without beacon feedback",
    ],
    "rain-fade-pass": [
        "studies_pass",
        "rain",
        "Couple illustrative rain attenuation and sky noise into adaptive service",
    ],
    "one-bit-phase-pass": [
        "studies_pass",
        "one_bit",
        "Measure pass degradation with one-bit receive phase shifters",
    ],
    "calibration-error-pass": [
        "studies_pass",
        "calibration",
        "Propagate unknown receiver phase calibration errors through pass tracking",
    ],
    "element-failure-pass": [
        "studies_pass",
        "failures",
        "Retain thermal-noise normalization with half the receive elements failed",
    ],
    "aperture-outage-pass": ["studies_pass", "outage", "Retain total array failure as a zero-service result"],
    "beacon-gap-pass": [
        "studies_pass",
        "gap",
        "Demonstrate lock coasting loss and reacquisition across a beacon blackout",
    ],
    "beacon-outlier-pass": [
        "studies_pass",
        "outlier",
        "Gate corrupted pointing and carrier measurements during a pass",
    ],
    "constellation-handover": [
        "studies_pass",
        "handover",
        "Handover among three predicted passes with receiver reacquisition",
    ],
    "spatial-interference-pass": [
        "studies_pass",
        "interference",
        "Inject a spatially coherent interferer into receiver noise",
    ],
    "null-steered-pass": [
        "studies_pass",
        "nulling",
        "Suppress a declared interferer with constrained receive weights",
    ],
    "kepler-invariants": [
        "studies_orbit",
        "invariants",
        "Audit universal-variable orbital energy and angular momentum over an eccentric revolution",
    ],
    "rk4-convergence": [
        "studies_orbit",
        "convergence",
        "Measure fourth-order Cartesian propagation against independent universal anomaly solutions",
    ],
    "j2-precession": [
        "studies_orbit",
        "precession",
        "Resolve J2 nodal precession and conserved axial momentum with an independent secular reference",
    ],
    "doppler-range-derivative": ["studies_orbit", "doppler", "Independent range derivative Doppler"],
    "hohmann-transfer": [
        "studies_orbit",
        "transfer",
        "Compute signed two-impulse orbit transfers and transfer-time tradeoffs",
    ],
    "beam-footprint": [
        "studies_array",
        "footprint",
        "Map receive gain over the physical direction-cosine disk",
    ],
    "beam-squint": ["studies_array", "squint", "Fixed phase-shifter beam squint"],
    "aperture-tapers": [
        "studies_array",
        "tapers",
        "Compare receive taper sidelobes against their effective thermal-noise gain",
    ],
    "constrained-null": [
        "studies_array",
        "null",
        "Audit complex receive constraints and spatial-interference suppression",
    ],
    "phase-quantization": [
        "studies_array",
        "quantization",
        "Sweep phase-shifter precision over scan angle with normalized receive-noise power",
    ],
    "qpsk-ber": [
        "studies_rf",
        "ber",
        "Compare seeded uncoded QPSK errors with analytic BER and Wilson uncertainty",
    ],
    "burst-acquisition": [
        "studies_rf",
        "burst",
        "Decode an actual noisy QPSK burst after tone acquisition and pilot correction",
    ],
    "coherent-frequency-loss": [
        "studies_rf",
        "coherent",
        "Verify analytic coherent integration loss against discrete complex tone sums",
    ],
    "atmosphere-noise": [
        "studies_rf",
        "atmosphere",
        "Couple slant attenuation to atmospheric brightness and system noise",
    ],
    "link-budget-tradeoff": [
        "studies_rf",
        "budget",
        "Compute required satellite EIRP from range and declared symbol-energy target",
    ],
    "adaptive-mode-hysteresis": [
        "studies_rf",
        "adaptation",
        "Exercise causal ACM upgrades downgrades and short pilot-estimate outages",
    ],
}


@dataclass(frozen=True)
class ExperimentConfig:
    case: str = "closed-loop-pass"
    resolution_scale: int = 1
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.case, str) or self.case not in CASES:
            raise ValueError("unknown experiment case")
        if type(self.resolution_scale) is not int or not 1 <= self.resolution_scale <= 3:
            raise ValueError("resolution_scale must be an integer from 1 to 3")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    def to_dict(self):
        return {"case": self.case, "resolution_scale": self.resolution_scale, "seed": self.seed}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"case", "resolution_scale", "seed"}:
            raise ValueError("request must be an object containing only case, resolution_scale and seed")
        return cls(**value)


def load_config(path):
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run(config, progress=None):
    from importlib import import_module

    from .reporting import provenance, stamp
    from .results import validate

    if not isinstance(config, ExperimentConfig):
        raise ValueError("validated ExperimentConfig required")
    configuration = config.to_dict()
    source = provenance(configuration)
    module, function, _ = CASES[config.case]
    if progress is not None:
        progress("Solving " + config.case)
    result = getattr(import_module("satlab." + module), function)(
        scale=config.resolution_scale, seed=config.seed
    )
    validate(result)
    if result["name"] != config.case:
        raise ValueError("study returned a different case name")
    return stamp(result, configuration, source)
