# Satellite Array Lab
[![Engineering checks](https://github.com/VictorOdoyo/satcom-array-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/VictorOdoyo/satcom-array-lab/actions/workflows/ci.yml)

A reproducible electrical-engineering laboratory connecting satellite geometry, phased receive arrays, physical link budgets, noisy beacon tracking and adaptive service. Twenty-eight studies cover coupled synthetic passes and independent orbital, array and modem benchmarks.

```powershell
uv sync --locked --extra dev
uv run satlab run --case closed-loop-pass --output outputs/pass
uv run satlab run --case beam-footprint --output outputs/beam
uv run satlab run --case burst-acquisition --output outputs/burst
uv run pytest -q
```

The receiver uses noisy beacon observations to correct ephemeris, panel and carrier biases. Handover and ACM decisions use predicted or measured quantities; actual truth only synthesizes observations and scores channel delivery. Adverse cases include rain, failed elements, phase errors, interference, beacon loss and outliers.

[Computed results and complete archives](docs/results.md) · [Architecture](docs/architecture.md) · [Equations and signs](docs/models.md) · [Receiver policy](docs/receiver.md) · [Experiments](docs/experiments.md) · [Reproduce](docs/reproduction.md) · [Validation](docs/validation.md) · [References](docs/references.md)

This is a synthetic laboratory. Coded service is a declared threshold proxy; actual waveform BER uses uncoded QPSK. The illustrative rain law is not a full availability model. The main pass uses circular ephemerides and a synthetic station, while independent benches exercise universal-variable and J2 propagation. MIT licensed.
