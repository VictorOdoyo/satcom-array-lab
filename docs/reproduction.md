# Reproduction
Install Python 3.11 or newer and uv. From the checkout:

```powershell
uv sync --locked --extra dev
uv run satlab list
uv run satlab run --case closed-loop-pass --output outputs/pass
uv run satlab batch --data-only --output outputs/catalog
uv run satlab render outputs/catalog
uv run satlab verify outputs/catalog
uv run satlab replay outputs/catalog/closed-loop-pass/fields.npz --output outputs/replayed
uv run pytest -q
```

A JSON request contains only case, resolution_scale (integer 1..3) and seed (integer 0..2³²-1). Example: `{"case":"beam-footprint","resolution_scale":2,"seed":7}`. Use `run --config request.json --output outputs/request`. CLI overrides cannot be mixed with a JSON request.

The Python API exposes `satlab.passmodel.Config` and `simulate` for custom pass settings. Resolution scale reduces the pass timestep and refines numerical study grids. It is not a claim of increased physical fidelity.

NPZ uses numeric arrays and a Unicode JSON metadata field, with pickle disabled. Every report has a complete SHA256 manifest. `verify` detects modified tables, plots, metadata and arrays. Replay retains original solve provenance. To publish a gallery, capture from one clean committed revision, render, verify, then run `uv run python examples/build_gallery.py outputs/catalog`.
