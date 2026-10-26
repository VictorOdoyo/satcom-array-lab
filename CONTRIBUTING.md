# Contributing
Use a dedicated branch and keep changes physically testable. State coordinate frames, units, receive/transmit normalization, sample timing and observability assumptions. Add independent analytic or conservation evidence for numerical changes. Retain adverse and failed cases.

Run `uv sync --locked --extra dev`, `uv run ruff check src tests examples`, `uv run ruff format --check src tests examples` and `uv run pytest -q`. If study outputs change, regenerate the whole gallery from one clean committed source revision, render, verify and publish with the gallery builder. Never edit captured summaries by hand.

Do not commit local credentials, service tokens, private station information or generated outputs outside the reviewed gallery.
