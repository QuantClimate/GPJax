skgp lets anyone fit Gaussian process models from dataframes, without needing to know JAX or GPJax. See `README.md` for its purpose and scope, and `GLOSSARY.md` for the project's vocabulary.

## Where skgp lives

skgp is the `skgp/` folder of the GPJax repository (`QuantClimate/GPJax`), a member of the uv workspace whose root is GPJax (`docs/adr/0003-skgp-as-workspace-member.md` at the repository root). Paths in this file are relative to `skgp/`. In development skgp uses the GPJax code in the same commit, so a GPJax change and the skgp change that it needs go in one pull request. The repository root holds the one `uv.lock`, the ruff configuration, the `[tool.uv]` settings and the prek hooks. Run skgp's commands from `skgp/` with `--package skgp`, for example `uv sync --package skgp` and `uv run --package skgp pytest -m ""`. skgp's releases are tagged `skgp-v0.1.0`; GPJax's are tagged `v1.3.0`.

skgp has no users outside QuantClimate yet: change any API freely, with no compatibility shims or deprecations, and weigh each decision on its merits rather than on what changing it later would cost.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for QuantClimate/GPJax, shared with GPJax, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

skgp's own context: `GLOSSARY.md` and `docs/adr/` in `skgp/`. GPJax's ADRs are in `docs/adr/` at the repository root. See `docs/agents/domain.md`.

## Docs site

`docs/` is a Sphinx site, built the same way as GPJax's and Impulso's: `uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`. The Python code of the landing page and of the saving page runs in the test suite (`tests/test_readme.py`). The `skgp docs` workflow (`.github/workflows/skgp-docs.yml` at the repository root) builds the site on pull requests and deploys it to https://skgp.quantclimate.com from main.

## CI

skgp's workflows are `skgp-tests.yml`, `skgp-release.yml` and `skgp-docs.yml` in `.github/workflows/` at the repository root. `skgp-tests.yml` runs for changes to skgp and to GPJax, and also tests skgp with the oldest GPJax that skgp permits. The commit hook (`uv run prek install` at the repository root, once per clone) runs `ruff check`, `ruff format` and `ty`.
