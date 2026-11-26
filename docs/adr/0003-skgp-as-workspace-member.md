# ADR-0003: skgp is a uv workspace member in the GPJax repository

- **Status:** accepted
- **Date:** 2026-11-26
- **Deciders:** Thomas Pinder (design settled in a grilling session)

## Context

skgp is a scikit-learn-style package that is built on GPJax. It uses
approximately 20 names from GPJax's public API, for example `gpx.fit`,
`gpx.gps.Prior`, `gpx.objectives.elbo` and `gpx.Dataset`. Thus a change to
GPJax can break skgp. When the two packages were in separate repositories, a
break showed only after a GPJax release reached skgp.

## Decision

skgp moves into the GPJax repository, in the folder `skgp/`. The two packages
make a uv workspace: GPJax is the workspace root and skgp is a workspace
member. They share one `uv.lock`.

- A GPJax change and the skgp change that it needs go in one pull request, and
  one CI run tests them together.
- Each package keeps its own version, dependencies, PyPI listing and docs site
  (gpjax.quantclimate.com and skgp.quantclimate.com). Users install them as
  before, with `pip install gpjax` and `pip install skgp`.
- GPJax releases keep the tag format `v1.3.0`. skgp releases use
  `skgp-v0.1.0`, so that a skgp tag does not start GPJax's release workflow.
- skgp's files come in as one commit. skgp's earlier history stays in the
  archived, private `QuantClimate/skgp` repository.
- Both packages build with `uv_build`. One ruff configuration in the root
  `pyproject.toml` applies to both packages, and one set of prek hooks runs
  before each commit.
- skgp requires `gpjax>=1.1.0,<2`. The workspace always tests skgp against
  the GPJax code in the same commit. Thus a CI job also tests skgp with the
  oldest GPJax that skgp permits, from PyPI.
- A rename or removal in GPJax's public API obeys the rule in `AGENTS.md` and
  `docs/contributing.md`: the old name stays, with a `DeprecationWarning`,
  until skgp stops using it, and goes only in a major release. skgp's tests
  fail on a GPJax `DeprecationWarning`.

## Considered options

- **Separate repositories, and GPJax's CI runs skgp's tests on each GPJax pull
  request.** A pull request that breaks skgp fails before it merges. But a
  planned GPJax API change must wait for a separate skgp pull request, and two
  sets of CI and release workflows need maintenance.
- **skgp as an optional module of the `gpjax` package (for example
  `gpjax.sklearn`).** There is one package and one install. But the two share
  one version number, so each break in skgp's API forces a GPJax major
  release. Also, skgp's dependencies enter GPJax, and the name `skgp` leaves
  PyPI.

## Consequences

- Contributors and reviewers of GPJax also see skgp's issues and pull
  requests.
- skgp's code is public.
- Each skgp workflow runs only when the files that it depends on change.
  Changes to GPJax's code also run skgp's tests.
