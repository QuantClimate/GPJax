# Benchmarks

Continuous benchmarking with [ASV](https://asv.readthedocs.io/).

Trends are tracked on a per-commit basis on `main`. The dashboard is
rendered into the docs site at `/benchmarks/` whenever docs deploy.

## Five rules every benchmark follows

1. **Always `realise(result)`** at the end of every timed call. JAX is
   async; without the `block_until_ready()` calls inside `realise`, we
   time dispatch, not work.
2. **No beartype, no hypothesis** in the bench env. Both come in via
   `tests/conftest.py`, which ASV does not load. Do not add a `conftest.py`
   here.
3. **Pin `--machine`** when running. Trend series are keyed on machine
   identity. Switching laptops = new series under a new name.
4. **Cache discipline.** `time_*` benchmarks rely on a warm JAX cache
   (set up in `setup()` with one untimed call). `track_compile_*`
   benchmarks call `jax.clear_caches()` in `setup()`.
5. **Python is pinned to 3.13.** Bumping the pin in `asv.conf.json`
   resets the historical series. Don't.

## One-time setup (per machine)

```bash
uv python install 3.13

# Orphan results branch (only needed once per repo)
git checkout --orphan asv-results
git rm -rf .
git commit --allow-empty -m "init asv-results"
git push origin asv-results
git checkout main

# Worktree (per machine)
git worktree add .asv-results asv-results

# Machine record (per machine)
uv run asv machine --machine "$(hostname -s)" --yes
```

## Running benchmarks

```bash
uv run poe bench-constraints  # refresh asv-constraints.txt from uv.lock
uv run poe bench              # benchmarks current HEAD
uv run poe bench-publish      # local HTML preview at .asv/html/
uv run poe bench-push         # commit + push results to asv-results branch
uv run poe bench-all          # all four, chained
```

`asv-constraints.txt` pins ASV's per-commit env to the same versions as
`.venv` (driven by `uv.lock`). It must be regenerated and committed
whenever `uv.lock` changes — `bench-all` does this automatically; manual
runs should call `bench-constraints` first.

## Comparing two refs (PR-gate flow)

```bash
uv run poe bench-compare main my-feature-branch
```

Prints a regression report with significance markers. No branches touched.

## Adding a benchmark

1. Edit the appropriate `kernels.py` / `linalg.py` / `objectives.py` /
   `compile.py` file.
2. Run `uv run pytest tests/test_benchmarks_smoke.py -v` to catch import
   or runtime issues.
3. Run `uv run asv run --quick --python=same --bench <YourSuite>` to see
   one timing against the working tree.
4. Once the benchmark lands on `main`, `uv run poe bench-check` will
   validate suite structure as part of CI.

## Notes on bench-check

`bench-check` runs `asv check`, which builds an env for the commit
referenced by `asv.conf.json`'s `branches: ["main"]`. It is wired into
the `all-tests` poe sequence, so it runs on every `uv run poe all-tests`
alongside `lint`, `docstrings` and `test`.
