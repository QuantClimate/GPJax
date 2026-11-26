"""Smoke tests for the benchmark suite itself.

`asv check` validates structure without running. These tests run one
method per benchmark module to catch breakage that structural validation
misses (import errors at module load, broken setup() bodies, dead API
references). They are intentionally cheap - minimal params, single call.
"""

from __future__ import annotations

import importlib

import pytest

BENCH_MODULES = [
    "benchmarks.kernels",
    "benchmarks.linalg",
    "benchmarks.objectives",
    "benchmarks.compile",
    "benchmarks.state_space",
]


@pytest.mark.parametrize("module_name", BENCH_MODULES)
def test_benchmark_module_imports(module_name):
    """Every benchmark module imports cleanly."""
    importlib.import_module(module_name)


def test_kernels_gram_runs():
    from benchmarks.kernels import GramSuite

    suite = GramSuite()
    suite.setup("RBF", 100, 1)
    suite.time_gram("RBF", 100, 1)


def test_linalg_cholesky_runs():
    from benchmarks.linalg import CholeskySuite

    suite = CholeskySuite()
    suite.setup(100)
    suite.time_cholesky_factor(100)


def test_objectives_conjugate_mll_runs():
    from benchmarks.objectives import ConjugateMllSuite

    suite = ConjugateMllSuite()
    suite.setup(200)
    suite.time_conjugate_mll(200)


def test_compile_runs():
    from benchmarks.compile import CompileSuite

    suite = CompileSuite()
    suite.setup()
    elapsed = suite.track_compile_conjugate_mll()
    assert elapsed > 0


def test_state_space_mll_runs():
    from benchmarks.state_space import StateSpaceMllSuite

    suite = StateSpaceMllSuite()
    suite.setup(1000)
    suite.time_state_space_mll(1000)


def test_compile_elbo_runs():
    from benchmarks.compile import CompileSuite

    suite = CompileSuite()
    suite.setup()
    elapsed = suite.track_compile_elbo()
    assert elapsed > 0


def test_objectives_svgp_elbo_runs():
    from benchmarks.objectives import SvgpElboSuite

    suite = SvgpElboSuite()
    suite.setup(200, 32, 64)
    suite.time_elbo(200, 32, 64)


def test_objectives_vfe_elbo_runs():
    from benchmarks.objectives import VfeElboSuite

    suite = VfeElboSuite()
    suite.setup(200, 32)
    suite.time_collapsed_elbo(200, 32)


@pytest.mark.parametrize("family", ["standard", "whitened", "dual"])
def test_objectives_variational_parametrisation_runs(family):
    from benchmarks.objectives import VariationalParametrisationSuite

    suite = VariationalParametrisationSuite()
    suite.setup(family)
    suite.time_elbo(family)


def test_kernels_rff_gram_runs():
    from benchmarks.kernels import RffGramSuite

    suite = RffGramSuite()
    suite.setup("RBF", "rff", 200)
    suite.time_gram("RBF", "rff", 200)


def test_kernels_multioutput_gram_runs():
    from benchmarks.kernels import MultiOutputGramSuite

    suite = MultiOutputGramSuite()
    suite.setup("ICM_Q1", 100)
    suite.time_gram("ICM_Q1", 100)
    suite.setup("LCM_Q2", 100)
    suite.time_gram("LCM_Q2", 100)


def test_objectives_shared_dataset_is_deterministic():
    """Two suites at the same n must see bit-identical X, y. This is the
    invariant that lets the dashboard plot them on a shared axis."""
    from benchmarks._setup import make_shared_dataset

    a = make_shared_dataset(200)
    b = make_shared_dataset(200)
    assert (a.X == b.X).all()
    assert (a.y == b.y).all()
