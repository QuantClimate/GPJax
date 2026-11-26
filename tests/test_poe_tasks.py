# Copyright 2026 The thomaspinder Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from pathlib import Path
import tomllib


def _poe_tasks() -> dict:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())["tool"]["poe"]["tasks"]


def test_bench_check_wired_into_all_tests() -> None:
    """`bench-check` must run as part of the `all-tests` CI gate (issue #638).

    It was left out of the sequence pending a soak period for `benchmarks/`
    and `asv-constraints.txt` to land on `main`; that precondition is now
    satisfied, so `bench-check` belongs between `docstrings` and `test`.
    The static type check (`typecheck`) runs right after `lint`.
    """
    tasks = _poe_tasks()

    assert tasks["all-tests"]["sequence"] == [
        "lint",
        "typecheck",
        "docstrings",
        "bench-check",
        "test",
    ]


def test_bench_check_task_defined() -> None:
    tasks = _poe_tasks()

    assert tasks["bench-check"]["cmd"] == "asv check"


def test_typecheck_task_defined() -> None:
    tasks = _poe_tasks()

    assert tasks["typecheck"]["cmd"] == "ty check gpjax"
