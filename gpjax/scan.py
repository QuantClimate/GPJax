# Copyright 2023 The thomaspinder Contributors. All Rights Reserved.
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

from collections.abc import Callable
from typing import (
    Any,
    TypeVar,
)

import jax
from jax import lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import (
    Array,
    Shaped,
)
from tqdm.auto import trange

from gpjax.typing import (
    ScalarBool,
    ScalarInt,
)

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def _callback(cond: ScalarBool, func: Callable, *args: Any) -> None:
    r"""Callback a function for a given argument if a condition is true.

    Args:
        cond (bool): The condition.
        func (Callable): The function to call.
        *args (Any): The arguments to pass to the function.
    """
    # lax.cond requires a result, so we use a dummy result.
    _dummy_result = 0

    def _do_callback(_) -> int:
        """Perform the callback."""
        jax.debug.callback(func, *args)
        return _dummy_result

    def _not_callback(_) -> int:
        """Do nothing."""
        return _dummy_result

    _ = lax.cond(cond, _do_callback, _not_callback, operand=None)


def vscan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool | None = False,
    unroll: int | None = 1,
    log_rate: int | None = 10,
    log_value: bool | None = True,
) -> tuple[
    Carry, Shaped[Array, "..."]
]:  # return type should be Tuple[Carry, Y[Array]]...
    r"""Scan with verbose output.

    This is based on code from this [excellent blog post](https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/).

    Example:
        >>> import jax.numpy as jnp
        ...
        >>> def f(carry, x):
        ...     return carry + x, carry + x
        >>> init = 0
        >>> xs = jnp.arange(10)
        >>> vscan(f, init, xs)  # doctest: +SKIP
        (Array(45, dtype=int32), Array([ 0,  1,  3,  6, 10, 15, 21, 28, 36, 45], dtype=int32))

    Args:
        f (Callable[[Carry, X], Tuple[Carry, Y]]): A function that takes in a carry and
            an input and returns a tuple of a new carry and an output.
        init (Carry): The initial carry.
        xs (X): The inputs.
        length (Optional[int]): The length of the inputs. If None, then the length of
            the inputs is inferred.
        reverse (bool): Whether to scan in reverse.
        unroll (int): The number of iterations to unroll.
        log_rate (int): The rate at which to log the progress bar.
        log_value (bool): Whether to log the value of the objective function.

    Returns:
        Tuple[Carry, list[Y]]: A tuple of the final carry and the outputs.
    """
    _xs_flat = jtu.tree_leaves(xs)
    _length = length if length is not None else len(_xs_flat[0])
    _iter_nums = jnp.arange(_length)
    _remainder = _length % log_rate

    _progress_bar = trange(_length)
    _progress_bar.set_description("Compiling...", refresh=True)

    def _set_running(*args: Any) -> None:
        """Set the tqdm progress bar to running."""
        _progress_bar.set_description("Running", refresh=False)

    def _update_tqdm(*args: Any) -> None:
        """Update the tqdm progress bar with the latest objective value."""
        _value, _iter_num = args
        _progress_bar.update(_iter_num.item())

        if log_value and _value is not None:
            _progress_bar.set_postfix({"Value": f"{_value: .2f}"})

    def _close_tqdm(*args: Any) -> None:
        """Close the tqdm progress bar."""
        _progress_bar.close()

    def _body_fun(carry: Carry, iter_num_and_x: tuple[ScalarInt, X]) -> tuple[Carry, Y]:
        # Unpack iter_num and x.
        iter_num, x = iter_num_and_x

        # Compute body function.
        carry, y = f(carry, x)

        # Conditions for iteration number.
        _is_first: bool = iter_num == 0
        _is_multiple: bool = (iter_num % log_rate == 0) & (
            iter_num != _length - _remainder
        )
        _is_remainder: bool = iter_num == _length - _remainder
        _is_last: bool = iter_num == _length - 1

        # Update progress bar, if first of log_rate.
        _callback(_is_first, _set_running)

        # Update progress bar, if multiple of log_rate.
        _callback(_is_multiple, _update_tqdm, y, log_rate)

        # Update progress bar, if remainder.
        _callback(_is_remainder, _update_tqdm, y, _remainder)

        # Close progress bar, if last iteration.
        _callback(_is_last, _close_tqdm)

        return carry, y

    carry, ys = jax.lax.scan(
        _body_fun,
        init,
        (_iter_nums, xs),
        length=length,
        reverse=reverse,
        unroll=unroll,
    )

    return carry, ys


__all__ = [
    "vscan",
]
