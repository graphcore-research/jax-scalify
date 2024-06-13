# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Callable, Dict

from jax import tree_util
from jax._src.debugging import debug_callback as debug_callback_orig
from jax._src.debugging import debug_callback_p

from .interpreters import ScaledArray, register_scaled_op


def get_debug_callback_effect(ordered: bool) -> Any:
    """Backward compatible effect factory method."""
    try:
        from jax._src.debugging import debug_effect, ordered_debug_effect

        return ordered_debug_effect if ordered else debug_effect
    except ImportError:
        from jax._src.debugging import DebugEffect

        return DebugEffect.ORDERED_PRINT if ordered else DebugEffect.PRINT


def debug_callback(callback: Callable[..., Any], *args: Any, ordered: bool = False, **kwargs: Any) -> None:
    # We need our custom version of `debug_callback` to deal with
    # changing JAX pytrees.
    # FIXME: probably patch `debug_callback` in JAX.
    flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
    effect = get_debug_callback_effect(ordered)

    def _flat_callback(*flat_args):
        args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
        callback(*args, **kwargs)
        return []

    # Storing in original PyTree and callback function.
    # Allowing custom interpreters to retrieve and modify this information.
    _flat_callback.__callback_fn = callback  # type:ignore
    _flat_callback.__callback_in_tree = in_tree  # type:ignore
    debug_callback_p.bind(*flat_args, callback=_flat_callback, effect=effect)


debug_callback.__doc__ = debug_callback_orig.__doc__


def scaled_debug_callback(*args: ScaledArray, **params: Dict[str, Any]) -> Any:
    """Scaled `debug_callback`: properly forwarding ScaledArrays
    to host callback.
    """
    flat_callback_fn = params["callback"]
    if not hasattr(flat_callback_fn, "__callback_fn"):
        raise NotImplementedError("Please use `jsa.debug_callback` function instead of original JAX function.")
    callback_fn = flat_callback_fn.__callback_fn
    in_pytree = flat_callback_fn.__callback_in_tree  # type:ignore
    # Re-build original input, with scaled arrays.
    scaled_args, scaled_kwargs = tree_util.tree_unflatten(in_pytree, args)
    # Re-build ordered boolean, in a backward compatible way.
    ordered = "ordered" in str(params["effect"]).lower()
    debug_callback(callback_fn, *scaled_args, ordered=ordered, **scaled_kwargs)
    return []


register_scaled_op(debug_callback_p, scaled_debug_callback)
