# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import tree_util

from jax_scalify.core import DTypeLike, is_scaled_leaf

Leaf = Any


def astype(tree: Any, dtype: DTypeLike, floating_only: bool = False) -> Any:
    """Map `astype` method to all pytree leaves, `Array` or `ScaledArray`.

    Args:
        tree: the pytree to cast.
        dtype: Dtype to cast to.
        floating_only: Only convert leaves with floating datatype.

    Returns:
        A new PyTree with the same structure, with casting to new dtype.
    """
    if floating_only:
        # Convert only leaves with floating dtype.
        cast_fn = lambda v: v.astype(dtype) if jnp.issubdtype(v.dtype, jnp.floating) else v
        return tree_util.tree_map(cast_fn, tree, is_leaf=is_scaled_leaf)
    return tree_util.tree_map(lambda v: v.astype(dtype), tree, is_leaf=is_scaled_leaf)


def all(tree: Any) -> bool:
    """Call all() over the leaves of a tree, `Array` or `ScaledArray`

    Args:
        tree: the pytree to evaluate
    Returns:
        result: boolean True or False
    """
    return all(jax.tree_util.tree_leaves(tree, is_leaf=is_scaled_leaf))


def flatten(tree: Any) -> tuple[list[Leaf], tree_util.PyTreeDef]:
    """Flattens a pytree, with `Array` or `ScaledArray` leaves.

    The flattening order (i.e. the order of elements in the output list)
    is deterministic, corresponding to a left-to-right depth-first tree
    traversal.

    Args:
        tree: a pytree to flatten.

    Returns:
        A pair where the first element is a list of leaf values and the second
        element is a treedef representing the structure of the flattened tree.

    See Also:
        - :func:`jax_scalify.tree.leaves`
        - :func:`jax_scalify.tree.structure`
        - :func:`jax_scalify.tree.unflatten`
    """
    return tree_util.tree_flatten(tree, is_leaf=is_scaled_leaf)


def leaves(
    tree: Any,
) -> list[Leaf]:
    """Gets the leaves (`Array` or `ScaledArray`) of a pytree.

    Args:
        tree: the pytree for which to get the leaves

    Returns:
        leaves: a list of tree leaves.

    See Also:
        - :func:`jax_scalify.tree.flatten`
        - :func:`jax_scalify.tree.structure`
        - :func:`jax_scalify.tree.unflatten`
    """
    return tree_util.tree_leaves(tree, is_leaf=is_scaled_leaf)


def map(f: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    """Maps a multi-input function over pytree args to produce a new pytree.

    Args:
      f: function that takes ``1 + len(rest)`` arguments, to be applied at the
        corresponding leaves of the pytrees.
      tree: a pytree to be mapped over, with each leaf providing the first
        positional argument to ``f``.
      rest: a tuple of pytrees, each of which has the same structure as ``tree``
        or has ``tree`` as a prefix.

    Returns:
      A new pytree with the same structure as ``tree`` but with the value at each
      leaf given by ``f(x, *xs)`` where ``x`` is the value at the corresponding
      leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in
      ``rest``.

    See Also:
      - :func:`jax_scalify.tree.leaves`
      - :func:`jax_scalify.tree.reduce`
    """
    return tree_util.tree_map(f, tree, *rest, is_leaf=is_scaled_leaf)


def structure(tree: Any) -> tree_util.PyTreeDef:
    """Gets the treedef for a pytree, with `Array` or `ScaledArray` leaves.

    Args:
        tree: the pytree for which to get the leaves

    Returns:
        pytreedef: a PyTreeDef representing the structure of the tree.

    See Also:
      - :func:`jax_scalify.tree.flatten`
      - :func:`jax_scalify.tree.leaves`
      - :func:`jax_scalify.tree.unflatten`
    """
    return tree_util.tree_structure(tree, is_leaf=is_scaled_leaf)


# Alias of JAX tree unflatten.
unflatten = jax.tree_util.tree_unflatten
