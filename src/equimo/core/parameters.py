"""Trainability and weight-decay semantics of model leaves.

Every leaf carries semantic tags. Most come from its path
(:func:`equimo.core.tags.canonical_tags_for_path`); the modules holding a leaf
add the rest, which does not depend on how fields are named:

- a field declared with :func:`tagged` gives its tags to every leaf below it;
- a leaf held directly by a normalization layer registered with
  :func:`equimo.core.layers.norm.register_norm` gets ``norm``, unless its field
  declares its own tags.

A *buffer* is an array a module stores but never trains, such as a frozen
random basis or a copy kept to undo a merge. Declare it with :func:`buffer`;
:func:`trainable_mask` and :func:`weight_decay_mask` then keep it away from
optimizers, which would otherwise change it through weight decay even though
``jax.lax.stop_gradient`` gives it zero gradients. Values that follow from a
module's static configuration are computed where they are used instead of
being stored.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Iterable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from .tags import Tagger, canonical_tags_for_path, key_path_to_path

PyTree = Any

_TAGS_METADATA_KEY = "equimo.tags"

DEFAULT_NO_DECAY_TAGS: tuple[str, ...] = (
    "bias",
    "norm",
    "block.layer_scale",
    "ssm",
    "vector",
    "embedding.position",
    "embedding.class_token",
    "embedding.register_token",
    "embedding.distillation_token",
    "embedding.mask_token",
)
"""Tags of trainable leaves that weight decay leaves alone by default.

Biases, normalization parameters, layer scales, and state-space dynamics
(``A_log``, ``D``, ``dt_bias``) follow nearly every reference training recipe.
``vector`` covers every other array with at most one axis, as timm, MAE, and
OpenCLIP exclude them: decay would pull scales such as IA3 vectors, SSF scales,
and DoRA magnitudes toward zero instead of toward the identity.
Position embeddings and learned class, register, distillation, and mask tokens
follow timm, DeiT, and MAE fine-tuning; DINOv2/v3 pretraining decays them, so
drop those tags to match it.
"""


def tagged(*tags: str, when: Callable[[Any], bool] | None = None, **kwargs: Any) -> Any:
    """Declare a module field whose leaves carry semantic *tags*.

    Args:
        *tags: Tags added to every leaf below the field.
        when: Optional predicate on the module instance. The tags apply only
            when it returns ``True``.
        **kwargs: Forwarded to :func:`equinox.field`.
    """

    if not tags:
        raise ValueError("tagged() requires at least one tag.")
    metadata = dict(kwargs.pop("metadata", {}))
    metadata[_TAGS_METADATA_KEY] = (frozenset(tags), when)
    return eqx.field(metadata=metadata, **kwargs)


def buffer(*, when: Callable[[Any], bool] | None = None, **kwargs: Any) -> Any:
    """Declare a module field whose arrays are model state but never trained.

    The forward pass must still read the field through
    ``jax.lax.stop_gradient``: the ``buffer`` tag only keeps the field out of
    trainability and weight-decay masks.

    Args:
        when: Optional predicate on the module instance. The field is a buffer
            only when it returns ``True``, for a value that is trainable on
            request.
        **kwargs: Forwarded to :func:`equinox.field`.
    """

    return tagged("buffer", when=when, **kwargs)


def module_tags(tree: PyTree) -> PyTree:
    """Return *tree* with every leaf replaced by the tags its modules give it."""

    return jtu.tree_map_with_path(lambda path, _: _module_tags(tree, path), tree)


def buffer_mask(tree: PyTree) -> PyTree:
    """Return *tree* with every leaf replaced by whether it is a buffer."""

    return jtu.tree_map(lambda tags: "buffer" in tags, module_tags(tree))


def trainable_mask(tree: PyTree) -> PyTree:
    """Return *tree* with every leaf replaced by whether optimizers may update it.

    Trainable leaves are the floating-point arrays that are not buffers. The
    result is an Equinox filter specification, for example
    ``params, static = eqx.partition(model, trainable_mask(model))``.
    """

    return jtu.tree_map(
        lambda leaf, is_buffer: eqx.is_inexact_array(leaf) and not is_buffer,
        tree,
        buffer_mask(tree),
    )


def weight_decay_mask(
    tree: PyTree,
    *,
    no_decay_tags: Iterable[str] = DEFAULT_NO_DECAY_TAGS,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Return *tree* with every leaf replaced by whether weight decay applies.

    A leaf decays when it is trainable and none of its tags is in
    *no_decay_tags*. Evaluate the mask on the tree the optimizer receives; the
    function itself can be passed as a callable mask, as in
    ``optax.adamw(learning_rate, mask=weight_decay_mask)``.
    """

    excluded = frozenset(no_decay_tags)

    def decays(key_path: tuple[Any, ...], leaf: Any) -> bool:
        tags = _module_tags(tree, key_path)
        if not eqx.is_inexact_array(leaf) or "buffer" in tags:
            return False
        tags = tags.union(tagger(key_path_to_path(key_path), leaf))
        return excluded.isdisjoint(tags)

    return jtu.tree_map_with_path(decays, tree)


def _module_tags(root: PyTree, key_path: tuple[Any, ...]) -> frozenset[str]:
    tags: set[str] = set()
    node = root
    declared = False
    for index, key in enumerate(key_path):
        owner = node
        if isinstance(key, jtu.GetAttrKey):
            field_tags = _declared_tags(owner, key.name)
            if field_tags:
                tags.update(field_tags)
                declared = declared or index == len(key_path) - 1
            node = getattr(owner, key.name)
        elif isinstance(key, jtu.SequenceKey):
            node = owner[key.idx]
        elif isinstance(key, jtu.DictKey):
            node = owner[key.key]
        else:
            break
        if index == len(key_path) - 1 and not declared and _is_norm_layer(owner):
            tags.add("norm")
    return frozenset(tags)


def _declared_tags(node: Any, name: str) -> frozenset[str]:
    declaration = _field_declarations(type(node)).get(name)
    if declaration is None:
        return frozenset()
    tags, when = declaration
    if when is not None and not when(node):
        return frozenset()
    return tags


@functools.cache
def _field_declarations(cls: type) -> dict[str, Any]:
    if not dataclasses.is_dataclass(cls):
        return {}
    return {
        field.name: field.metadata[_TAGS_METADATA_KEY]
        for field in dataclasses.fields(cls)
        if _TAGS_METADATA_KEY in field.metadata
    }


def _is_norm_layer(node: Any) -> bool:
    # Imported here because normalization layers declare their own fields
    # with this module.
    from .layers.norm import _NORM_REGISTRY

    return isinstance(node, tuple(_NORM_REGISTRY.values()))


__all__ = (
    "DEFAULT_NO_DECAY_TAGS",
    "buffer",
    "buffer_mask",
    "module_tags",
    "tagged",
    "trainable_mask",
    "weight_decay_mask",
)
