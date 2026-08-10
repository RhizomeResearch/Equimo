"""Shared factory for building registered model variants."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import cast

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray


def build_model_variant[M: eqx.Module](
    model_cls: Callable[..., M],
    registry: Mapping[str, tuple[dict, dict]],
    variant: str,
    *,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    deepcopy_cfg: bool = False,
    pretrained_variants: frozenset[str] | None = None,
    pretrained_identifiers: Mapping[str, str] | None = None,
    pretrained_label: str | None = None,
    allow_pretrained_overrides: bool = True,
    **overrides,
) -> M:
    """Build ``model_cls`` from a variant registry, optionally loading weights.

    Args:
        model_cls: Model class (or factory callable) accepting the merged
            config plus a ``key`` keyword.
        registry: Mapping of variant name to ``(base_cfg, variant_cfg)``.
        variant: A key in ``registry``.
        pretrained: If ``True``, download and deserialise the pretrained
            checkpoint from the default repository.
        inference_mode: Passed to :func:`equimo.serialization.load_weights`
            when ``pretrained`` is ``True``.
        key: PRNG key for parameter initialisation. Defaults to
            ``PRNGKey(42)`` when ``None``.
        deepcopy_cfg: Deep-copy the merged config before construction, for
            models that mutate nested config containers.
        pretrained_variants: When given, restrict ``pretrained=True`` to this
            set and raise a ``ValueError`` for anything else.
        pretrained_identifiers: Optional variant-to-checkpoint-identifier
            overrides; defaults to the variant name itself.
        pretrained_label: Human-readable family name used in the
            unsupported-pretrained error message.
        allow_pretrained_overrides: Whether configuration overrides are accepted
            when ``pretrained`` is ``True``.
        **overrides: Extra keyword arguments merged into the model config,
            overriding stored defaults (e.g. ``num_classes=10``).

    Raises:
        KeyError: If ``variant`` is not found in ``registry``.
        ValueError: If pretrained weights are unavailable for ``variant`` or
            configuration overrides are disabled for pretrained models.
    """

    if variant not in registry:
        raise KeyError(f"Unknown model variant: {variant!r}.")

    if pretrained:
        if pretrained_variants is not None and variant not in pretrained_variants:
            supported = ", ".join(sorted(pretrained_variants))
            label = pretrained_label or "model"
            raise ValueError(
                f"No pretrained weights are available for {variant!r}. "
                f"Supported {label} pretrained variants: {supported}."
            )
        if not allow_pretrained_overrides and overrides:
            names = ", ".join(sorted(overrides))
            label = pretrained_label or "model"
            raise ValueError(
                f"Pretrained {label} variants do not accept configuration "
                f"overrides; got: {names}."
            )

    if key is None:
        key = jax.random.PRNGKey(42)
    base_cfg, variant_cfg = registry[variant]
    cfg = base_cfg | variant_cfg | overrides
    if deepcopy_cfg:
        cfg = copy.deepcopy(cfg)
    model = model_cls(**cfg, key=key)

    if pretrained:
        from equimo.serialization import load_weights

        identifier = (pretrained_identifiers or {}).get(variant, variant)
        model = cast(
            "M",
            load_weights(
                model,
                identifier=identifier,
                inference_mode=inference_mode,
            ),
        )

    return model
