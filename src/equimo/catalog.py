"""Experimental discovery for a small set of model variants.

This module is a prototype. Its descriptor and query API may change while
coverage expands beyond the representative vision, audio, and tabular entries.
"""

from dataclasses import dataclass
from difflib import get_close_matches
from importlib import import_module
import re
from typing import Any, Literal


MetadataStatus = Literal["complete", "experimental", "unavailable"]
ShapeDimension = int | str

_CATALOG_PROVIDERS = (
    "equimo.vision.models.vit",
    "equimo.audio.models.ast",
    "equimo.tabular.models.tabpfn",
)
_FIELD_NAMES = ("inputs", "pretrained", "provenance", "notes")
_STATUSES = frozenset(("complete", "experimental", "unavailable"))
_KEY_RE = re.compile(r"^[a-z0-9]+/[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class ModelInput:
    """One argument in a variant's structured input contract."""

    name: str
    shape: tuple[ShapeDimension, ...]
    axes: tuple[str, ...]
    dtype: str
    description: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible input metadata."""
        return {
            "name": self.name,
            "shape": self.shape,
            "axes": self.axes,
            "dtype": self.dtype,
            "description": self.description,
        }


@dataclass(frozen=True)
class PretrainedWeights:
    """Existing checkpoint availability for a variant."""

    available: bool
    identifier: str | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible checkpoint metadata."""
        return {"available": self.available, "identifier": self.identifier}


@dataclass(frozen=True)
class ModelProvenance:
    """Repository-local evidence for conversion and numerical references."""

    conversion: str | None
    reference: str | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible provenance metadata."""
        return {"conversion": self.conversion, "reference": self.reference}


@dataclass(frozen=True)
class ModelVariant:
    """Immutable, serializable metadata for one model configuration.

    Identity fields are complete by contract. ``field_status`` records the
    completeness of the four metadata sections that may be migrated
    incrementally.
    """

    key: str
    modality: str
    family: str
    variant: str
    model_registry_key: str
    constructor: str
    inputs: tuple[ModelInput, ...]
    pretrained: PretrainedWeights
    provenance: ModelProvenance
    notes: tuple[str, ...]
    field_status: tuple[tuple[str, MetadataStatus], ...]

    def to_dict(self) -> dict[str, object]:
        """Return stable data only; no callables, models, or arrays."""
        return {
            "key": self.key,
            "modality": self.modality,
            "family": self.family,
            "variant": self.variant,
            "model_registry_key": self.model_registry_key,
            "constructor": self.constructor,
            "inputs": tuple(item.to_dict() for item in self.inputs),
            "pretrained": self.pretrained.to_dict(),
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
            "field_status": dict(self.field_status),
        }


def list_models(
    *,
    modality: str | None = None,
    family: str | None = None,
    pretrained: bool | None = None,
) -> tuple[ModelVariant, ...]:
    """List covered variants in deterministic key order.

    This experimental catalog is intentionally incomplete. It currently
    covers one representative ViT, AST, and TabPFN variant.
    """
    variants = _load_catalog()
    if modality is not None:
        modality = modality.lower()
        variants = tuple(item for item in variants if item.modality == modality)
    if family is not None:
        family = family.lower()
        variants = tuple(item for item in variants if item.family == family)
    if pretrained is not None:
        variants = tuple(
            item for item in variants if item.pretrained.available is pretrained
        )
    return variants


def model_info(key: str) -> ModelVariant:
    """Return metadata for a full catalog key or unique bare variant name."""
    return _resolve_model(key, _load_catalog())


def create_model(key: str, **kwargs: Any) -> Any:
    """Create a covered variant through its existing modality factory.

    Checkpoint availability is descriptive. Weights are loaded only when the
    caller explicitly passes ``pretrained=True``.
    """
    descriptor = model_info(key)
    module_name, separator, factory_name = descriptor.constructor.rpartition(".")
    if not separator:
        raise RuntimeError(
            f"Invalid constructor path for catalog entry {descriptor.key!r}."
        )
    factory = getattr(import_module(module_name), factory_name)
    return factory(**kwargs)


def _load_catalog() -> tuple[ModelVariant, ...]:
    variants = []
    for module_name in _CATALOG_PROVIDERS:
        module = import_module(module_name)
        provider = getattr(module, "_catalog_model_variants")
        variants.extend(provider())
    return _validate_catalog(tuple(variants))


def _validate_catalog(variants: tuple[ModelVariant, ...]) -> tuple[ModelVariant, ...]:
    seen: dict[str, str] = {}
    for descriptor in variants:
        _validate_descriptor(descriptor)
        normalized_key = descriptor.key.lower()
        if normalized_key in seen:
            raise ValueError(
                f"Duplicate catalog key {descriptor.key!r}; it collides with "
                f"{seen[normalized_key]!r} after normalization."
            )
        seen[normalized_key] = descriptor.key
    return tuple(sorted(variants, key=lambda item: item.key))


def _validate_descriptor(descriptor: ModelVariant) -> None:
    if not _KEY_RE.fullmatch(descriptor.key):
        raise ValueError(
            f"Catalog key {descriptor.key!r} must use lowercase "
            "'<modality>/<variant>' syntax."
        )
    expected_key = f"{descriptor.modality}/{descriptor.variant}"
    if descriptor.key != expected_key:
        raise ValueError(
            f"Catalog key {descriptor.key!r} does not match {expected_key!r}."
        )
    for name in ("modality", "family", "variant", "model_registry_key"):
        value = getattr(descriptor, name)
        if not value or value != value.lower():
            raise ValueError(
                f"Catalog field {name!r} must be non-empty lowercase text."
            )
    if not descriptor.constructor.endswith(f".{descriptor.variant}"):
        raise ValueError(
            f"Constructor {descriptor.constructor!r} does not match variant "
            f"{descriptor.variant!r}."
        )
    if not descriptor.inputs:
        raise ValueError(f"Catalog entry {descriptor.key!r} has no input contract.")
    input_names = set()
    for item in descriptor.inputs:
        if not item.name or not item.dtype or not item.description:
            raise ValueError(
                f"Catalog entry {descriptor.key!r} has incomplete input metadata."
            )
        if len(item.shape) != len(item.axes):
            raise ValueError(
                f"Input {item.name!r} for {descriptor.key!r} has mismatched "
                "shape and axes."
            )
        if item.name in input_names:
            raise ValueError(
                f"Catalog entry {descriptor.key!r} repeats input {item.name!r}."
            )
        input_names.add(item.name)
    if descriptor.pretrained.available != (
        descriptor.pretrained.identifier is not None
    ):
        raise ValueError(
            f"Catalog entry {descriptor.key!r} has inconsistent pretrained metadata."
        )
    statuses = dict(descriptor.field_status)
    if tuple(statuses) != _FIELD_NAMES or len(statuses) != len(descriptor.field_status):
        raise ValueError(
            f"Catalog entry {descriptor.key!r} must declare statuses for "
            f"{list(_FIELD_NAMES)} in that order."
        )
    invalid_statuses = set(statuses.values()) - _STATUSES
    if invalid_statuses:
        raise ValueError(
            f"Catalog entry {descriptor.key!r} has invalid metadata statuses: "
            f"{sorted(invalid_statuses)}."
        )
    if statuses["provenance"] == "complete" and (
        descriptor.provenance.conversion is None
        or descriptor.provenance.reference is None
    ):
        raise ValueError(
            f"Catalog entry {descriptor.key!r} marks incomplete provenance complete."
        )
    if statuses["notes"] == "complete" and not descriptor.notes:
        raise ValueError(
            f"Catalog entry {descriptor.key!r} marks empty notes complete."
        )


def _resolve_model(key: str, variants: tuple[ModelVariant, ...]) -> ModelVariant:
    normalized = key.lower()
    exact = tuple(item for item in variants if item.key == normalized)
    if exact:
        return exact[0]

    bare = tuple(item for item in variants if item.variant == normalized)
    if len(bare) == 1:
        return bare[0]
    if len(bare) > 1:
        choices = ", ".join(item.key for item in bare)
        raise ValueError(f"Ambiguous model variant {key!r}; use one of: {choices}.")

    candidates = tuple(
        sorted({item.key for item in variants} | {item.variant for item in variants})
    )
    matches = get_close_matches(normalized, candidates, n=3, cutoff=0.4)
    suggestions = sorted(
        {
            item.key
            for match in matches
            for item in variants
            if match in (item.key, item.variant)
        }
    )
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(f"Unknown model variant {key!r}.{hint}")


__all__ = [
    "MetadataStatus",
    "ModelInput",
    "ModelProvenance",
    "ModelVariant",
    "PretrainedWeights",
    "create_model",
    "list_models",
    "model_info",
]
