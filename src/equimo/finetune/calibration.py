"""Pure streaming collectors for data-aware fine-tuning statistics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from .config import CalibrationArtifact
from .peft._compat import validate_logical_parameter_schema

CalibrationCollectorKind = Literal[
    "activation_covariance", "activation_svd", "input_covariance"
]
CovarianceNormalization = Literal["sum", "sample_mean"]


class CalibrationCollectorState(eqx.Module):
    """Bounded-memory, PyTree-compatible state for named covariance streams."""

    counts: tuple[jax.Array, ...]
    means: tuple[jax.Array, ...]
    centered_sums: tuple[jax.Array, ...]
    kind: CalibrationCollectorKind = eqx.field(static=True)
    logical_parameter_ids: tuple[str, ...] = eqx.field(static=True)
    feature_dims: tuple[int, ...] = eqx.field(static=True)
    base_checkpoint_hash: str = eqx.field(static=True)
    data_fingerprint: str = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    normalization: CovarianceNormalization = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    rank: int | None = eqx.field(static=True)
    distributed_reduction: str = eqx.field(static=True)


def initialize_calibration_collector(
    *,
    kind: CalibrationCollectorKind,
    logical_parameter_dims: Mapping[str, int] | Iterable[tuple[str, int]],
    base_checkpoint_hash: str,
    data_fingerprint: str,
    centered: bool = False,
    normalization: CovarianceNormalization | None = None,
    accumulation_dtype: str | jnp.dtype = jnp.float32,
    rank: int | None = None,
    distributed_reduction: str = "none",
) -> CalibrationCollectorState:
    """Initialize named streaming statistics without retaining samples.

    Logical IDs are EVA module paths for activation statistics and matrix-weight
    logical IDs for RegMean input statistics. Each observation is one row after
    flattening all leading dimensions of an update value.
    """

    if kind not in {
        "activation_covariance",
        "activation_svd",
        "input_covariance",
    }:
        raise ValueError(f"Unsupported calibration collector kind {kind!r}.")
    logical_ids, feature_dims = validate_logical_parameter_schema(
        logical_parameter_dims
    )
    if not base_checkpoint_hash:
        raise ValueError("Calibration base_checkpoint_hash must be non-empty.")
    if not data_fingerprint:
        raise ValueError("Calibration data_fingerprint must be non-empty.")
    if not distributed_reduction:
        raise ValueError("Calibration distributed_reduction must be non-empty.")
    if kind == "input_covariance" and centered:
        raise ValueError(
            "RegMean input_covariance collectors must be uncentered to produce X.T @ X."
        )
    if normalization is None:
        normalization = "sum" if kind == "input_covariance" else "sample_mean"
    if normalization not in {"sum", "sample_mean"}:
        raise ValueError("Calibration normalization must be 'sum' or 'sample_mean'.")
    if kind == "activation_svd":
        if rank is None or rank < 1:
            raise ValueError("activation_svd collectors require a positive rank.")
        too_narrow = [
            logical_id
            for logical_id, feature_dim in zip(logical_ids, feature_dims, strict=True)
            if rank > feature_dim
        ]
        if too_narrow:
            raise ValueError(
                "activation_svd rank exceeds the feature width for: "
                + ", ".join(too_narrow)
                + "."
            )
    elif rank is not None:
        raise ValueError("rank is only valid for activation_svd collectors.")

    dtype = jnp.dtype(accumulation_dtype)
    if dtype not in {jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)}:
        raise ValueError("Calibration accumulation_dtype must be float32 or float64.")
    if dtype == jnp.dtype(jnp.float64) and not jax.config.read("jax_enable_x64"):
        raise ValueError("float64 accumulation requires JAX_ENABLE_X64=1.")
    return cast(
        CalibrationCollectorState,
        CalibrationCollectorState(
            counts=tuple(jnp.zeros((), dtype=jnp.int32) for _ in logical_ids),
            means=tuple(jnp.zeros((dim,), dtype=dtype) for dim in feature_dims),
            centered_sums=tuple(
                jnp.zeros((dim, dim), dtype=dtype) for dim in feature_dims
            ),
            kind=kind,
            logical_parameter_ids=logical_ids,
            feature_dims=feature_dims,
            base_checkpoint_hash=base_checkpoint_hash,
            data_fingerprint=data_fingerprint,
            centered=centered,
            normalization=normalization,
            accumulation_dtype=str(dtype),
            rank=rank,
            distributed_reduction=distributed_reduction,
        ),
    )


def update_calibration_collector(
    state: CalibrationCollectorState,
    values: Mapping[str, jax.Array],
    *,
    sample_mask: jax.Array | Mapping[str, jax.Array] | None = None,
    sample_count: int | Mapping[str, int] | None = None,
) -> CalibrationCollectorState:
    """Update a state from caller-supplied rows and an explicit mask or count."""

    if (sample_mask is None) == (sample_count is None):
        raise ValueError("Provide exactly one of sample_mask or sample_count.")
    _validate_named_keys(state.logical_parameter_ids, values, label="values")
    if isinstance(sample_mask, Mapping):
        _validate_named_keys(
            state.logical_parameter_ids,
            cast(Mapping[str, jax.Array], sample_mask),
            label="sample_mask",
        )
    if isinstance(sample_count, Mapping):
        _validate_named_keys(
            state.logical_parameter_ids,
            cast(Mapping[str, int], sample_count),
            label="sample_count",
        )

    batch_counts: list[jax.Array] = []
    batch_means: list[jax.Array] = []
    batch_centered_sums: list[jax.Array] = []
    for logical_id, feature_dim, mean in zip(
        state.logical_parameter_ids, state.feature_dims, state.means, strict=True
    ):
        value = jnp.asarray(values[logical_id])
        if value.ndim < 1 or value.shape[-1] != feature_dim:
            raise ValueError(
                f"Calibration value {logical_id!r} must have feature width "
                f"{feature_dim}; got shape {value.shape}."
            )
        matrix = value.reshape((-1, feature_dim)).astype(mean.dtype)
        mask = _resolve_mask(
            logical_id,
            value,
            matrix.shape[0],
            sample_mask=sample_mask,
            sample_count=sample_count,
        )
        batch_count, batch_mean, batch_centered_sum = _masked_batch_state(matrix, mask)
        batch_counts.append(batch_count)
        batch_means.append(batch_mean)
        batch_centered_sums.append(batch_centered_sum)

    batch_state = _replace_dynamic_state(
        state,
        counts=tuple(batch_counts),
        means=tuple(batch_means),
        centered_sums=tuple(batch_centered_sums),
    )
    return combine_calibration_collectors(state, batch_state)


def combine_calibration_collectors(
    left: CalibrationCollectorState,
    right: CalibrationCollectorState,
) -> CalibrationCollectorState:
    """Associatively combine compatible streaming states using Chan's update."""

    _validate_compatible_states(left, right)
    counts: list[jax.Array] = []
    means: list[jax.Array] = []
    centered_sums: list[jax.Array] = []
    for left_count, left_mean, left_sum, right_count, right_mean, right_sum in zip(
        left.counts,
        left.means,
        left.centered_sums,
        right.counts,
        right.means,
        right.centered_sums,
        strict=True,
    ):
        count = left_count + right_count
        count_float = count.astype(left_mean.dtype)
        safe_count = jnp.maximum(count_float, 1)
        left_float = left_count.astype(left_mean.dtype)
        right_float = right_count.astype(left_mean.dtype)
        delta = right_mean - left_mean
        mean = left_mean + delta * (right_float / safe_count)
        correction = jnp.outer(delta, delta) * (left_float * right_float / safe_count)
        centered_sum = left_sum + right_sum + correction
        counts.append(count)
        means.append(jnp.where(count > 0, mean, jnp.zeros_like(mean)))
        centered_sums.append(centered_sum)
    return _replace_dynamic_state(
        left,
        counts=tuple(counts),
        means=tuple(means),
        centered_sums=tuple(centered_sums),
    )


def finalize_calibration_collector(
    state: CalibrationCollectorState,
) -> dict[str, CalibrationArtifact]:
    """Finalize each logical ID into a validated immutable artifact."""

    artifacts: dict[str, CalibrationArtifact] = {}
    for logical_id, count_array, mean, centered_sum in zip(
        state.logical_parameter_ids,
        state.counts,
        state.means,
        state.centered_sums,
        strict=True,
    ):
        count = int(count_array)
        if count == 0:
            raise ValueError(
                f"Cannot finalize calibration collector {logical_id!r}: no samples."
            )
        covariance_sum = centered_sum
        if not state.centered:
            covariance_sum = covariance_sum + count * jnp.outer(mean, mean)
        covariance = (
            covariance_sum / count
            if state.normalization == "sample_mean"
            else covariance_sum
        )
        statistics: dict[str, object] = {
            "mean": mean,
            "centered": state.centered,
            "normalization": state.normalization,
            "matrix_orientation": "features_by_features",
        }
        if state.kind == "activation_svd":
            assert state.rank is not None
            effective_samples = count - int(state.centered)
            if effective_samples < state.rank:
                raise ValueError(
                    f"Cannot finalize activation_svd {logical_id!r}: rank "
                    f"{state.rank} requires at least {state.rank} effective samples; "
                    f"got {effective_samples}."
                )
            eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
            order = jnp.argsort(eigenvalues)[::-1][: state.rank]
            statistics.update(
                right_singular_vectors=eigenvectors[:, order].T,
                singular_values=jnp.sqrt(jnp.maximum(eigenvalues[order], 0)),
            )
        elif state.kind == "activation_covariance":
            statistics["covariance"] = covariance
        else:
            statistics["input_covariance"] = covariance
        artifacts[logical_id] = CalibrationArtifact(
            kind=state.kind,
            base_checkpoint_hash=state.base_checkpoint_hash,
            logical_parameter_ids=(logical_id,),
            statistics=statistics,
            sample_count=count,
            data_fingerprint=state.data_fingerprint,
            accumulation_dtype=state.accumulation_dtype,
            distributed_reduction=state.distributed_reduction,
        )
    validate_calibration_artifacts(artifacts)
    return artifacts


def validate_calibration_artifacts(
    artifacts: Mapping[str, CalibrationArtifact],
) -> None:
    """Validate the identity, provenance, and statistic schema of an artifact set."""

    if not artifacts:
        raise ValueError("Calibration artifact mapping must be non-empty.")
    checkpoint_hashes: set[str] = set()
    fingerprints: set[str] = set()
    kinds: set[str] = set()
    logical_ids: set[str] = set()
    for mapping_id, artifact in artifacts.items():
        if not isinstance(mapping_id, str) or not mapping_id:
            raise ValueError("Calibration artifact mapping keys must be non-empty IDs.")
        if not isinstance(artifact, CalibrationArtifact):
            raise TypeError(
                f"Calibration artifact {mapping_id!r} must be a CalibrationArtifact."
            )
        if artifact.logical_parameter_ids != (mapping_id,):
            raise ValueError(
                f"Calibration artifact {mapping_id!r} does not identify mapping key "
                f"exactly once; got {artifact.logical_parameter_ids!r}."
            )
        if mapping_id in logical_ids:
            raise ValueError(f"Found duplicate logical parameter ID {mapping_id!r}.")
        logical_ids.add(mapping_id)
        if not artifact.base_checkpoint_hash:
            raise ValueError(
                f"Calibration artifact {mapping_id!r} has no checkpoint hash."
            )
        if not artifact.data_fingerprint:
            raise ValueError(f"Calibration artifact {mapping_id!r} has no fingerprint.")
        if artifact.sample_count < 1:
            raise ValueError(
                f"Calibration artifact {mapping_id!r} must have a positive sample count."
            )
        if artifact.accumulation_dtype not in {"float32", "float64"}:
            raise ValueError(
                f"Calibration artifact {mapping_id!r} has unsupported accumulation dtype."
            )
        if not artifact.distributed_reduction:
            raise ValueError(
                f"Calibration artifact {mapping_id!r} has no reduction metadata."
            )
        _validate_statistics(mapping_id, artifact)
        checkpoint_hashes.add(artifact.base_checkpoint_hash)
        fingerprints.add(artifact.data_fingerprint)
        kinds.add(artifact.kind)
    if len(checkpoint_hashes) != 1:
        raise ValueError("Calibration artifacts mix base checkpoint hashes.")
    if len(fingerprints) != 1:
        raise ValueError("Calibration artifacts mix data fingerprints.")
    if len(kinds) != 1:
        raise ValueError("Calibration artifacts mix statistic kinds.")


def input_covariance_from_artifact(artifact: CalibrationArtifact) -> jax.Array:
    """Return a RegMean-compatible ``(input_dim, input_dim)`` Gram matrix."""

    if artifact.kind != "input_covariance":
        raise ValueError(
            "RegMean requires a calibration artifact with kind 'input_covariance'."
        )
    _validate_statistics(
        artifact.logical_parameter_ids[0]
        if artifact.logical_parameter_ids
        else "<missing>",
        artifact,
    )
    return jnp.asarray(artifact.statistics["input_covariance"])


def _resolve_mask(
    logical_id: str,
    value: jax.Array,
    row_count: int,
    *,
    sample_mask: jax.Array | Mapping[str, jax.Array] | None,
    sample_count: int | Mapping[str, int] | None,
) -> jax.Array:
    if sample_count is not None:
        count = (
            cast(Mapping[str, int], sample_count)[logical_id]
            if isinstance(sample_count, Mapping)
            else sample_count
        )
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("Calibration sample_count entries must be integers.")
        if count != row_count:
            raise ValueError(
                "Calibration sample_count must match the flattened row count; "
                f"{logical_id!r} has {row_count} rows and count {count}."
            )
        return jnp.ones((row_count,), dtype=bool)
    assert sample_mask is not None
    selected = (
        cast(Mapping[str, jax.Array], sample_mask)[logical_id]
        if isinstance(sample_mask, Mapping)
        else sample_mask
    )
    mask = jnp.asarray(selected)
    if mask.dtype != jnp.bool_:
        raise TypeError("Calibration sample masks must have boolean dtype.")
    if mask.shape != value.shape[:-1]:
        raise ValueError(
            f"Calibration sample mask for {logical_id!r} must have shape "
            f"{value.shape[:-1]}; got {mask.shape}."
        )
    return mask.reshape((row_count,))


def _masked_batch_state(
    matrix: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    count = jnp.sum(mask, dtype=jnp.int32)
    count_float = count.astype(matrix.dtype)
    safe_count = jnp.maximum(count_float, 1)
    masked = jnp.where(mask[:, None], matrix, 0)
    mean = jnp.sum(masked, axis=0) / safe_count
    differences = jnp.where(mask[:, None], matrix - mean, 0)
    centered_sum = differences.T @ differences
    return count, mean, centered_sum


def _replace_dynamic_state(
    state: CalibrationCollectorState,
    *,
    counts: tuple[jax.Array, ...],
    means: tuple[jax.Array, ...],
    centered_sums: tuple[jax.Array, ...],
) -> CalibrationCollectorState:
    return cast(
        CalibrationCollectorState,
        CalibrationCollectorState(
            counts=counts,
            means=means,
            centered_sums=centered_sums,
            kind=state.kind,
            logical_parameter_ids=state.logical_parameter_ids,
            feature_dims=state.feature_dims,
            base_checkpoint_hash=state.base_checkpoint_hash,
            data_fingerprint=state.data_fingerprint,
            centered=state.centered,
            normalization=state.normalization,
            accumulation_dtype=state.accumulation_dtype,
            rank=state.rank,
            distributed_reduction=state.distributed_reduction,
        ),
    )


def _validate_named_keys(
    expected: tuple[str, ...], values: Mapping[str, object], *, label: str
) -> None:
    missing = sorted(set(expected) - set(values))
    unexpected = sorted(set(values) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Calibration {label} logical IDs mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )


def _validate_compatible_states(
    left: CalibrationCollectorState, right: CalibrationCollectorState
) -> None:
    if left.kind != right.kind:
        raise ValueError("Cannot combine calibration collectors with different kinds.")
    if left.base_checkpoint_hash != right.base_checkpoint_hash:
        raise ValueError(
            "Cannot combine calibration collectors from different checkpoints."
        )
    if left.data_fingerprint != right.data_fingerprint:
        raise ValueError(
            "Cannot combine calibration collectors with different fingerprints."
        )
    if (
        left.logical_parameter_ids != right.logical_parameter_ids
        or left.feature_dims != right.feature_dims
    ):
        raise ValueError(
            "Cannot combine calibration collectors with different schemas."
        )
    if (
        left.centered != right.centered
        or left.normalization != right.normalization
        or left.accumulation_dtype != right.accumulation_dtype
        or left.rank != right.rank
        or left.distributed_reduction != right.distributed_reduction
    ):
        raise ValueError(
            "Cannot combine calibration collectors with different reductions."
        )


def _validate_statistics(logical_id: str, artifact: CalibrationArtifact) -> None:
    statistics = artifact.statistics
    if not isinstance(statistics, Mapping):
        raise ValueError(
            f"Calibration artifact {logical_id!r} statistics must be a mapping."
        )
    expected_key = {
        "activation_covariance": "covariance",
        "activation_svd": "right_singular_vectors",
        "input_covariance": "input_covariance",
    }.get(artifact.kind)
    if expected_key is None or expected_key not in statistics:
        raise ValueError(
            f"Calibration artifact {logical_id!r} has invalid kind or statistic payload."
        )
    if artifact.kind == "activation_svd":
        vectors = jnp.asarray(statistics["right_singular_vectors"])
        singular_values = jnp.asarray(statistics.get("singular_values"))
        if (
            vectors.ndim != 2
            or singular_values.ndim != 1
            or vectors.shape[0] != singular_values.shape[0]
        ):
            raise ValueError(
                f"Calibration artifact {logical_id!r} has incompatible SVD shapes."
            )
        return
    matrix = jnp.asarray(statistics[expected_key])
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Calibration artifact {logical_id!r} covariance must be square."
        )
    if artifact.kind == "input_covariance" and (
        statistics.get("normalization") != "sum"
        or statistics.get("centered") is not False
    ):
        raise ValueError(
            f"RegMean input_covariance artifact {logical_id!r} must be uncentered with sum normalization."
        )


__all__ = (
    "CalibrationCollectorKind",
    "CalibrationCollectorState",
    "CovarianceNormalization",
    "combine_calibration_collectors",
    "finalize_calibration_collector",
    "initialize_calibration_collector",
    "input_covariance_from_artifact",
    "update_calibration_collector",
    "validate_calibration_artifacts",
)
