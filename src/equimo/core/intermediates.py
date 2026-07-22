"""Utilities for model intermediate-feature APIs."""

from __future__ import annotations

from collections.abc import Sequence


def intermediate_indices(
    total: int,
    *,
    indices: Sequence[int] | None = None,
    n_last_blocks: int | None = None,
) -> frozenset[int]:
    """Resolve requested intermediate indices to a validated set."""

    if indices is not None and n_last_blocks is not None:
        raise ValueError("indices and n_last_blocks are mutually exclusive.")
    if total < 0:
        raise ValueError("total must be non-negative.")
    if indices is None and n_last_blocks is None:
        return frozenset(range(total))
    if n_last_blocks is not None:
        if n_last_blocks < 1:
            raise ValueError("n_last_blocks must be >= 1.")
        if n_last_blocks > total:
            raise ValueError(
                f"n_last_blocks={n_last_blocks} exceeds available intermediates={total}."
            )
        return frozenset(range(total - n_last_blocks, total))

    assert indices is not None
    resolved = []
    for index in indices:
        index = int(index)
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise ValueError(
                f"intermediate index {index} is out of range for total={total}."
            )
        resolved.append(index)
    return frozenset(resolved)


__all__ = ("intermediate_indices",)
