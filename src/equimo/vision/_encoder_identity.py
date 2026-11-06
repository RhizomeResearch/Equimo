"""Content identity for frozen vision-encoder parameters."""

from __future__ import annotations

import hashlib
import json

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np

from equimo._legacy_layouts import legacy_leaves_with_path


def encoder_array_digest(encoder: eqx.Module, *, legacy_layout: bool = False) -> str:
    """Hash every array path, shape, dtype, and value outside traced forwards.

    With *legacy_layout*, hash the arrays a pre-2.8 version of *encoder* had,
    including values Equimo now computes (see :mod:`equimo._legacy_layouts`),
    to check digests recorded by older artifacts.
    """
    entries = (
        legacy_leaves_with_path(encoder)
        if legacy_layout
        else jtu.tree_leaves_with_path(encoder)
    )
    digest = hashlib.sha256()
    for path, leaf in entries:
        if not eqx.is_array(leaf):
            continue
        value = np.asarray(jax.device_get(leaf))
        descriptor = json.dumps(
            [jtu.keystr(tuple(path)), value.shape, str(value.dtype)],
            separators=(",", ":"),
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()
