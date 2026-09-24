"""Content identity for frozen vision-encoder parameters."""

from __future__ import annotations

import hashlib
import json

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np


def encoder_array_digest(encoder: eqx.Module) -> str:
    """Hash every array path, shape, dtype, and value outside traced forwards."""
    digest = hashlib.sha256()
    for path, leaf in jtu.tree_leaves_with_path(encoder):
        if not eqx.is_array(leaf):
            continue
        value = np.asarray(jax.device_get(leaf))
        descriptor = json.dumps(
            [jtu.keystr(path), value.shape, str(value.dtype)], separators=(",", ":")
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()
