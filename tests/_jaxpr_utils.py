"""Compatibility import for JAXPR assertions used by root tests."""

from finetune._jaxpr_utils import (  # noqa: F401
    assert_prng_free_jaxpr,
    primitive_names,
    prng_primitive_names,
)
