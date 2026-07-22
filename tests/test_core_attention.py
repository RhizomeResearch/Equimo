import jax.numpy as jnp
import jax.random as jr

from equimo.core.layers.attention import AttentionBlock


KEY = jr.PRNGKey(0)


def test_attention_block_split_masks_preserve_common_mask_behavior():
    block = AttentionBlock(dim=8, num_heads=2, key=KEY)
    x = jr.normal(KEY, (4, 8))
    mask = jnp.array([[1], [1], [0], [0]])

    common_output = block(x, mask=mask, key=KEY, inference=True)
    split_output = block(
        x,
        attn_mask=mask,
        ffn_mask=mask,
        key=KEY,
        inference=True,
    )

    assert jnp.allclose(common_output, split_output)


def test_attention_block_accepts_distinct_attention_and_ffn_masks():
    block = AttentionBlock(dim=8, num_heads=2, key=KEY)
    x = jr.normal(KEY, (4, 8))
    attn_mask = jnp.array([[[1, 1, 0, 0]]])
    ffn_mask = jnp.array([[1], [1], [0], [0]])

    output = block(
        x,
        attn_mask=attn_mask,
        ffn_mask=ffn_mask,
        key=KEY,
        inference=True,
    )

    assert output.shape == x.shape
    assert jnp.all(jnp.isfinite(output))
