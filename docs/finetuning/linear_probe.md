# Linear Probe

Linear probing freezes the backbone and trains only a new head.

```python
probe = eqft.make_linear_probe(
    model,
    in_features=384,
    out_features=10,
    key=key,
    pool="cls",
)
plan = eqft.prepare_finetune(
    probe,
    trainable=eqft.TrainableSpec(mode="head"),
)
```

The wrapper replaces the original backbone head with an identity head so
head-only training selects only the probe head.

For ViT-like backbones, `pool="cls_patch_mean"` concatenates the CLS token with
the mean over patch tokens. Prefix/register/distillation tokens are excluded
from the patch mean when model metadata is available, so the probe head input
width is `2 * dim`:

```python
probe = eqft.make_linear_probe(
    model,
    in_features=2 * model.dim,
    out_features=10,
    key=key,
    pool="cls_patch_mean",
)
```

If the transfer setup should train a fresh normalization layer for the
concatenated readout, wrap the probe head with `LayerNormReadoutHead`:

```python
head = eqft.LayerNormReadoutHead(
    2 * model.dim,
    eqft.LinearHead(2 * model.dim, 10, key=key),
)
probe = eqft.make_linear_probe(
    model,
    in_features=2 * model.dim,
    out_features=10,
    key=key,
    pool="cls_patch_mean",
    head=head,
)
```

## Attention-Pooling Probe

`AttentionPoolingClassifierHead` is a FINO/DINOv3-style classifier for token
features. It is different from `AttentionPool`: `AttentionPool` is a lightweight
pooling policy, while `AttentionPoolingClassifierHead` is a trainable head with
an input projection, `LayerNorm`, learned multi-head query, K/V projection,
dropout, and final classifier.

The head consumes one example at a time:

```python
head = eqft.AttentionPoolingClassifierHead(
    in_features=4 * model.dim,
    out_features=10,
    key=key,
    embed_dim=512,
    num_heads=8,
)

# tokens: [num_tokens, 4 * model.dim]
logits = head(tokens, key=key, inference=True)
```

For DINOv3/FINO-style probing over the last `n` transformer blocks, use the
probe wrapper. It calls `intermediate_features(...)`, builds the FINO token
matrix by concatenating patch tokens along the feature axis, and trains only
the probe head under `TrainableSpec(mode="head")`:

```python
probe = eqft.make_attention_pool_probe(
    model,
    in_features=4 * model.dim,
    out_features=10,
    key=key,
    n_last_blocks=4,
    embed_dim=512,
    num_heads=8,
    prepend_cls_token=False,
)

plan = eqft.prepare_finetune(
    probe,
    trainable=eqft.TrainableSpec(mode="head"),
)
```

For a final-layer baseline, omit `n_last_blocks`. The probe then consumes the
model's `forward_features()` dictionary and uses `x_norm_patchtokens`, with an
optional prepended `x_norm_cls_token`:

```python
probe = eqft.make_attention_pool_probe(
    model,
    in_features=model.dim,
    out_features=10,
    key=key,
    prepend_cls_token=False,
)
```

Set `prepend_cls_token=True` to prepend the concatenated CLS token before patch
tokens. Set `l2_normalize_cls=True` to L2-normalize that CLS token before
prepending, matching FINO's optional CLS normalization behavior.
