# Layer-Wise Learning-Rate Decay

LLRD labels blocks by original depth. Multipliers live in `GroupSpec`, not in the label string.

For `n` blocks:

```python
lr_mult(i) = decay ** (n - 1 - i)
```

```python
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(
        mode="full",
        freeze=eqft.TargetSpec(tags_any=("embedding.patch",)),
    ),
    labels=eqft.LLRDConfig(decay=0.75),
)
```

Use `eqft.LLRDConfig.uniform()` when a recipe should keep a uniform backbone learning-rate multiplier while still
emitting weight-decay labels.

Patch embedding freeze means `patch_embed` leaves are absent from `plan.trainable`; they do not get a zero learning
rate.

Bias, norm, layer-scale, state-space dynamics (`ssm`), position embedding, and learned class, register, distillation,
and mask token leaves, and every other array with at most one axis (`vector`, such as IA3 vectors or DoRA magnitudes),
are no-weight-decay by default (`equimo.core.parameters.DEFAULT_NO_DECAY_TAGS`, the same policy as
`equimo.core.parameters.weight_decay_mask`). DINOv2/v3-style pretraining decays position embeddings and tokens; pass
`no_weight_decay_tags` without those tags to match it. Buffers, such as frozen PEFT bases, are never trainable and get
no label, in plans and in `make_param_labels`.
