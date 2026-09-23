# Full and Partial Fine-Tuning

Full fine-tuning selects all parameter leaves unless a freeze selector removes
them:

```python
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(
        mode="full",
        freeze=eqft.TargetSpec(tags_any=("embedding.patch",)),
    ),
)
```

Partial fine-tuning selects a depth range. Ranges are half-open:

```python
eqft.TrainableSpec(mode="partial", depth_range=(8, 12))
```

For a model with chunked transformer blocks, use the recipe to count actual
blocks in execution order. This selects exactly the final two blocks and the
new probe head while keeping the backbone's final normalization, embeddings,
and auxiliary tokens frozen:

```python
plan = eqft.partial_ft_last_k_blocks(
    probe,
    k=2,
    train_norm=False,
)
```

The selected blocks' own normalization leaves are included. A request for more
blocks than the model contains raises `ValueError`.
