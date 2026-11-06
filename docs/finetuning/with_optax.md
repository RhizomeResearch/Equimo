# Optax Integration

Optax remains external. Build transforms from `plan.group_specs`:

```python
def make_optax_tx_from_plan(plan, base_schedule, weight_decay=0.05):
    transforms = {}
    for label, group in plan.group_specs.items():
        wd = weight_decay if group.weight_decay else 0.0
        mult = group.lr_multiplier
        transforms[label] = optax.adamw(
            learning_rate=lambda step, mult=mult: base_schedule(step) * mult,
            weight_decay=wd,
        )
    # A function returning the labels: optax would call a callable label tree, such as a model.
    return optax.partition(transforms, lambda _: plan.labels)
```

This code belongs in user code or integration examples, not Equimo core.
