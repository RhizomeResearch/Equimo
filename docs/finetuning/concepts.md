# Concepts

Equimo fine-tuning is model surgery plus PyTree planning.

Equinox does not use PyTorch-style `requires_grad`. A leaf is trainable when it is present in the optimizer parameter
PyTree:

```python
plan = eqft.prepare_finetune(model, trainable=eqft.TrainableSpec(mode="head"))
trainable = plan.trainable
frozen = plan.frozen
model = plan.combine(trainable)
```

Frozen leaves are absent from `plan.trainable`. For example, frozen patch embedding parameters are not assigned a zero
learning rate; they are removed from optimizer state entirely.

`FineTunePlan` contains:

- `trainable`: parameters passed to the optimizer.
- `frozen`: parameters kept out of the optimizer.
- `labels`: optimizer-group labels matching trainable leaves.
- `group_specs`: learning-rate multiplier and weight-decay metadata.
- `param_info`: per-leaf path, tags, labels, and trainability.
- `report`: parameter counts, selected target paths, per-leaf metadata, a parameter signature, and a plan fingerprint.

`plan.report.to_dict()` produces a JSON-compatible inventory of logical IDs, shapes, dtypes, roles, tags, selected
leaves, and effective optimizer groups. Save its `plan_fingerprint` with the optimizer checkpoint. After rebuilding a
plan from a restored model, call `eqft.validate_plan(plan, expected_fingerprint=saved_fingerprint)` before restoring
optimizer state. The fingerprint covers structure and group policy; ordinary parameter updates do not change it.

`GroupSpec.role` and `GroupSpec.tags` preserve the first leaf's representative metadata for compatibility. Use
`GroupSpec.roles`, `GroupSpec.tags_all`, and `GroupSpec.mixed_roles` when external tooling needs deterministic metadata
for all leaves assigned to a label.

Non-goals: optimizers, schedules, distributed training, dataloaders, feature caches, experiment tracking, and training
loops.
