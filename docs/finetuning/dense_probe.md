# Spatial Linear Probe

`equimo.finetune.vision.make_dense_probe` composes an explicit spatial feature
contract with a pointwise `LinearHead`. One call consumes one image and returns
raw class-first logits shaped `(classes, grid_h, grid_w)`. Map the probe with
`jax.vmap` to obtain `(batch, classes, grid_h, grid_w)`.

```python
import jax
import jax.random as jr

import equimo.finetune as eqft


spatial_features = eqft.FeatureSpec(
    endpoint="forward_features",
    output_layout="BNC",
    token_selection="patches",
    pooling=None,
    return_metadata=True,
)
probe = eqft.vision.make_dense_probe(
    model,
    in_features=384,
    out_features=3,
    key=jr.PRNGKey(0),
    feature_spec=spatial_features,
)

logits = probe(image)
batch_logits = jax.vmap(probe)(images)
```

The feature specification is required and must describe either row-major `BNC`
patch tokens or all features from a `BCHW` endpoint. It must be unpooled and set
`return_metadata=True`. Token outputs are reshaped only from the published
`grid_size`; the probe never infers a square grid from the token count. Prefix
tokens are removed by the feature contract before projection. Channel-first
feature maps are moved to a last-axis linear projection and returned in
class-first order.

The constructor rejects absent geometry, multiple separate feature levels,
feature/head width mismatches, and a head whose input or output width differs
from the declared configuration. It performs no spatial interpolation or loss
calculation.

## Trainability

The wrapper owns separate `backbone` and `head` fields, so the usual head-only
plan applies:

```python
head_only = eqft.prepare_finetune(
    probe,
    trainable=eqft.TrainableSpec(mode="head"),
)
```

Select transformer blocks by their explicit paths when exact block membership
is part of the recipe:

```python
head_and_blocks = eqft.prepare_finetune(
    probe,
    trainable=eqft.TrainableSpec(
        mode="surgical",
        target=eqft.TargetSpec(
            include=(
                "head",
                "backbone.blocks.0.blocks.10",
                "backbone.blocks.0.blocks.11",
            )
        ),
    ),
)
```

## Full-model checkpoints

Use the ordinary full-model checkpoint API for a probe and a partially tuned
backbone. Record the wrapper, complete backbone constructor configuration,
feature specification, head dimensions and initialization, and output layout
in `model_config`:

```python
from dataclasses import asdict

from equimo.serialization import load_weights, save_model


model_config = {
    "wrapper": "dense_probe",
    "backbone": backbone_config,
    "base_checkpoint_sha256": base_checkpoint_sha256,
    "feature_spec": asdict(spatial_features),
    "head": {
        "in_features": 384,
        "out_features": 3,
        "bias": True,
        "weight_init": "trunc_normal_0.02",
        "bias_init": 0.0,
    },
    "output_layout": "CHW",
    "trainability_report": head_and_blocks.report.to_dict(),
    "evaluated_parameter_view": "optimizer",
}
save_model(path, probe, model_config)

# Construct the same wrapper and static configuration before loading arrays.
restored = load_weights(
    probe_template,
    path=path,
    expected_model_config=model_config,
)
```

`expected_model_config` rejects a checkpoint recorded for different head
dimensions, feature extraction, trainability plan, parameter view, or output
layout. Validate the restored plan against the recorded fingerprint before an
optimizer owns its parameters. See
[Serialization](serialization.md) for integrity and archive behavior. A
complete offline example is available in
`examples/finetuning/dense_probe_vit.py`.
