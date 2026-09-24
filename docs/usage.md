# Equimo Usage Guide

This guide covers the non-fine-tuning surface of Equimo: building models, running inference, extracting features, using
modality namespaces, and saving or loading weights. Fine-tuning APIs are documented separately in
[`docs/finetuning`](./finetuning/index.md).

## Import Layout

Equimo is organized by modality:

```python
import equimo.vision.models as vision_models
from equimo.language import TextTransformerEncoder
from equimo.audio.models import AudioSpectrogramTransformer
import equimo.tabular.models as tabular_models
import equimo.timeseries.models as timeseries_models
```

Shared layers live under `equimo.core.layers`; vision-specific layers live under `equimo.vision.layers`; other
modality-specific layers live under their respective namespaces. Generic checkpoint helpers are exported from
`equimo.serialization`. The entire `equimo.timeseries` namespace is experimental.

Pretrained weights have family-specific upstream terms. Before downloading or using them, review the
[pretrained-model license index](../LICENSES/pretrained/README.md), then check the latest upstream license linked there.
Bundled snapshots can become outdated; issues or pull requests updating stale terms are welcome.

## Vision Models

Use constructor functions for published variants, or instantiate the model class directly when experimenting with small
local configurations:

```python
import jax.random as jr
import equimo.vision.models as em

key = jr.PRNGKey(0)
model = em.VisionTransformer(
    img_size=64,
    in_channels=3,
    dim=64,
    patch_size=8,
    num_heads=[2],
    depths=[2],
    num_classes=10,
    key=key,
)

image = jr.normal(key, (3, 64, 64))
logits = model(image, key=key, inference=True)
features = model.features(image, key=key, inference=True)
```

Most vision models accept channel-first arrays shaped `(channels, height, width)`. Pass `num_classes=None` or
`num_classes=0` when you want a feature backbone without a classification head.

## Deterministic Inference and PRNG Keys

Pass the Python value `inference=True` to make deterministic built-in model and layer paths PRNG-free. Equimo continues
to accept a key for API consistency, but reuses it instead of staging `jax.random.split`, `jax.random.fold_in`, or a
fallback key when every stochastic operation on that path is disabled. This is useful for ahead-of-time export systems
such as ONNX converters that do not lower JAX randomness primitives. Training and non-static inference values (`False`
or `None`) retain their existing key schedules exactly.

The branch is intentionally Python-static: passing a traced boolean does not provide this export guarantee. Custom
registered layers remain responsible for their own inference semantics.

`VisionParcae` is the one built-in family whose default inference remains key-dependent. Its default
`state_init="like-init"`, and the `"normal"`, `"embed"`, and `"unit"` alternatives, intentionally sample the recurrent
initial state at inference. Set `state_init="zero"` when deterministic, PRNG-free inference is required:

```python
model = em.VisionParcae(
    # architecture arguments ...
    state_init="zero",
    key=key,
)
logits = model(image, key=key, inference=True)
```

## Text Encoders

`TextTransformerEncoder` operates on token IDs and padding masks. Tokenizers are optional and require the `language`
extra when using `SentencePieceTokenizer`.

```python
import jax.numpy as jnp
import jax.random as jr
from equimo.language import TextTransformerEncoder

key = jr.PRNGKey(0)
model = TextTransformerEncoder(
    dim=16,
    mlp_ratio=2.0,
    depth=2,
    num_heads=2,
    vocab_size=128,
    key=key,
)

token_ids = jnp.array([12, 7, 91, 4, 0, 0])
padding = jnp.array([0, 0, 0, 0, 1, 1])
embedding = model(token_ids, padding, key=key, inference=True)
```

## TabPFN Core Models

TabPFN constructors expose the model core, not a scikit-learn style estimator. `n_train` is a Python integer that
determines the context/test slice boundary and must be static when JIT-compiling a call. `x` and `y` are unbatched JAX
arrays. Classifier variants return log probabilities for test rows; regressor variants return bucket logits.

```python
import jax
import jax.numpy as jnp
import equimo.tabular.models as tm

key = jax.random.PRNGKey(42)
model = tm.tabpfn_v3_classifier_default(pretrained=False)

x = jnp.ones((12, 5))
y = jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
n_train = 8

predict = jax.jit(
    lambda x, y, n_train: model(x, y, n_train, key=key, inference=True),
    static_argnums=2,
)
log_probs = predict(x, y, n_train)
```

The pretrained-model license index includes the current TabPFN-3 terms and the required attribution notice.

## T0 Forecasting

T0 constructors expose the direct patch-transformer forward pass. The `predict()` method also accepts raw contexts
shaped `(T)`, `(batch, T)`, or `(batch, variates, T)` and returns a quantile forecast:

```python
forecast = model.predict(context, horizon=32, quantiles=(0.1, 0.5, 0.9))
forecast.quantiles  # (batch, horizon, quantiles)
```

For direct backbone calls, inputs are unbatched arrays shaped `(variates, time)`:

```python
import jax.numpy as jnp
import jax.random as jr
import equimo.timeseries.models as tm

key = jr.PRNGKey(0)
model = tm.t0_alpha(pretrained=True, key=key)

values_scaled = jnp.zeros((1, 96), dtype=jnp.float32)
mask = jnp.zeros((1, 96), dtype=jnp.int8)
group_ids = jnp.zeros((1, 96), dtype=jnp.int32)
variate_type = jnp.zeros((1, 96), dtype=jnp.int32)

raw_quantiles = model(
    values_scaled,
    mask,
    group_ids,
    variate_type,
    key=key,
    inference=True,
)
```

The direct output has shape `(variates, ceil(time / patch_size), patch_size, quantiles)` and remains in the input's
transformed domain. Use `model.predict(...)` for upstream scaling, inverse scaling, quantile interpolation, forecast
selection, and long-horizon rollout. See the [time-series guide](./timeseries.md) for the mask and variate vocabularies,
grouping rules, padding alignment, feature shape, and full experimental compatibility boundary.

## Serialization

Use `equimo.serialization` for model archives and pretrained weight loading:

```python
from pathlib import Path
from equimo.serialization import inspect_checkpoint, load_weights, save_model

model = load_weights(model, identifier="dinov2_vits14_reg")
checkpoint_path = save_model(
    Path("checkpoint"),
    model,
    model_config={},
    torch_hub_cfg=[],
)
checkpoint = inspect_checkpoint(checkpoint_path, model=model)
assert checkpoint.verified and not checkpoint.legacy
print(checkpoint.weights_sha256)
```

`load_weights` resolves Equimo-hosted identifiers, while `save_model` writes a local archive or directory depending on
the compression option and returns the exact path written. Repeated compressed saves of the same model and metadata with
the same Equimo/JAX/Equinox versions are byte-identical.

`inspect_checkpoint` only accepts local paths and does not deserialize weights. For modern checkpoints, `verified=True`
means the metadata schema and embedded parameter-stream digest were validated; supplying `model=` additionally checks
its class and array-leaf structure. `weights_sha256` identifies the serialized Equinox parameter stream, while a digest
of the complete `.tar.lz4` file identifies its packaging. Schema-less v2-alpha checkpoints require `allow_legacy=True`
for inspection and are always returned with `legacy=True, verified=False`; `load_weights` retains their documented
loading compatibility and emits a warning.

## Registries

Equimo registries let model constructors accept string names for layers and blocks. Layer lookup is scoped by modality:
the core resolver sees only shared core layers, while the vision resolver also sees vision layers and prefers the vision
class when a name exists in both scopes.

```python
from equimo.core.layers import Attention as CoreAttention
from equimo.core.layers import get_layer as get_core_layer
from equimo.registry import get_model_cls
from equimo.vision.layers import Attention as VisionAttention
from equimo.vision.layers import get_layer as get_vision_layer

vit_cls = get_model_cls("vit", modality="vision")
assert get_core_layer("attention") is CoreAttention
assert get_vision_layer("attention") is VisionAttention
```

When the same model name exists in more than one modality, pass `modality=` to avoid ambiguity. For layer names shared
by core and vision, choose the resolver for the intended modality; use a family resolver such as `get_attn` when you
want to target that registry directly.

## Runnable Examples

- [`examples/vision_feature_extraction.py`](../examples/vision_feature_extraction.py)
- [`examples/language_encoder.py`](../examples/language_encoder.py)
- [`examples/finetuning/`](../examples/finetuning)
