# Equimo Usage Guide

This guide covers the non-fine-tuning surface of Equimo: building models,
running inference, extracting features, using modality namespaces, and saving or
loading weights. Fine-tuning APIs are documented separately in
[`docs/finetuning`](./finetuning/index.md).

## Import Layout

Equimo is organized by modality:

```python
import equimo.vision.models as vision_models
from equimo.language import TextTransformerEncoder
from equimo.audio.models import AudioSpectrogramTransformer
import equimo.tabular.models as tabular_models
```

Shared layers live under `equimo.core.layers`; vision-specific layers live under
`equimo.vision.layers`. Generic checkpoint helpers are exported from
`equimo.serialization`.

## Vision Models

Use constructor functions for published variants, or instantiate the model class
directly when experimenting with small local configurations:

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

Most vision models accept channel-first arrays shaped `(channels, height,
width)`. Pass `num_classes=None` or `num_classes=0` when you want a feature
backbone without a classification head.

## Text Encoders

`TextTransformerEncoder` operates on token IDs and padding masks. Tokenizers are
optional and require the `language` extra when using `SentencePieceTokenizer`.

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

TabPFN constructors expose the model core, not a scikit-learn style estimator.
`n_train` is a Python integer that determines the context/test slice boundary
and must be static when JIT-compiling a call. `x` and `y` are unbatched JAX
arrays. Classifier variants return log probabilities for test rows; regressor
variants return bucket logits.

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

Use pretrained TabPFN weights only after reviewing the upstream TabPFN-3 license.

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

`load_weights` resolves Equimo-hosted identifiers, while `save_model` writes a
local archive or directory depending on the compression option and returns the
exact path written. Repeated compressed saves of the same model and metadata
with the same Equimo/JAX/Equinox versions are byte-identical.

`inspect_checkpoint` only accepts local paths and does not deserialize weights.
For modern checkpoints, `verified=True` means the metadata schema and embedded
parameter-stream digest were validated; supplying `model=` additionally checks
its class and array-leaf structure. `weights_sha256` identifies the serialized
Equinox parameter stream, while a digest of the complete `.tar.lz4` file
identifies its packaging. Schema-less v2-alpha checkpoints require
`allow_legacy=True` for inspection and are always returned with
`legacy=True, verified=False`; `load_weights` retains their documented loading
compatibility and emits a warning.

## Registries

Equimo registries let model constructors accept string names for layers and
blocks. Layer lookup is scoped by modality: the core resolver sees only shared
core layers, while the vision resolver also sees vision layers and prefers the
vision class when a name exists in both scopes.

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

When the same model name exists in more than one modality, pass `modality=` to
avoid ambiguity. For layer names shared by core and vision, choose the resolver
for the intended modality; use a family resolver such as `get_attn` when you
want to target that registry directly.

## Runnable Examples

- [`examples/vision_feature_extraction.py`](../examples/vision_feature_extraction.py)
- [`examples/language_encoder.py`](../examples/language_encoder.py)
- [`examples/finetuning/`](../examples/finetuning)
