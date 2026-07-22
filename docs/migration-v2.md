# Migrating to Equimo 2.0

Equimo 2.0 establishes the first stable public contract. It intentionally does
not preserve the v1 package layout or v1 checkpoint-loading API. Existing model
archives produced and uploaded for the 2.0 alpha series remain supported.

## Runtime support

Equimo 2.0 supports CPython 3.12, 3.13, and 3.14. Core installs require JAX
0.8.1 or newer and Equinox 0.13.3 or newer. The `language` tokenizer extra is
tested on Python 3.12 and 3.13 because its TensorFlow dependencies do not publish
Python 3.14 wheels; the other extras are tested across the full core matrix.

## Import paths

Replace the removed v1 entrypoints with the owning modality or shared-core
namespace:

| v1 import | v2 import |
| --- | --- |
| `equimo.models` | `equimo.vision.models` |
| `equimo.layers` | `equimo.vision.layers` or `equimo.core.layers` |
| `equimo.io.save_model` | `equimo.serialization.save_model` |
| `equimo.io.load_weights` | `equimo.serialization.load_weights` |
| `equimo.io.load_image` | `equimo.vision.io.load_image` |
| `equimo.experimental.text` | `equimo.language` |

There are no compatibility aliases for the removed v1 modules.

## Loading model weights

Construct the architecture with a modality factory, then load leaves into that
model:

```python
from pathlib import Path

from equimo.serialization import load_weights
from equimo.vision.models import dinov2_vits14_reg

model = dinov2_vits14_reg(pretrained=False, dynamic_img_size=True)
model = load_weights(model, path=Path("checkpoint.tar.lz4"))
```

The v1 `load_model(cls, ...)` API is removed. Equimo 2.0 does not promise to
reconstruct v1 archives from embedded constructor metadata. Re-convert those
weights into a v2 model when necessary.

New v2 checkpoints contain a versioned format marker, a model-structure
signature, and a weights checksum. Schema-less archives already uploaded for
the v2 alpha releases take a compatibility path and still load without a
re-upload. The built-in remote points to an immutable repository revision.
Its uploaded model archives are checked against digests embedded from that
revision's Git LFS pointers.

Compressed extraction caches now use `<archive>.extracted`, for example
`model.tar.lz4.extracted`. Older alpha releases used the natural sibling name
`model`; v2 leaves that path untouched because it may belong to the caller.

## Fine-tuning bundles

All supported LoRA-family methods now round-trip through `save_delta` and
`load_delta`, including AdaLoRA, LoRA-FA, RandLoRA, and FourierFT. When ordinary
parameters such as a head or base weight are trainable, pass the original
`base_model` to `save_delta` so their additive changes can be recorded.

Stable path strings remain dot-separated for ordinary attribute/index paths.
Mapping keys containing dots or backslashes are escaped, and integer-looking
string keys use a `\s` prefix so `"0"` remains distinct from index `0`. Use
`path_to_str` and `str_to_path`; do not parse persisted paths by splitting on
dots.

New bundles and calibration artifacts include an array checksum. Bundles from
the v2 alpha series without that field remain readable.

## Numerical changes

Several corrections intentionally change results relative to the alpha builds:

- state-space duality scans now use the correct segmented recurrence and chunk
  state transitions;
- DINO heads no longer apply an activation after their final projection;
- odd-sized convolutional patch embeddings report the convolution's real
  output grid;
- token ID zero is retained when it is an unknown token rather than padding;
- bfloat16 language features preserve their public dtype.

Re-run numerical baselines that depended on the affected alpha behavior. Model
parameter layouts remain unchanged for these fixes.

## Release checklist for downstream projects

1. Update imports to the modality layout.
2. Replace `load_model` calls with a factory plus `load_weights`.
3. Re-run reference outputs for the corrected numerical paths.
4. Re-save long-lived local checkpoints or fine-tuning bundles to gain embedded
   checksums; existing v2-alpha model archives can remain as-is.
5. Run the downstream suite on a supported Python/JAX/Equinox combination.

The exact compatibility boundary is defined in the
[stability policy](stability.md).
