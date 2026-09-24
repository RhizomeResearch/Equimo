# Plain mask transformer segmentation

`equimo.vision.models.PMT` joins a frozen dynamic-size ViT to a trainable plain mask decoder (`PlainMaskDecoder`). The
Small configuration is the default: 384 channels, six heads, six decoder blocks, four encoder taps at `(2, 5, 8, 11)`,
and 100 queries. `PMTConfig.base()` and `PMTConfig.large()` select the corresponding author widths, heads, and taps;
`PMTConfig.small(num_queries=200)` selects the author Small instance-segmentation query count.

The decoder predicts class logits (including one no-object class) and mask logits. `output.final` is the last
prediction; `output.auxiliary` contains one prediction immediately before each masked-attention block. Images are
single, channel-first arrays already preprocessed for the selected backbone. Use `jax.vmap` across images. Height and
width must be divisible by the encoder patch size.

```python
import jax.random as jr

from equimo.finetune.vision import pmt_head_finetune
from equimo.vision.models import PMT, dinov3_vits16_pretrain_lvd1689m

backbone = dinov3_vits16_pretrain_lvd1689m(pretrained=True)
model = PMT(backbone, num_classes=19, key=jr.PRNGKey(1),
            backbone_id="dinov3_vits16_pretrain_lvd1689m")
plan = pmt_head_finetune(model)
output = model(normalised_image, key=jr.PRNGKey(2))
```

This example initializes a new decoder. Load a trained PMT checkpoint before using its predictions as segmentation
results.

The default lateral norm is GroupNorm. Each level uses `nearest_power_of_2_divisor(dim, norm_max_group)` groups, with a
maximum of 32 by default. GroupNorm normalizes each image independently and needs no model state. The original
implementation uses SyncBatchNorm over images and tokens, including class/register tokens, with epsilon `1e-5`, momentum
`0.1`, affine scale/bias, and an unbiased running variance. A single-host equivalent is available as
`PMTConfig.small(norm_layer="batchnorm")`. Its running statistics depend on real batch examples, and it requires
explicit state. GroupNorm and BatchNorm checkpoints are distinct architectures. `layernorm`, `rmsnorm`, and `none` are
additional lateral variants; `norm_kwargs` passes supported norm options, and `norm_max_group` controls the GroupNorm
limit.

```python
import equinox as eqx
import jax

from equimo.vision.models import PMTConfig

model, state = eqx.nn.make_with_state(PMT)(
    backbone, 19, key=jr.PRNGKey(1),
    config=PMTConfig.small(norm_layer="batchnorm"),
    backbone_id="dinov3_vits16_pretrain_lvd1689m",
)
plan = pmt_head_finetune(model, state=state)
keys = jr.split(jr.PRNGKey(3), image_batch.shape[0])
outputs, state = jax.vmap(
    lambda image, key: model(image, state, inference=False, key=key),
    axis_name="pmt_batch", out_axes=(0, None),
)(image_batch, keys)
```

Each included `image_batch` entry must be an actual independently sampled image. For padded batches, pass
`example_valid=False` for padded rows so BatchNorm excludes them from statistics and counts. Configure additional named
device axes through `norm_kwargs={"axis_name": ("pmt_batch", "devices")}` for synchronized statistics. See
[query-segmentation training](query_training.md) for distributed reductions, empty-batch behavior, and state replay.
Carry the returned state into the next step and save it with the model. Inference uses its stored statistics; GroupNorm
callers can omit the state argument and receive only the output. Passing state explicitly always returns
`(output, state)`, which is useful for a shared call site.

`encode(image)` yields immutable-to-training frozen features before lateral normalization. `decode(features, ...)` can
reuse them for later decoder steps when the encoder, input preprocessing, and view are unchanged. The cache stores all
selected normalized encoder tokens in class/register/patch order, its grid and positional geometry, and a SHA-256 digest
of the encoder arrays plus the declared input view. Decoding rejects features from a different encoder even when its
descriptive `backbone_id` matches. The digest is computed when PMT is constructed; construct a new PMT if the encoder
arrays are replaced or cast. `pmt_head_finetune` selects every decoder and lateral parameter while excluding the entire
encoder, including its final norm, position components, and tokens. Matching and losses are available through
`equimo.vision.query_training`; the caller supplies target sets and the training transaction. `encoder_features(image)`
returns the final normalized frozen encoder tap in class/register/patch order. The existing `features(image)` endpoint
returns the same array and accepts unused `key` and `inference` arguments for the library's generic feature interface.
It does not contain decoder query tokens. For code that accepts either PMT or EoMT predictions, annotate the result with
`equimo.vision.segmentation.QuerySegmentationOutput`.

For either Small backbone above with three output classes, the native `pmt_head_finetune` plan reports 29,204,756 total
and 7,603,588 trainable parameters. GroupNorm has zero model-state bytes; BatchNorm has 12,304 bytes of running state
across four taps. Two float32 optimizer slots for the trainable leaves require 60,828,704 bytes, excluding gradients,
step counters, and temporary activations. Recompute these four values when class count, precision, norm, or decoder
configuration changes.

For LingBot-Vision Small, use `lingbot_vits16(pretrained=True)` as the backbone and a distinct `backbone_id`. Its own
position and feature definitions are retained. A decoder trained on one backbone is a new transfer experiment on the
other, even when width and taps match.

```python
from equimo.vision.pmt_checkpoint import (
    load_pmt_checkpoint, load_pmt_decoder,
    save_pmt_checkpoint, save_pmt_decoder,
)

save_pmt_checkpoint(full_path, model, state=state, mask_state=mask_state)
save_pmt_decoder(decoder_path, model, state=state, mask_state=mask_state)
restored, state, mask_state = load_pmt_checkpoint(full_path, template,
                                                   state=template_state)
restored, state, mask_state = load_pmt_decoder(decoder_path, matching_template,
                                                state=matching_state)
```

Full archives contain the encoder, decoder, normalization state, and mask schedule state. Decoder-only archives are
bound to the exact encoder array digest, feature and positional configuration, input view, and ordered class ontology.
Both use the bounded native checkpoint reader; pass `limits=` when loading under a stricter local budget. Give the model
a descriptive `backbone_id` and `class_ontology` before saving a decoder-only archive.

`anneal_pmt_mask_state(step, start_steps, end_steps)` computes per-block polynomial mask-retention probabilities. Pass
the result as `mask_state` during training, and save it with the decoder. Once every probability is zero,
`mask_free_pmt(model, mask_state)` disables intermediate prediction and masking for evaluation. This changes the
evaluation path and should be validated for the trained checkpoint.

The tiny numerical fixture in `tests/data/pmt_tiny_reference.npz` was generated from the
[PMT authors' implementation](https://github.com/tue-mps/pmt) at revision `0e803722aa5737a242b383dec1b90c2c66b86baa`.
The generator is `models/qualify_pmt.py`. `tests/data/pmt_dinov3_taps_reference.npz` holds all four normalized taps from
the pinned official DINOv3 Small checkpoint; its generator is `models/qualify_pmt_dinov3.py`. Their hashes and schemas
are in `tests/data/reference_provenance.json`. The author [paper](https://arxiv.org/abs/2603.25398) describes the
architecture.
