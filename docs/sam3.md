# SAM 3.1

Equimo implements SAM 3.1 in JAX and Equinox: open-vocabulary segmentation from a text prompt or box exemplars,
interactive segmentation from points, boxes or masks, and forward-only video tracking. Every model endpoint is unbatched
and takes the usual `key=None, inference=True`; batch images with `jax.vmap`.

The image endpoints, the text model and the tokenizer are stable. The video path (`Sam3VideoTracker`, `TrackerState`,
`CommitPlan`, `propagate`, `associate`, `commit`) is experimental and may change in a minor release; it has not yet been
compared with the reference on real footage.

## Weights and license

```bash
pip install "equimo[clip]"  # the tokenizer needs ftfy and regex
```

`sam31_multiplex(pretrained=True)` downloads the vision model (2.1 GB) and `sam31_text(pretrained=True)` the text model
(1.4 GB). Both are float32; cast them to bfloat16 after loading, as shown below.

Equimo's code is under `MIT AND Apache-2.0`. The weights are SAM Materials under the
[SAM License](../LICENSES/pretrained/sam-License.md), which applies to anyone who uses them. Among other terms, it
requires a copy of the license with any redistribution (section 1.b.i), an acknowledgement of the use of SAM Materials
in any publication of results (section 1.b.ii), and compliance with trade controls, including the exclusion of uses
subject to the International Traffic in Arms Regulations and of the prohibited end uses it lists (sections 1.b.iii and
1.b.v). Read the license before using the weights; this summary is not legal advice.

## Preparing an image

The model takes `[3, 1008, 1008]`, resized without keeping the aspect ratio and normalized with mean and standard
deviation 0.5. Equimo does not preprocess images; use your data pipeline. The output below is produced at the original
image size, so results map back to the source pixels even though the aspect ratio changed.

```python
import jax.numpy as jnp
from PIL import Image

source = Image.open("image.png").convert("RGB")
resized = source.resize((1008, 1008))
pixels = jnp.asarray(resized, jnp.float32) / 255.0
image = jnp.moveaxis((pixels - 0.5) / 0.5, -1, 0)
output_size = (source.height, source.width)
```

## Finding every instance of a concept

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from equimo.language import ClipBpeTokenizer, sam31_text
from equimo.vision.models.sam3 import ConceptPrompt, decode_concepts, sam31_multiplex

def to_bfloat16(tree):
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


vision = to_bfloat16(sam31_multiplex(pretrained=True))
text = to_bfloat16(sam31_text(pretrained=True))
image = image.astype(jnp.bfloat16)
tokenizer = ClipBpeTokenizer()

token_ids, padding_mask = tokenizer.encode(["yellow school bus"])
token_ids = jnp.asarray(token_ids[0], jnp.int32)
padding_mask = jnp.asarray(padding_mask[0])


@eqx.filter_jit
def detect(vision, text, image, token_ids, padding_mask):
    features = vision.encode_image(image)
    prompt = ConceptPrompt.create(
        text_features=text.features(token_ids, padding_mask),
        text_padding_mask=padding_mask,
    )
    return vision.detect(features, prompt)


output = detect(vision, text, image, token_ids, padding_mask)
detections = decode_concepts(
    output, output_size=output_size, max_detections=64, threshold=0.5
)

for index in range(int(detections.keep.sum())):
    mask = detections.masks[index]  # boolean, at output_size
    box = detections.boxes[index]  # pixels, x1 y1 x2 y2
    score = detections.scores[index]
```

`detections` has a fixed shape, with entries ordered by decreasing score. `keep` marks the entries that hold a
detection, and `presence` is the probability that the concept appears at all; each score already includes it.
`decode_concepts` compiles once per `output_size`, and its memory scales with `max_detections`.

Text is limited to 32 tokens, including the start and end tokens. Longer text is truncated with the end token kept.

### Box exemplars

`ConceptPrompt.create` also takes box exemplars in normalized `cx, cy, w, h`, with label 1 for a positive box and 0 for
a negative one, and exemplar points in normalized `x, y`.

Without text, the reference encodes a fixed placeholder string as the text prompt, and the two results differ
substantially, so pass its features when using exemplars alone:

```python
ids, padding = tokenizer.encode(["<text placeholder>"])
ids, padding = jnp.asarray(ids[0], jnp.int32), jnp.asarray(padding[0])

exemplar = jnp.array([[0.5, 0.5, 0.2, 0.3]], jnp.float32)
features = eqx.filter_jit(vision.encode_image)(image)
prompt = ConceptPrompt.create(
    text_features=text.features(ids, padding),
    text_padding_mask=padding,
    boxes=exemplar,
    box_labels=jnp.ones(1, jnp.int32),
)
output = eqx.filter_jit(vision.detect)(features, prompt)
```

## Clicking on an object

The interactive head works from the features `encode_image` returned. Points and boxes are in input pixels, that is on
the 1008 x 1008 image; point labels are 1 for positive and 0 for negative.

```python
from equimo.vision.models.sam3 import InteractivePrompt
from equimo.vision.ops import resize_bilinear

features = eqx.filter_jit(vision.encode_image)(image)
prompt = InteractivePrompt.create(
    points=jnp.array([[504.0, 504.0]], jnp.float32),
    point_labels=jnp.array([1], jnp.int32),
)
output = eqx.filter_jit(vision.predict_interactive)(features, prompt)

logits = output.mask_logits[output.selected]  # 288 x 288
mask = resize_bilinear(logits, output_size) > 0.0
```

`InteractivePrompt.create` also takes `box` (input pixels, `x1 y1 x2 y2`) and `mask_input`, the `288 x 288` logits of a
previous prediction. `mask_logits` holds the single-mask output first and the three multimask outputs after it. With at
most one label (counting a box's two corners) and no mask input, `selected` is the best multimask output by predicted
IoU; otherwise it is the single-mask output unless its stability score is below 0.98.

## Low precision and memory

The examples above cast the models and the image to bfloat16 after loading, the supported mode on a GPU. Float32 at 1008
x 1008 with XLA attention can run out of memory on a 16 GB GPU, because its attention logits are float32 either way;
float32 is for numerical comparisons on smaller inputs or the CPU.

Attention, position rotation, layer norms and mask logits still run in float32, so bfloat16 detections stay close to
float32 detections. Image inference fits comfortably on a 16 GB GPU; fine-tuning LoRA through the trunk needs
`encode_image(x, checkpoint_trunk=True)` with XLA attention. Measurements are in [the performance notes](performance.md)
under "SAM 3.1 image inference and fine-tuning".

## Video

`Sam3VideoTracker` runs detection, tracking and the association policy on frames you decoded and preprocessed. It is
experimental: the policy thresholds come from the reference, but the driver has not yet been compared with it on real
footage.

```python
from equimo.vision.models.sam3.video import Sam3VideoTracker

# `frames` are [3, 1008, 1008] arrays, preprocessed as above.
num_frames = len(frames)
tracker = Sam3VideoTracker(
    vision, num_frames=num_frames, source_size=output_size, max_objects=16
)
tracker.set_prompt(
    ConceptPrompt.create(
        text_features=text.features(token_ids, padding_mask),
        text_padding_mask=padding_mask,
    )
)

predictions = []
for frame in frames:
    predictions += tracker.push_frame(frame)
predictions += tracker.finish()

for prediction in predictions:
    prediction.frame_index, prediction.object_ids, prediction.masks, prediction.boxes
```

`push_frame` returns the frames that became final. Decisions need two frames of look-ahead, so from the third frame on
each call returns the frame two steps earlier, and `finish` returns the rest. `set_prompt` must be called before the
first frame and raises afterwards.

Objects occupy 16-slot buckets, and `max_objects` (default 16) caps how many are tracked. The device code compiles once
per bucket count, which grows in powers of two: when objects no longer fit in the current buckets, the next `push_frame`
recompiles. Set `max_objects` above 16 to allow that growth.

To add an object the detector missed, call `add_object` with a frame, its index and positive and negative points:

```python
track = tracker.add_object(
    frame_index=40,
    frame=frames[40],
    points=jnp.array([[504.0, 504.0], [200.0, 200.0]], jnp.float32),
    labels=jnp.array([1, 0], jnp.int32),
)
track.first_prediction  # the object on frame 40
next_prediction = track.push(frames[41])  # the object on the next frame
```

The new object runs on its own state, so it is not added to the output of `push_frame`; frames already returned do not
change. Refining an existing object with points, reverse propagation and a new prompt after the first frame are not
supported.
