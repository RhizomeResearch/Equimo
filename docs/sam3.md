# SAM 3.1

Equimo implements SAM 3.1 in JAX and Equinox: open-vocabulary segmentation from a text prompt or box exemplars,
interactive segmentation from points, boxes or masks, and forward-only video tracking. Every model endpoint is unbatched
and takes the usual `key=None, inference=True`; batch images with `jax.vmap`.

The image endpoints, the text model and the tokenizer are stable. The video path (`Sam3VideoTracker`, `TrackerState`,
`CommitPlan`, `propagate`, `associate`, `commit`) is experimental and may change in a minor release.

![SAM 3.1 tracking the concept "person" through a film scene](https://rhizome-labs.com/equimo/sam31-video-tracking.gif)

*SAM 3.1 weights tracking the text prompt `"person"`, with [this code](#tracking-a-film-clip). Excerpt from* Pulp
Fiction *(1994, Miramax), which belongs to its rights holders and is not covered by Equimo's licenses.*

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
from equimo.vision.models.sam3 import decode_concepts, encode_concept, sam31_multiplex

def to_bfloat16(tree):
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


vision = to_bfloat16(sam31_multiplex(pretrained=True))
text = to_bfloat16(sam31_text(pretrained=True))
image = image.astype(jnp.bfloat16)
tokenizer = ClipBpeTokenizer()

prompt = encode_concept(text, tokenizer, "yellow school bus")
features = eqx.filter_jit(vision.encode_image)(image)
output = eqx.filter_jit(vision.detect)(features, prompt)
detections = decode_concepts(output, output_size=output_size, threshold=0.5)

for index in range(int(detections.keep.sum())):
    mask = detections.masks[index]  # boolean, at output_size
    box = detections.boxes[index]  # pixels, x1 y1 x2 y2
    score = detections.scores[index]
```

`detections` has a fixed shape, with entries ordered by decreasing score. `keep` marks the entries that hold a
detection, and `presence` is the probability that the concept appears at all; each score already includes it.
`decode_concepts` compiles once per `output_size` and keeps all 200 queries by default. Its memory scales with
`max_detections`, so pass a cap such as `max_detections=64` for large output sizes; lower-scoring detections beyond the
cap are dropped.

Text is limited to 32 tokens, including the start and end tokens. Longer text is truncated with the end token kept.

### Box exemplars

`encode_concept` also takes box exemplars, with label 1 for a positive box and 0 for a negative one, and exemplar
points. Without `source_size`, boxes are normalized `cx, cy, w, h` and points normalized `x, y`. With `source_size`,
boxes are `x1 y1 x2 y2` and points `x y` in those original pixels. Labels default to positive.

With exemplars and no concept, `encode_concept` encodes the placeholder text the reference uses, because an empty text
gives substantially different detections:

```python
prompt = encode_concept(
    text,
    tokenizer,
    boxes=jnp.array([[120.0, 80.0, 320.0, 260.0]], jnp.float32),
    source_size=output_size,
)
output = eqx.filter_jit(vision.detect)(features, prompt)
```

`ConceptPrompt.create` takes the same exemplars and the text features directly, for code that encodes text elsewhere.

## Clicking on an object

The interactive head works from the features `encode_image` returned. Points and boxes are in input pixels, that is on
the 1008 x 1008 image, or in original pixels with `source_size`. Point labels are 1 for positive, 0 for negative and -1
for padding, default to positive, and any other value raises.

```python
from equimo.vision.models.sam3 import InteractivePrompt
from equimo.vision.ops import resize_bilinear

features = eqx.filter_jit(vision.encode_image)(image)
prompt = InteractivePrompt.create(
    points=jnp.array([[320.0, 180.0]], jnp.float32),
    source_size=output_size,
)
output = eqx.filter_jit(vision.predict_interactive)(features, prompt)

logits = output.mask_logits[output.selected]  # 288 x 288
mask = resize_bilinear(logits, output_size) > 0.0
```

`InteractivePrompt.create` also takes `box` (`x1 y1 x2 y2`) and `mask_input`, the `288 x 288` logits of a previous
prediction. `mask_logits` holds the single-mask output first and the three multimask outputs after it. With at most one
label (counting a box's two corners) and no mask input, `selected` is the best multimask output by predicted IoU;
otherwise it is the single-mask output unless its stability score is below 0.98.

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
experimental. Its decisions, object IDs and masks are checked against the reference on a hosted real clip.

On a GPU, use cuDNN attention: it runs a frame about 2.5 times faster than XLA attention, and XLA runs out of memory on
a 16 GB GPU at four buckets in bfloat16 (see [the performance notes](performance.md)).

```python
from equimo.core.layers.attention import with_attention_implementation
from equimo.vision.models.sam3.video import Sam3VideoTracker

vision = with_attention_implementation(vision, "cudnn")

# `frames` are [3, 1008, 1008] arrays, preprocessed as above.
num_frames = len(frames)
tracker = Sam3VideoTracker(
    vision, num_frames=num_frames, source_size=output_size, max_objects=16
)
tracker.set_prompt(encode_concept(text, tokenizer, "yellow school bus"))

predictions = []
for frame in frames:
    predictions += tracker.push_frame(frame)
predictions += tracker.finish()

for prediction in predictions:
    prediction.frame_index, prediction.object_ids, prediction.masks, prediction.boxes
```

`push_frame` returns the frames that became final. Decisions need two frames of look-ahead, so from the third frame on
each call returns the frame two steps earlier, and `finish` returns the rest. `set_prompt` must be called before the
first frame and raises afterwards. `push_frame` raises without a prompt, after `finish`, and past `num_frames`. The text
applies to every frame and box or point exemplars to the first frame only, as in the reference. `tracker.decisions`
holds one decision record per planned pass.

Objects occupy 16-slot buckets, and `max_objects` (default 16) caps how many detected objects are tracked. A batch of
new objects that fits no existing group of buckets opens new ones, and a removed object's slot stays unused until its
whole bucket is released, so the bucket count can grow even with 16 objects or fewer, up to the next power of two of
`max_objects`. The device code compiles once per bucket count, and a `push_frame` that grows the count recompiles, which
takes about a minute on an RTX 4080.

To stop tracking an object, call `remove_object`. It disappears from every frame not yet returned, and frames already
returned keep it:

```python
tracker.remove_object(object_id=3)
```

A later detection of the same object starts a new object with a new ID, and an ID that is not being tracked raises.

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
change, `max_objects` does not count it, and `remove_object` does not apply to it. `object_id` can name an unused ID,
and naming one in use raises. Refining an existing object with points, reverse propagation and a new prompt after the
first frame are not supported.

### Tracking a film clip

This is the code behind the animation at the top of this page, from a video file to a GIF, on a GPU. Equimo does not
decode video, so `ffmpeg` splits five seconds of `clip.mp4` into frames and assembles the result. The Python part also
needs Pillow and a CUDA build of JAX:

```bash
mkdir frames
ffmpeg -ss 5 -t 5 -i clip.mp4 -an -vf fps=12 frames/%04d.png
```

```python
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equimo.core.layers.attention import with_attention_implementation
from equimo.language import ClipBpeTokenizer, sam31_text
from equimo.vision.models.sam3 import encode_concept, sam31_multiplex
from equimo.vision.models.sam3.video import Sam3VideoTracker
from PIL import Image


def to_bfloat16(tree):
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


vision = to_bfloat16(sam31_multiplex(pretrained=True))
vision = with_attention_implementation(vision, "cudnn")
text = to_bfloat16(sam31_text(pretrained=True))

images = [Image.open(p).convert("RGB") for p in sorted(Path("frames").glob("*.png"))]
tracker = Sam3VideoTracker(
    vision,
    num_frames=len(images),
    source_size=(images[0].height, images[0].width),
)
tracker.set_prompt(encode_concept(text, ClipBpeTokenizer(), "person"))

predictions = []
for image in images:
    pixels = jnp.asarray(image.resize((1008, 1008)), jnp.float32) / 255.0
    frame = jnp.moveaxis((pixels - 0.5) / 0.5, -1, 0).astype(jnp.bfloat16)
    predictions += tracker.push_frame(frame)
predictions += tracker.finish()

Path("overlays").mkdir(exist_ok=True)
colors = np.array([[255, 64, 96], [64, 200, 255], [255, 200, 64], [120, 255, 120]])
for prediction, image in zip(predictions, images, strict=True):
    canvas = np.array(image)
    for object_id, mask in zip(prediction.object_ids, prediction.masks, strict=True):
        canvas[mask] = 0.45 * canvas[mask] + 0.55 * colors[object_id % len(colors)]
    Image.fromarray(canvas).save(f"overlays/{prediction.frame_index:04d}.png")
```

```bash
ffmpeg -framerate 12 -i overlays/%04d.png \
  -vf "scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" \
  -loop 0 tracked.gif
```
