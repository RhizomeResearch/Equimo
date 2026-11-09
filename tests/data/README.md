# Numerical reference fixtures

The `.npz` files in this directory are checked-in outputs from pinned upstream models. Their commands, upstream
revisions, input seeds and shapes, locked library versions, expected schemas, comparison tolerances, and fixture hashes
are recorded in `reference_provenance.json`.

`model_output_references.json` is a separate, small Equimo-only regression baseline for the tiny production model-family
matrix. It detects unintended local numerical changes; it is not evidence of PyTorch/upstream parity and is therefore
never listed in `reference_provenance.json`.

The EUPE fixture retains the original generator's deterministic RNG offset: a 256×256 image was sampled before its
224×224 input. The manifest records this as `rng_discard_shape`, and `models/torch_models.py` reproduces that sequence.

## Environment

Create the locked maintainer environment from the repository root:

```bash
uv sync --locked --group dev --group reference --extra audio
```

The reference group covers the tracked generators' published dependencies. The `audio` extra supplies the exact
TorchAudio filter-bank backend used by the pinned AST preprocessing references. Generating the T0-alpha reference
additionally requires accepting the model's Hugging Face access terms and authenticating with `hf auth login` or
`HF_TOKEN`. TIPS and EUPE still require their separately cloned upstream source trees; local paths must never be written
into the provenance file.

## Generate and review

Never generate directly into `tests/data`. Use the exact command recorded for the fixture, replacing documented
`/path/to/...` placeholders, and target a temporary directory such as `/tmp/equimo-references`.

First validate the committed set and its provenance:

```bash
uv run python models/validate_references.py
```

Then compare a complete candidate set:

```bash
uv run python models/validate_references.py \
  --candidate-dir /tmp/equimo-references
```

The comparison checks fixture coverage, SHA-256, keys, shapes, dtypes, maximum absolute delta, and mean absolute delta.
A delta beyond the recorded tolerance stops the review. Inspect every reported delta and the upstream release notes
before replacing a fixture. Replacement is a deliberate human action; the validator never writes into this directory.

When a fixture is accepted, update its generation command, revision, checkpoint checksum (when published), environment
versions, schema, tolerance, and new fixture SHA-256 in `reference_provenance.json` in the same change.

## SAM 3.1

`models/sam31_reference.py` generates the SAM 3.1 fixtures on a CUDA GPU, because the reference hard-codes CUDA in
several constructors and forward passes. It runs in its own environment, separate from `uv.lock` because the reference
requires NumPy below 2. Create it and clone the pinned reference outside any synced path:

```bash
uv venv --python 3.12 /path/to/environment
uv pip install --python /path/to/environment/bin/python -r models/sam31_reference_requirements.txt
git clone https://github.com/facebookresearch/sam3 /path/to/sam3
git -C /path/to/sam3 checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66
```

Generate into a staging directory, not `tests/data`:

```bash
/path/to/environment/bin/python models/sam31_reference.py tokenizer --reference-dir /path/to/sam3 --output-dir /path/to/output
/path/to/environment/bin/python models/sam31_reference.py operators --output-dir /path/to/output
/path/to/environment/bin/python models/sam31_reference.py tiny --reference-dir /path/to/sam3 --output-dir /path/to/output
```

Each fixture must regenerate byte-identically. The `operators` and `tiny` commands also write
`provenance_fragment.json`, the entries to copy into `reference_provenance.json`. The tokenizer command needs no GPU.

`sam31_image_reference.npz` holds full-weight component outputs on one image, as seeded index subsets, and the
reference's detections for one box exemplar with no text, generated with:

```bash
/path/to/environment/bin/python models/sam31_reference.py image --checkpoint /path/to/sam3.1_multiplex.pt --image /path/to/coins.png --concept coin --output-dir /path/to/output
```

The image is `skimage/data/coins.png` (Greek coins from Pompeii, Brooklyn Museum Collection, no known copyright
restrictions), and the fixture stores it resized to 1008 x 1008. To run the checks on a GPU, set
`JAX_DEFAULT_MATMUL_PRECISION=highest`; the default XLA GPU matmul precision alone exceeds the 5e-4 mean absolute error.
The checks read the hosted archives through `pretrained=True`, from the cache filled by
`models/prefetch_reference_checkpoints.py`, and skip without them.

The tiny clip's decision trace is in `sam3_tiny_reference.json`. Its detection and track masks are a fixed script,
because random weights produce nothing the association policy would act on; the real detector and tracker still run and
are recorded under `inputs.*` and `outputs.*`. `frames.*` holds, per pass, the scripted detections and tracks, what the
tracker computed before the script replaced its masks (`raw_track_*`), and what each state stored once the frame was
executed (`maskmem_features`, `obj_ptr`, `pred_masks`, `object_score_logits`). `points.*` holds the singleton state of
the object added with points.

`sam3_tiny_occlusion_reference.npz` and its JSON companion cover occlusion suppression, which no clip tried for the
full-weight fixture triggers. Two tracks go empty on different frames and then land on one mask, so the reference
suppresses the more recently occluded object. The `tiny` command writes it along with the tiny clip.

The hosted full-weight video fixture is `sam31_video_reference.tar.lz4` in `poiretclement/equimo`, a `tar` holding one
`npz`: the 48 preprocessed frames, the concept's token ids, the reference's decision trace in the `decisions` array, and
the masks and IDs it emits per frame. `tests/cases/sam31_video_fixture.py` pins its digest and reads it. The clip is
"[Pigeon feeding, Jardim da Estrela, 2025 01](https://commons.wikimedia.org/wiki/File:Pigeon_feeding,_Jardim_da_Estrela,_2025_01.webm)"
(CC0), non-square, with several objects of one concept and one track removed as unmatched. Decode 48 frames, every
second one, from 13 s, and write the float32 fixture with the concept `pigeon`:

```bash
ffmpeg -ss 13 -i Pigeon_feeding.webm -vf "select='not(mod(n\,2))',setpts=N/TB,scale=640:-2" -fps_mode passthrough \
  -frames:v 48 -start_number 0 -q:v 2 frames/%05d.jpg
/path/to/environment/bin/python models/sam31_reference.py video --checkpoint /path/to/sam3.1_multiplex.pt --frames-dir frames --concept pigeon --output-dir /path/to/output
uv run python tests/cases/sam31_video_fixture.py /path/to/output/sam31_video_reference.npz /path/to/output/sam31_video_reference.tar.lz4
```

`sam31_video_exemplar_reference.npz` holds seeded subsets of the reference's detector masks and boxes on the first four
frames of that clip, for one box exemplar on frame 0. The box is on a pigeon the text-prompted run detected on frame 0:

```bash
/path/to/environment/bin/python models/sam31_reference.py video-exemplar --checkpoint /path/to/sam3.1_multiplex.pt --frames-dir frames --box 0.1906 0.2472 0.1219 0.2611 --output-dir /path/to/output
```

The harness fails when a kept detection, overlap or box center lies within 1e-3 of its threshold, and when the clip does
not trigger an addition, an unmatched removal, a reconditioning and a confirmation; suppression is covered by the tiny
occlusion fixture. To run the checks on a GPU, set `JAX_DEFAULT_MATMUL_PRECISION=highest` as for the image fixture.
