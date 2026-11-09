"""SAM 3.1 image and tracker modules on a tiny random-weight configuration."""

import importlib.util
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.core.layers.attention import with_attention_implementation
from equimo.conversion.utils import stringify_name
from equimo.core.parameters import trainable_mask
from equimo.finetune import (
    AdapterConfig,
    LoRAConfig,
    OutputAdapterModule,
    TargetSpec,
    apply_adapters,
    apply_lora,
)
from equimo.vision.layers.posemb import VisionRoPE
from equimo.vision.models.sam3 import (
    CommitPlan,
    ConceptPrompt,
    DetectorOutput,
    ImageFeatures,
    InteractivePrompt,
    Sam3,
    TrackerState,
    TrackOutput,
    decode_concepts,
    pad_tracker_buckets,
)
from equimo.vision.models.sam3 import tracker as tracker_module
from equimo.vision.models.sam3.common import (
    exchanged_rope_factors,
    flatten_map,
    sine_position_encoding,
)
from equimo.vision.models.sam3.decoding import greedy_iom_nms, single_frame_nms
from equimo.vision.models.sam3.detector import joint_logit
from equimo.vision.models.sam3.encoder import ViTDetBlock
from equimo.vision.models.sam3.interactive import _select_mask
from equimo.vision.models.sam3.tracker import (
    SLOT_ACTIVE,
    SLOT_EMPTY,
    SLOT_REMOVED,
    _keep_per_group,
    _memory_key_rotary,
    _selected_memory_frames,
    _shrunk_slots,
    associate,
)

GPU = pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="cuDNN attention requires a CUDA GPU"
)

KEY = jr.PRNGKey(0)
DIM = 16
GRID = 4
IMAGE = jr.normal(jr.PRNGKey(1), (3, 56, 56))
MASK_SIZE = 4 * GRID
TEXT = jr.normal(jr.PRNGKey(2), (5, DIM))
NUM_KEYS = 2 * GRID * GRID + 3
MEMORY = jr.normal(jr.PRNGKey(3), (NUM_KEYS, DIM))
SLOT_MASKS = jr.normal(jr.PRNGKey(4), (4, MASK_SIZE, MASK_SIZE))


def _concept_prompt(*, dim: int = DIM, **kwargs) -> ConceptPrompt:
    return ConceptPrompt.create(
        **{"text_features": TEXT, **kwargs},
        dim=dim,
        max_text_tokens=8,
        max_boxes=2,
        max_points=2,
    )


def _interactive_prompt(**kwargs) -> InteractivePrompt:
    if "points" not in kwargs:
        kwargs.update(points=jnp.asarray([[10.0, 20.0]]), point_labels=jnp.asarray([1]))
    return InteractivePrompt.create(**kwargs, mask_size=MASK_SIZE, max_points=2)


def _tiny_model(dim: int) -> Sam3:
    return Sam3(
        key=KEY,
        in_channels=3,
        img_size=56,
        patch_size=14,
        embed_dim=dim,
        depth=2,
        num_heads=2,
        mlp_features=2 * dim,
        window_size=2,
        global_blocks=(1,),
        dim=dim,
        head_num_heads=2,
        ffn_features=2 * dim,
        geometry_depth=1,
        fusion_depth=1,
        decoder_depth=2,
        num_queries=4,
        two_way_depth=1,
        memory_depth=1,
        memory_heads=2,
        memory_start_channels=1,
        memory_fuser_depth=1,
        num_slots=2,
    )


@pytest.fixture(scope="module")
def model() -> Sam3:
    return _tiny_model(DIM)


def _to_bfloat16(tree):
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


def _all_outputs(model: Sam3, image, text, slot_masks, memory) -> dict[str, jax.Array]:
    features = model.encode_image(image)
    dim = features.propagation[2].shape[0]
    detected = model.detect(features, _concept_prompt(text_features=text, dim=dim))
    interactive = model.predict_interactive(features, _interactive_prompt())
    tracker = model.tracker
    low = features.propagation[2]
    multiplex = tracker.mask_decoder(
        low,
        tracker.image_position(low.dtype),
        tracker.mask_decoder.upsampler.project(features.propagation[:2]),
        jnp.asarray([True, False]),
    )
    tokens = jnp.moveaxis(low, 0, -1).reshape(GRID * GRID, dim)
    attended = tracker.memory_attention(
        tokens,
        jnp.zeros_like(tokens),
        jnp.zeros_like(memory),
        memory,
        memory,
        num_spatial_keys=jnp.asarray(GRID * GRID),
        key_lengths=jnp.asarray(GRID * GRID + 2),
    )
    return {
        "features": model.features(image),
        "pyramid_high": features.detector[0],
        "logits": detected.logits,
        "presence": detected.presence_logit,
        "boxes": detected.boxes,
        "masks": detected.mask_logits,
        "layer_boxes": detected.layer_boxes,
        "layer_logits": detected.layer_logits,
        "interactive_masks": interactive.mask_logits,
        "interactive_iou": interactive.iou,
        "object_score": interactive.object_score_logit,
        "multiplex_masks": multiplex.mask_logits,
        "multiplex_iou": multiplex.iou,
        "multiplex_scores": multiplex.object_score_logits,
        "multiplex_tokens": multiplex.mask_tokens,
        "memory": tracker.memory_encoder(low, slot_masks),
        "attended": attended,
    }


# Compiled and eager float32 runs of the whole graph differ by up to 3e-3 on an
# RTX 4080, although each endpoint compiled alone agrees to rounding. On CPU the
# remaining difference comes from the tiny mask decoder's two-channel LayerNorm2d,
# which amplifies rounding wherever both channels are nearly equal.
_GPU = jax.default_backend() == "gpu"
_JIT_TOLERANCE = 1e-2 if _GPU else 1e-5
_VMAP_TOLERANCE = 1e-2 if _GPU else 1e-3

EXPECTED_SHAPES = {
    "features": (DIM, GRID, GRID),
    "pyramid_high": (DIM, MASK_SIZE, MASK_SIZE),
    "logits": (4,),
    "presence": (),
    "boxes": (4, 4),
    "masks": (4, MASK_SIZE, MASK_SIZE),
    "layer_boxes": (2, 4, 4),
    "layer_logits": (2, 4),
    "interactive_masks": (4, MASK_SIZE, MASK_SIZE),
    "interactive_iou": (4,),
    "object_score": (),
    "multiplex_masks": (2, 3, MASK_SIZE, MASK_SIZE),
    "multiplex_iou": (2, 3),
    "multiplex_scores": (2,),
    "multiplex_tokens": (2, 3, DIM),
    "memory": (DIM, GRID, GRID),
    "attended": (GRID * GRID, DIM),
}


def test_endpoints_return_documented_shapes(model):
    outputs = _all_outputs(model, IMAGE, TEXT, SLOT_MASKS, MEMORY)
    assert {name: value.shape for name, value in outputs.items()} == EXPECTED_SHAPES
    for name, value in outputs.items():
        assert jnp.all(jnp.isfinite(value)), name


def test_features_are_the_detector_pyramid_low_level(model):
    assert jnp.array_equal(model.features(IMAGE), model.encode_image(IMAGE).detector[2])


def test_outputs_match_between_eager_jit_and_vmap(model):
    eager = _all_outputs(model, IMAGE, TEXT, SLOT_MASKS, MEMORY)
    compiled = eqx.filter_jit(_all_outputs)(model, IMAGE, TEXT, SLOT_MASKS, MEMORY)
    batched = eqx.filter_vmap(_all_outputs, in_axes=(None, 0, 0, 0, 0))(
        model,
        jnp.stack((IMAGE, IMAGE)),
        jnp.stack((TEXT, TEXT)),
        jnp.stack((SLOT_MASKS, SLOT_MASKS)),
        jnp.stack((MEMORY, MEMORY)),
    )
    for name, value in eager.items():
        assert jnp.allclose(
            compiled[name], value, rtol=_JIT_TOLERANCE, atol=_JIT_TOLERANCE
        ), name
        assert batched[name].shape == (2, *value.shape), name
        assert jnp.allclose(
            batched[name][0], value, rtol=_VMAP_TOLERANCE, atol=_VMAP_TOLERANCE
        ), name


def test_gradients_reach_every_module_and_stay_finite(model):
    def loss(model):
        outputs = _all_outputs(model, IMAGE, TEXT, SLOT_MASKS, MEMORY)
        return sum(jnp.sum(value.astype(jnp.float32)) for value in outputs.values())

    gradients = eqx.filter_grad(loss)(model)
    leaves = [
        leaf for leaf in jax.tree.leaves(eqx.filter(gradients, eqx.is_inexact_array))
    ]
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    for module in ("trunk", "neck", "detector", "interactive", "tracker"):
        grads = jax.tree.leaves(
            eqx.filter(getattr(gradients, module), eqx.is_inexact_array)
        )
        assert any(jnp.any(leaf != 0) for leaf in grads), module


def test_gradients_stay_finite_without_any_prompt_tokens(model):
    def loss(model):
        features = model.encode_image(IMAGE)
        prompt = ConceptPrompt.create(
            dim=DIM, max_text_tokens=8, max_boxes=2, max_points=2
        )
        return jnp.sum(model.detect(features, prompt).mask_logits)

    gradients = eqx.filter_grad(loss)(model)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree.leaves(eqx.filter(gradients, eqx.is_inexact_array))
    )


def test_bfloat16_outputs_are_finite_and_keep_their_dtypes(model):
    outputs = _all_outputs(
        _to_bfloat16(model),
        _to_bfloat16(IMAGE),
        _to_bfloat16(TEXT),
        _to_bfloat16(SLOT_MASKS),
        _to_bfloat16(MEMORY),
    )
    for name, value in outputs.items():
        expected = (
            jnp.float32
            if name in ("masks", "interactive_masks", "multiplex_masks")
            else jnp.bfloat16
        )
        assert value.dtype == expected, name
        assert jnp.all(jnp.isfinite(value.astype(jnp.float32))), name


def test_padded_exemplars_do_not_change_detections(model):
    features = model.encode_image(IMAGE)
    boxes = jnp.asarray([[0.5, 0.5, 0.4, 0.4]])
    reference = model.detect(features, _concept_prompt(boxes=boxes))

    prompt = _concept_prompt(boxes=boxes)
    edited = eqx.tree_at(
        lambda p: (p.boxes, p.box_labels),
        prompt,
        (
            prompt.boxes.at[1].set(jnp.asarray([0.2, 0.3, 0.1, 0.1])),
            prompt.box_labels.at[1].set(1),
        ),
    )
    assert jnp.array_equal(model.detect(features, edited).logits, reference.logits)
    assert jnp.array_equal(
        model.detect(features, edited).mask_logits, reference.mask_logits
    )


def test_padded_points_and_absent_box_do_not_change_interactive_output(model):
    features = model.encode_image(IMAGE)
    prompt = _interactive_prompt()
    reference = model.predict_interactive(features, prompt)
    edited = eqx.tree_at(
        lambda p: (p.points, p.box),
        prompt,
        (
            prompt.points.at[1].set(jnp.asarray([30.0, 40.0])),
            jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        ),
    )
    result = model.predict_interactive(features, edited)
    assert jnp.array_equal(result.mask_logits, reference.mask_logits)
    assert jnp.array_equal(result.iou, reference.iou)


def test_interactive_prompt_kinds_change_the_output(model):
    features = model.encode_image(IMAGE)
    points = model.predict_interactive(features, _interactive_prompt())
    box = model.predict_interactive(
        features, _interactive_prompt(box=jnp.asarray([4.0, 4.0, 30.0, 30.0]))
    )
    mask = model.predict_interactive(
        features,
        _interactive_prompt(mask_input=jr.normal(KEY, (MASK_SIZE, MASK_SIZE))),
    )
    assert not jnp.array_equal(points.mask_logits, box.mask_logits)
    assert not jnp.array_equal(points.mask_logits, mask.mask_logits)


def test_mask_selection_follows_the_label_count_and_stability():
    single_stable = jnp.ones((4, 4)) * 5.0
    logits = jnp.stack([single_stable] * 4)
    iou = jnp.asarray([0.1, 0.2, 0.9, 0.3])
    no_mask = jnp.asarray(False)
    assert _select_mask(logits, iou, jnp.asarray(1), no_mask) == 2
    assert _select_mask(logits, iou, jnp.asarray(2), no_mask) == 0
    assert _select_mask(logits, iou, jnp.asarray(0), jnp.asarray(True)) == 0

    unstable = jnp.zeros((4, 4))
    unstable = unstable.at[0, 0].set(1.0)
    logits = jnp.stack([unstable] * 4)
    assert _select_mask(logits, iou, jnp.asarray(2), no_mask) == 2


def test_joint_logit_equals_the_product_of_probabilities():
    class_logit = jnp.asarray([-2.0, 0.0, 3.0])
    presence = jnp.asarray([1.5, -1.0, 2.0])
    probability = jax.nn.sigmoid(class_logit) * jax.nn.sigmoid(presence)
    expected = jnp.log(probability) - jnp.log1p(-probability)
    assert jnp.allclose(joint_logit(class_logit, presence), expected, rtol=1e-5)

    extreme = joint_logit(jnp.asarray([12.0, -12.0]), jnp.asarray([200.0, -200.0]))
    assert jnp.all(jnp.isfinite(extreme))


def _row_masks(*spans: tuple[int, int]) -> jax.Array:
    masks = np.zeros((len(spans), 1, 8), dtype=bool)
    for index, (start, stop) in enumerate(spans):
        masks[index, 0, start:stop] = True
    return jnp.asarray(masks)


def test_single_frame_nms_lets_suppressed_masks_suppress_later_ones():
    masks = _row_masks((0, 4), (3, 7), (6, 8))
    keep = single_frame_nms(masks)
    assert keep.tolist() == [True, False, False]


def test_greedy_nms_only_lets_kept_masks_suppress():
    masks = _row_masks((0, 4), (3, 7), (6, 8))
    scores = jnp.asarray([0.9, 0.8, 0.7])
    everything = jnp.ones(3, bool)
    assert greedy_iom_nms(masks, scores, everything, 0.1).tolist() == [
        True,
        False,
        True,
    ]
    skip_first = jnp.asarray([False, True, True])
    assert greedy_iom_nms(masks, scores, skip_first, 0.1).tolist() == [
        False,
        True,
        False,
    ]


def test_greedy_nms_breaks_score_ties_by_index():
    masks = _row_masks((0, 4), (0, 4))
    keep = greedy_iom_nms(masks, jnp.asarray([0.5, 0.5]), jnp.ones(2, bool), 0.1)
    assert keep.tolist() == [True, False]


def _output_with_masks(
    mask_logits: jax.Array, logits: jax.Array, presence: float = 3.0
) -> DetectorOutput:
    count = mask_logits.shape[0]
    return DetectorOutput(
        logits=logits,
        presence_logit=jnp.asarray(presence),
        boxes=jnp.zeros((count, 4)),
        mask_logits=mask_logits,
        layer_boxes=jnp.zeros((1, count, 4)),
        layer_logits=logits[None],
    )


def test_decode_concepts_ranks_thresholds_and_removes_overlaps():
    a = np.full((4, 8), -1.0)
    b = np.full((4, 8), -1.0)
    c = np.full((4, 8), -1.0)
    a[:, :4] = 1.0
    b[0, 4:6] = 1.0
    c[3, 6:8] = 1.0
    out = _output_with_masks(
        jnp.asarray(np.stack((c, a, b)), jnp.float32),
        jnp.asarray([-3.0, 2.0, 0.5], jnp.float32),
    )
    decoded = decode_concepts(out, output_size=(4, 8), max_detections=3)

    # ``c`` falls below the threshold, and ``b`` does not overlap ``a``.
    assert decoded.keep.tolist() == [True, True, False]
    assert decoded.query_index[:2].tolist() == [1, 2]
    assert jnp.allclose(
        decoded.scores[:2], jax.nn.sigmoid(jnp.asarray([2.0, 0.5])), rtol=1e-6
    )
    assert jnp.allclose(decoded.presence, jax.nn.sigmoid(jnp.asarray(3.0)))
    assert decoded.boxes[0].tolist() == [0.0, 0.0, 3.0, 3.0]
    assert decoded.boxes[1].tolist() == [4.0, 0.0, 5.0, 0.0]
    assert jnp.sum(decoded.masks[0] & decoded.masks[1]) == 0
    assert decoded.masks[1].sum() == 2
    assert not jnp.any(decoded.masks[2])


def test_decode_concepts_keeps_only_the_highest_scores_at_the_cap():
    masks = np.full((3, 2, 6), -1.0)
    for index in range(3):
        masks[index, :, index * 2 : index * 2 + 2] = 1.0
    out = _output_with_masks(
        jnp.asarray(masks, jnp.float32), jnp.asarray([1.0, 3.0, 2.0], jnp.float32)
    )
    decoded = decode_concepts(out, output_size=(2, 6), max_detections=2)
    assert decoded.keep.tolist() == [True, True]
    assert decoded.query_index.tolist() == [1, 2]


def test_decode_concepts_runs_under_jit_and_vmap():
    logits = jr.normal(KEY, (2, 4))
    masks = jr.normal(jr.PRNGKey(5), (2, 4, 8, 8))
    decode = eqx.filter_vmap(
        lambda m, l: decode_concepts(
            _output_with_masks(m, l),
            output_size=(16, 12),
            max_detections=2,
            threshold=0.0,
        )
    )
    decoded = eqx.filter_jit(decode)(masks, logits)
    assert decoded.masks.shape == (2, 2, 16, 12)
    assert decoded.boxes.shape == (2, 2, 4)


def test_windowed_block_attends_only_within_its_window():
    block = ViTDetBlock(16, 2, 32, window_size=2, rope_grid=2, key=KEY)
    x = jr.normal(jr.PRNGKey(6), (16, GRID, GRID))
    base = block(x)
    # A change to one channel only: layer norm cancels a shift of all channels.
    edited = block(x.at[0, 3, 3].add(1.0))

    # Tokens outside the bottom-right window keep their exact values.
    assert jnp.array_equal(edited[:, 0, :], base[:, 0, :])
    assert jnp.array_equal(edited[:, :2, :], base[:, :2, :])
    assert not jnp.array_equal(edited[:, 2, 2], base[:, 2, 2])


def test_global_block_attends_across_the_whole_grid():
    block = ViTDetBlock(16, 2, 32, window_size=0, rope_grid=2, key=KEY)
    x = jr.normal(jr.PRNGKey(6), (16, GRID, GRID))
    base = block(x)
    edited = block(x.at[0, 3, 3].add(1.0))
    assert not jnp.array_equal(edited[:, 0, 0], base[:, 0, 0])


def test_rope_factors_rotate_width_before_height():
    rope = VisionRoPE(
        "mode", dim=16, num_heads=2, freqs_for="lang", theta=10000, pt_seq_len=2
    )
    factors = exchanged_rope_factors(rope, 3)
    original = rope.get_factors(H=3, W=3)
    assert jnp.array_equal(factors.sin[0 * 3 + 1], original.sin[1 * 3 + 0])
    assert jnp.array_equal(factors.cos[2 * 3 + 1], original.cos[1 * 3 + 2])
    assert factors.layout == original.layout


def test_memory_key_rotary_repeats_per_frame_and_skips_pointers():
    rope = VisionRoPE(
        "mode", dim=16, num_heads=2, freqs_for="lang", theta=10000, pt_seq_len=2
    )
    table = exchanged_rope_factors(rope, 2)
    rotary = _memory_key_rotary(table, 10, jnp.asarray(8), 2)
    assert jnp.array_equal(rotary.sin[:4], table.sin)
    assert jnp.array_equal(rotary.sin[4:8], table.sin)
    assert jnp.array_equal(rotary.sin[8:], jnp.zeros((2, 8)))
    assert jnp.array_equal(rotary.cos[8:], jnp.ones((2, 8)))


def test_memory_attention_ignores_keys_past_the_length(model):
    attention = model.tracker.memory_attention
    x = jr.normal(jr.PRNGKey(7), (GRID * GRID, DIM))
    lengths = jnp.asarray(GRID * GRID + 1)
    position = jnp.zeros_like(x)
    image = jnp.zeros_like(MEMORY)
    base = attention(
        x,
        position,
        image,
        MEMORY,
        MEMORY,
        num_spatial_keys=jnp.asarray(GRID * GRID),
        key_lengths=lengths,
    )
    changed_memory = MEMORY.at[GRID * GRID + 1 :].set(100.0)
    other = attention(
        x,
        position,
        image,
        changed_memory,
        changed_memory,
        num_spatial_keys=jnp.asarray(GRID * GRID),
        key_lengths=lengths,
    )
    assert jnp.array_equal(base, other)


def test_memory_attention_chunking_matches_one_block(model, monkeypatch):
    attention = model.tracker.memory_attention
    x = jr.normal(jr.PRNGKey(7), (GRID * GRID, DIM))
    kwargs = {
        "num_spatial_keys": jnp.asarray(GRID * GRID),
        "key_lengths": jnp.asarray(GRID * GRID + 2),
    }
    arguments = (x, jnp.zeros_like(x), jnp.zeros_like(MEMORY), MEMORY, MEMORY)
    whole = attention(*arguments, **kwargs)
    monkeypatch.setattr(tracker_module, "_ATTENTION_CHUNK_QUERIES", 3)
    chunked = attention(*arguments, **kwargs)
    assert jnp.allclose(whole, chunked, rtol=1e-6, atol=1e-6)


def test_slot_without_an_object_gets_the_no_object_logit(model):
    tracker = model.tracker
    low = jr.normal(jr.PRNGKey(8), (DIM, GRID, GRID))
    high = jr.normal(jr.PRNGKey(9), (DIM, MASK_SIZE, MASK_SIZE))
    middle = jr.normal(jr.PRNGKey(10), (DIM, 2 * GRID, 2 * GRID))
    output = tracker.mask_decoder(
        low,
        tracker.image_position(low.dtype),
        tracker.mask_decoder.upsampler.project((high, middle)),
        jnp.asarray([True, False]),
    )
    absent = output.object_score_logits <= 0
    assert jnp.all(output.mask_logits[absent] == -1024.0)
    assert jnp.all(output.mask_logits[~absent] != -1024.0)


def test_slot_validity_changes_the_decoded_slot(model):
    tracker = model.tracker
    low = jr.normal(jr.PRNGKey(8), (DIM, GRID, GRID))
    high = jr.normal(jr.PRNGKey(9), (DIM, MASK_SIZE, MASK_SIZE))
    middle = jr.normal(jr.PRNGKey(10), (DIM, 2 * GRID, 2 * GRID))

    def decode(valid):
        return tracker.mask_decoder(
            low,
            tracker.image_position(low.dtype),
            tracker.mask_decoder.upsampler.project((high, middle)),
            jnp.asarray(valid),
        )

    a = decode([True, False])
    b = decode([False, True])
    assert not jnp.array_equal(a.mask_tokens[0], b.mask_tokens[0])


def test_fourier_matrices_are_never_trained(model):
    mask = trainable_mask(model)
    assert mask.interactive.prompt_encoder.fourier_matrix is False
    assert mask.tracker.image_position.fourier_matrix is False
    assert mask.interactive.prompt_encoder.point_embeddings is True


def test_adapters_after_attention_run_forward(model):
    adapted = apply_adapters(
        model,
        AdapterConfig(
            bottleneck=2, placement="both", target=TargetSpec(include=("*.blocks.*",))
        ),
        key=jr.PRNGKey(7),
    )
    assert isinstance(adapted.trunk.blocks[0].attn, OutputAdapterModule)
    assert isinstance(adapted.detector.head.decoder.blocks[0].attn, OutputAdapterModule)
    outputs = _all_outputs(adapted, IMAGE, TEXT, SLOT_MASKS, MEMORY)
    assert {name: value.shape for name, value in outputs.items()} == EXPECTED_SHAPES
    for name, value in outputs.items():
        assert jnp.isfinite(value).all(), name


def test_lora_on_attention_projections_runs_forward(model):
    adapted = apply_lora(
        model,
        LoRAConfig(
            rank=2,
            target=TargetSpec(
                tags_any=("attention.proj",),
                exclude=("interactive", "tracker"),
            ),
        ),
        key=KEY,
    )
    decoder_block = adapted.detector.head.decoder.blocks[0]
    assert not isinstance(decoder_block.attn.proj, eqx.nn.Linear)
    assert all(
        not isinstance(attention.proj, eqx.nn.Linear)
        for attention in decoder_block.cross_attn
    )
    assert isinstance(
        adapted.tracker.mask_decoder.transformer.blocks[0].attn.proj, eqx.nn.Linear
    )
    features = adapted.encode_image(IMAGE)
    out = adapted.detect(features, _concept_prompt())
    assert jnp.all(jnp.isfinite(out.mask_logits))


def test_attention_implementation_switches_every_attention(model):
    switched = with_attention_implementation(model, "cudnn")
    block = switched.trunk.blocks[0]
    assert block.attn.implementation == "cudnn"
    assert switched.tracker.memory_attention.blocks[0].cross_attn.implementation == (
        "cudnn"
    )
    assert switched.detector.head.decoder.blocks[0].cross_attn[1].implementation == (
        "cudnn"
    )


@GPU
def test_cudnn_attention_matches_xla_in_bfloat16():
    # cuDNN requires head widths that are multiples of 8.
    model = _to_bfloat16(_tiny_model(32))
    fused = with_attention_implementation(model, "cudnn")
    args = (
        _to_bfloat16(IMAGE),
        _to_bfloat16(jr.normal(jr.PRNGKey(2), (5, 32))),
        _to_bfloat16(SLOT_MASKS),
        _to_bfloat16(jr.normal(jr.PRNGKey(3), (NUM_KEYS, 32))),
    )
    reference = _all_outputs(model, *args)
    outputs = _all_outputs(fused, *args)
    for name, value in reference.items():
        assert jnp.allclose(
            outputs[name].astype(jnp.float32),
            value.astype(jnp.float32),
            rtol=0.05,
            atol=0.05,
        ), name


SOURCE_SIZE = (40, 60)


@pytest.fixture(scope="module")
def frame_features(model) -> ImageFeatures:
    return model.encode_image(IMAGE)


def _new_mask() -> jax.Array:
    return jr.normal(jr.PRNGKey(11), (MASK_SIZE, MASK_SIZE))


def _plan(state: TrackerState, frame, **fields) -> CommitPlan:
    plan = CommitPlan.empty(state, jnp.asarray(frame, jnp.int32), SOURCE_SIZE)
    if not fields:
        return plan
    return eqx.tree_at(
        lambda p: tuple(getattr(p, name) for name in fields),
        plan,
        tuple(fields.values()),
    )


def _add_slots(state: TrackerState, frame, *slots: tuple[int, int]) -> CommitPlan:
    add = jnp.zeros_like(state.slot_status, dtype=bool)
    new_masks = jnp.zeros_like(state.last_masks)
    for bucket, slot in slots:
        add = add.at[bucket, slot].set(True)
        new_masks = new_masks.at[bucket, slot].set(_new_mask())
    return _plan(state, frame, add=add, new_masks=new_masks)


def _step(model, state, features, frame, plan):
    tracks = model.propagate(state, features, frame)
    return model.commit(state, features, tracks, plan), tracks


@pytest.fixture(scope="module")
def step():
    return eqx.filter_jit(_step)


def _run(step, model, features, state, plans):
    for plan in plans:
        state, tracks = step(model, state, features, plan.frame_index, plan)
    return state, tracks


def test_tracking_compiles_once_per_bucket_count(model, frame_features):
    traced_buckets = []

    def counted(model, state, features, frame, plan, detections):
        traced_buckets.append(state.slot_status.shape[0])
        state, tracks = _step(model, state, features, frame, plan)
        return state, model.associate(detections, tracks, state)

    counted = eqx.filter_jit(counted)
    detections = _output_with_masks(
        jr.normal(jr.PRNGKey(12), (4, MASK_SIZE, MASK_SIZE)), jnp.zeros(4)
    )

    def advance(state, plan):
        return counted(
            model, state, frame_features, plan.frame_index, plan, detections
        )[0]

    state = model.tracker.empty_state(1)
    state = advance(state, _add_slots(state, 0, (0, 0)))
    state = advance(state, _add_slots(state, 1, (0, 1)))
    remove = jnp.zeros((1, 2), bool).at[0, 0].set(True)
    state = advance(state, _plan(state, 2, remove=remove))
    state = advance(state, _plan(state, 3))
    assert traced_buckets == [1]

    state = pad_tracker_buckets(state, 2)
    state = advance(state, _add_slots(state, 4, (1, 0)))
    state = advance(state, _plan(state, 5))
    assert traced_buckets == [1, 2]

    state = pad_tracker_buckets(state, 4)
    state = advance(state, _plan(state, 6))
    assert traced_buckets == [1, 2, 4]


def test_ring_buffers_hold_the_latest_frames(model, frame_features, step):
    state = model.tracker.empty_state(1)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0)),
    )
    assert state.recent_is_cond[0, 0]
    for frame in range(1, 18):
        state, _ = step(
            model,
            state,
            frame_features,
            jnp.asarray(frame, jnp.int32),
            _plan(state, frame),
        )
        recent = set(state.recent_frame[0].tolist()) - {-1}
        pointers = set(state.pointer_frame[0].tolist()) - {-1}
        assert recent == set(range(max(0, frame - 5), frame + 1))
        assert pointers == set(range(max(0, frame - 14), frame + 1))
        assert state.recent_frame[0, frame % 6] == frame
        assert state.pointer_frame[0, frame % 15] == frame
        assert not state.recent_is_cond[0, frame % 6]
    assert state.cond_frame[0].tolist() == [0, -1, -1, -1]


def test_conditioning_buffer_keeps_the_four_latest_frames(model, frame_features, step):
    state = model.tracker.empty_state(1)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0)),
    )
    recondition = jnp.zeros((1, 2), bool).at[0, 0].set(True)
    for frame in range(1, 7):
        plan = _plan(
            state,
            frame,
            recondition=recondition,
            recondition_masks=jnp.zeros((1, 2, MASK_SIZE, MASK_SIZE)),
        )
        state, _ = step(
            model, state, frame_features, jnp.asarray(frame, jnp.int32), plan
        )
    assert set(state.cond_frame[0].tolist()) == {3, 4, 5, 6}
    assert state.recent_is_cond[0, 6 % 6]


def test_selected_frames_skip_conditioning_entries_read_from_their_buffer(model):
    state = eqx.tree_at(
        lambda s: (
            s.cond_frame,
            s.recent_frame,
            s.recent_is_cond,
            s.pointer_frame,
        ),
        model.tracker.empty_state(1),
        (
            jnp.asarray([[5, 6, 7, 8]], jnp.int32),
            jnp.asarray([[6, 7, 8, 3, 4, -1]], jnp.int32),
            jnp.asarray([[True, True, True, True, False, False]]),
            jnp.asarray([[8, 7, 6, 3] + [-1] * 11], jnp.int32),
        ),
    )
    cond, recent, pointer = _selected_memory_frames(
        jax.tree.map(lambda leaf: leaf[0], state)
    )
    assert cond.tolist() == [True] * 4
    # Frame 3 was a conditioning frame that left the buffer, so it counts as recent.
    assert recent.tolist() == [False, False, False, True, True, False]
    assert pointer.tolist() == [False, False, False, True] + [False] * 11


def test_memory_keys_count_only_selected_frames(model, frame_features, step):
    state = model.tracker.empty_state(1)
    bucket = jax.tree.map(lambda leaf: leaf[0], state)
    frame = jnp.asarray(0, jnp.int32)
    _, _, _, spatial, lengths = model.tracker._memory_bank(bucket, frame, 15)
    assert spatial == 0 and lengths == 0

    state, _ = step(model, state, frame_features, frame, _add_slots(state, 0, (0, 0)))
    bucket = jax.tree.map(lambda leaf: leaf[0], state)
    image, memory, _, spatial, lengths = model.tracker._memory_bank(
        bucket, jnp.asarray(1, jnp.int32), 15
    )
    # Frame 0 is read once, as a conditioning frame, not again as a recent one.
    assert spatial == GRID * GRID
    assert lengths == GRID * GRID + model.tracker.mask_decoder.num_slots
    assert image.shape == memory.shape


def test_stored_features_are_the_spatial_keys(model, frame_features, step):
    state = model.tracker.empty_state(1)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0)),
    )
    frame = jnp.asarray(1, jnp.int32)
    base = model.propagate(state, frame_features, frame).mask_logits
    selected = eqx.tree_at(
        lambda s: s.cond_features, state, state.cond_features.at[0, 0].add(1.0)
    )
    unselected = eqx.tree_at(
        lambda s: s.cond_features, state, state.cond_features.at[0, 1].add(1.0)
    )
    assert not jnp.array_equal(
        model.propagate(selected, frame_features, frame).mask_logits, base
    )
    assert jnp.array_equal(
        model.propagate(unselected, frame_features, frame).mask_logits, base
    )


def test_pointer_encoding_depends_on_the_known_clip_length(model, frame_features, step):
    state = model.tracker.empty_state(1)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0)),
    )
    frame = jnp.asarray(1, jnp.int32)
    streaming = model.propagate(state, frame_features, frame)
    long_clip = model.propagate(state, frame_features, frame, num_frames=40)
    short_clip = model.propagate(state, frame_features, frame, num_frames=3)
    assert jnp.array_equal(streaming.mask_logits, long_clip.mask_logits)
    assert not jnp.array_equal(streaming.mask_logits, short_clip.mask_logits)


def test_slot_status_never_returns_from_removed(model, frame_features, step):
    state = model.tracker.empty_state(1)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0)),
    )
    assert state.slot_status[0].tolist() == [SLOT_ACTIVE, SLOT_EMPTY]

    remove = jnp.zeros((1, 2), bool).at[0, 0].set(True)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(1, jnp.int32),
        _plan(state, 1, remove=remove),
    )
    assert state.slot_status[0].tolist() == [SLOT_REMOVED, SLOT_EMPTY]
    assert jnp.all(state.last_masks[0, 0] == -1024.0)

    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(2, jnp.int32),
        _add_slots(state, 2, (0, 0), (0, 1)),
    )
    assert state.slot_status[0].tolist() == [SLOT_REMOVED, SLOT_ACTIVE]


def test_released_bucket_is_cleared_and_reused(model, frame_features, step):
    state = pad_tracker_buckets(model.tracker.empty_state(1), 2)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0), (1, 0)),
    )
    release = jnp.asarray([False, True])
    remove = jnp.zeros((2, 2), bool).at[1, 0].set(True)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(1, jnp.int32),
        _plan(state, 1, release=release, remove=remove),
    )
    cleared = jax.tree.map(lambda leaf: leaf[1], state)
    empty = jax.tree.map(lambda leaf: leaf[0], model.tracker.empty_state(1))
    for got, expected in zip(jax.tree.leaves(cleared), jax.tree.leaves(empty)):
        assert jnp.array_equal(got, expected)
    assert state.recent_frame[0].tolist() == [0, 1, -1, -1, -1, -1]

    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(2, jnp.int32),
        _add_slots(state, 2, (1, 1)),
    )
    assert state.bucket_valid.tolist() == [True, True]
    assert state.slot_status[1].tolist() == [SLOT_EMPTY, SLOT_ACTIVE]
    assert state.cond_frame[1].tolist() == [2, -1, -1, -1]
    assert state.recent_frame[1].tolist() == [-1, -1, 2, -1, -1, -1]


def test_padding_appends_cleared_buckets_and_rejects_other_sizes(model):
    state = eqx.tree_at(
        lambda s: s.slot_status,
        model.tracker.empty_state(1),
        jnp.full((1, 2), SLOT_ACTIVE, jnp.int8),
    )
    padded = pad_tracker_buckets(state, 2)
    assert padded.slot_status.tolist() == [
        [SLOT_ACTIVE, SLOT_ACTIVE],
        [SLOT_EMPTY, SLOT_EMPTY],
    ]
    assert padded.cond_frame[1].tolist() == [-1] * 4
    for size in (1, 3):
        with pytest.raises(ValueError, match="power of two"):
            pad_tracker_buckets(state, size)


def test_scan_matches_the_step_by_step_loop(model, frame_features, step):
    state = model.tracker.empty_state(1)
    # Adding to a filled slot does nothing, so one plan works for every frame.
    plan = _add_slots(state, 0, (0, 0))
    frames = jnp.arange(8, dtype=jnp.int32)

    def scanned(model, state, features, frames, plan):
        def body(state, frame):
            state, tracks = _step(
                model,
                state,
                features,
                frame,
                eqx.tree_at(lambda p: p.frame_index, plan, frame),
            )
            return state, tracks.mask_logits

        return jax.lax.scan(body, state, frames)

    scanned_state, scanned_masks = eqx.filter_jit(scanned)(
        model, state, frame_features, frames, plan
    )

    loop_state = state
    loop_masks = []
    for frame in frames:
        frame_plan = eqx.tree_at(lambda p: p.frame_index, plan, frame)
        loop_state, tracks = step(model, loop_state, frame_features, frame, frame_plan)
        loop_masks.append(tracks.mask_logits)

    assert jnp.allclose(
        scanned_masks,
        jnp.stack(loop_masks),
        rtol=_JIT_TOLERANCE,
        atol=_JIT_TOLERANCE,
    )
    for got, expected in zip(
        jax.tree.leaves(scanned_state), jax.tree.leaves(loop_state)
    ):
        if jnp.issubdtype(got.dtype, jnp.inexact):
            assert jnp.allclose(got, expected, rtol=_JIT_TOLERANCE, atol=_JIT_TOLERANCE)
        else:
            assert jnp.array_equal(got, expected)


def test_tracking_gradients_are_finite_and_reach_the_tracker(model, frame_features):
    state = model.tracker.empty_state(1)

    def loss(model, state):
        plan = _add_slots(state, 0, (0, 0))
        state, _ = _step(model, state, frame_features, plan.frame_index, plan)
        tracks = model.propagate(state, frame_features, jnp.asarray(1, jnp.int32))
        return jnp.sum(tracks.mask_logits) + jnp.sum(tracks.pointers)

    gradients = eqx.filter_jit(eqx.filter_grad(loss))(model, state)
    leaves = jax.tree.leaves(eqx.filter(gradients, eqx.is_inexact_array))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert jnp.any(gradients.tracker.memory_time_embed != 0)
    assert any(
        jnp.any(leaf != 0)
        for leaf in jax.tree.leaves(
            eqx.filter(gradients.tracker.pointer_proj, eqx.is_inexact_array)
        )
    )


def test_propagate_runs_under_vmap(model, frame_features):
    state = model.tracker.empty_state(1)
    batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), frame_features)
    tracks = eqx.filter_vmap(
        lambda features, frame: model.propagate(state, features, frame)
    )(batched, jnp.asarray([0, 1], jnp.int32))
    assert tracks.mask_logits.shape == (2, 1, 2, MASK_SIZE, MASK_SIZE)
    assert tracks.pointers.shape == (2, 1, 2, DIM)


def test_tracking_stays_finite_and_keeps_its_dtypes_in_bfloat16(model, step):
    half = _to_bfloat16(model)
    features = half.encode_image(_to_bfloat16(IMAGE))
    state = half.tracker.empty_state(1)
    assert state.cond_memory.dtype == jnp.bfloat16
    state, _ = step(
        half, state, features, jnp.asarray(0, jnp.int32), _add_slots(state, 0, (0, 0))
    )
    assert state.recent_memory.dtype == jnp.bfloat16
    assert state.cond_pointers.dtype == jnp.bfloat16
    assert state.last_masks.dtype == jnp.float32
    tracks = half.propagate(state, features, jnp.asarray(1, jnp.int32))
    assert tracks.mask_logits.dtype == jnp.float32
    assert tracks.pointers.dtype == jnp.bfloat16
    for leaf in (state.recent_memory, tracks.pointers):
        assert jnp.all(jnp.isfinite(leaf.astype(jnp.float32)))


def test_unoccupied_slots_produce_zero_pointers(model, frame_features, step):
    state = model.tracker.empty_state(1)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0)),
    )
    tracks = model.propagate(state, frame_features, jnp.asarray(1, jnp.int32))
    assert jnp.all(tracks.pointers[0, 1] == 0)
    assert jnp.any(tracks.pointers[0, 0] != 0)
    assert jnp.all(state.cond_pointers[0, 0, 1] == 0)
    assert jnp.all(state.last_masks[0, 1] == -1024.0)


def test_commit_reconditions_and_suppresses_the_output_masks(
    model, frame_features, step
):
    state = model.tracker.empty_state(1)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0), (0, 1)),
    )
    replacement = jr.normal(jr.PRNGKey(13), (1, 2, MASK_SIZE, MASK_SIZE))
    plan = _plan(
        state,
        1,
        recondition=jnp.asarray([[True, False]]),
        recondition_masks=replacement,
        suppress=jnp.asarray([[False, True]]),
    )
    state, _ = step(model, state, frame_features, jnp.asarray(1, jnp.int32), plan)
    assert jnp.array_equal(state.last_masks[0, 0], replacement[0, 0])
    assert jnp.all(state.last_masks[0, 1] == -1024.0)
    assert state.cond_frame[0].tolist() == [0, 1, -1, -1]


def test_a_frame_committed_twice_is_read_once(model, frame_features, step):
    frame = jnp.asarray(0, jnp.int32)
    state = model.tracker.empty_state(1)
    state, _ = step(model, state, frame_features, frame, _add_slots(state, 0, (0, 0)))
    tracks = TrackOutput(
        mask_logits=state.last_masks,
        object_score_logits=state.last_scores,
        iou=jnp.zeros_like(state.last_scores),
        pointers=state.pointers[:, 0],
    )
    state = model.commit(state, frame_features, tracks, _plan(state, 0))
    bucket = jax.tree.map(lambda leaf: leaf[0], state)
    _, _, _, spatial, _ = model.tracker._memory_bank(
        bucket, jnp.asarray(1, jnp.int32), 15
    )
    assert state.recent_is_cond[0, 0]
    assert spatial == GRID * GRID


def test_commit_keeps_an_already_decoded_object(model, frame_features):
    frame = jnp.asarray(0, jnp.int32)
    state = model.tracker.empty_state(1)
    prompt = _interactive_prompt()
    mask, pointer, score = model.tracker.decode_object(
        model.interactive, frame_features, prompt
    )
    add = jnp.zeros((1, 2), bool).at[0, 0].set(True)
    plan = _plan(
        state,
        0,
        add=add,
        decoded=add,
        new_masks=jnp.zeros_like(state.last_masks).at[0, 0].set(mask),
        decoded_pointers=jnp.zeros_like(state.pointers[:, 0]).at[0, 0].set(pointer),
        decoded_scores=jnp.zeros_like(state.last_scores)
        .at[0, 0]
        .set(score.astype(jnp.float32)),
    )
    tracks = model.propagate(state, frame_features, frame)
    state = model.commit(state, frame_features, tracks, plan)
    assert jnp.array_equal(state.last_masks[0, 0], mask)
    assert jnp.array_equal(state.pointers[0, 0, 0], pointer)
    assert state.last_scores[0, 0] == score.astype(jnp.float32)
    assert state.slot_status[0].tolist() == [SLOT_ACTIVE, SLOT_EMPTY]


def test_reencoding_changes_only_the_flagged_buckets(model, frame_features, step):
    state = pad_tracker_buckets(model.tracker.empty_state(1), 2)
    state, _ = step(
        model,
        state,
        frame_features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0), (1, 0)),
    )
    frame = jnp.asarray(1, jnp.int32)
    plain, _ = step(model, state, frame_features, frame, _plan(state, 1))
    reencoded, _ = step(
        model,
        state,
        frame_features,
        frame,
        _plan(state, 1, reencode=jnp.asarray([False, True])),
    )
    assert jnp.array_equal(reencoded.recent_memory[0], plain.recent_memory[0])
    assert not jnp.array_equal(reencoded.recent_memory[1], plain.recent_memory[1])


def test_masks_keep_pixels_within_their_group_only():
    masks = jnp.asarray([[[True, True]], [[True, False]], [[False, True]]], dtype=bool)
    kept = _keep_per_group(masks, jnp.asarray([1, 0, 2]), jnp.asarray([0, 0, 1]), 2)
    assert kept[:, 0].tolist() == [[False, True], [True, False], [False, True]]


def test_area_shrinkage_marks_tracks_losing_most_of_their_area():
    order = jnp.asarray([1, 0])
    # The second track outranks the first, which keeps 1 of its 8 pixels here.
    mostly_lost = jnp.where(_row_masks((0, 8), (0, 7)), 1.0, -1.0)
    assert _shrunk_slots(mostly_lost, order).tolist() == [True, False]
    # Here it keeps 4 of 8, above the 30% limit.
    half_lost = jnp.where(_row_masks((0, 8), (0, 4)), 1.0, -1.0)
    assert _shrunk_slots(half_lost, order).tolist() == [False, False]


def _associate(model, *, logits, boxes, detection_masks, track_masks, active):
    status = jnp.where(jnp.asarray(active), SLOT_ACTIVE, SLOT_EMPTY)
    state = eqx.tree_at(
        lambda s: s.slot_status,
        model.tracker.empty_state(track_masks.shape[0]),
        status.astype(jnp.int8),
    )
    tracks = TrackOutput(
        mask_logits=jnp.where(track_masks, 1.0, -1.0),
        object_score_logits=jnp.full(track_masks.shape[:2], 3.0),
        iou=jnp.ones(track_masks.shape[:2]),
        pointers=jnp.zeros((*track_masks.shape[:2], DIM)),
    )
    detections = eqx.tree_at(
        lambda d: d.boxes,
        _output_with_masks(jnp.where(detection_masks, 1.0, -1.0), jnp.asarray(logits)),
        jnp.asarray(boxes),
    )
    return associate(detections, tracks, state)


CENTERED = [0.5, 0.5, 0.1, 0.1]


def test_association_runs_the_boundary_filter_after_nms(model):
    detection_masks = _row_masks((0, 4), (0, 4))
    out = _associate(
        model,
        logits=[3.0, 2.0],
        boxes=[[0.01, 0.5, 0.1, 0.1], CENTERED],
        detection_masks=detection_masks,
        track_masks=_row_masks((0, 1), (0, 1))[None],
        active=[[True, True]],
    )
    # The first detection wins NMS, so the second is gone even though the first
    # is then dropped by the boundary filter.
    assert out.keep.tolist() == [False, False]


def test_new_detections_ignore_the_keep_mask_only_when_nothing_is_tracked(model):
    detection_masks = _row_masks((0, 4), (0, 4))
    kwargs = {
        "logits": [3.0, 2.0],
        "boxes": [CENTERED, CENTERED],
        "detection_masks": detection_masks,
        "track_masks": _row_masks((4, 8), (6, 8))[None],
    }
    untracked = _associate(model, active=[[False, False]], **kwargs)
    tracked = _associate(model, active=[[True, False]], **kwargs)
    assert untracked.keep.tolist() == [True, False]
    assert untracked.new_detection.tolist() == [True, True]
    assert tracked.new_detection.tolist() == [True, False]


def test_association_flags_matches_and_ambiguous_tracks(model):
    out = _associate(
        model,
        logits=[3.0, 3.0],
        boxes=[CENTERED, CENTERED],
        detection_masks=_row_masks((0, 2), (4, 8)),
        track_masks=_row_masks((0, 2), (0, 1))[None],
        active=[[True, True]],
    )
    assert out.keep.tolist() == [True, True]
    assert out.iom.shape == (2, 2)
    assert out.iom[0].tolist() == [1.0, 1.0]
    assert out.matches[0].tolist() == [True, True]
    assert out.high_iom[0].tolist() == [True, True]
    # The detection overlaps two tracks, so both pairs are ambiguous.
    assert out.ambiguous[0].tolist() == [True, True]
    assert out.new_detection.tolist() == [False, True]
    assert out.confident_track.tolist() == [True, True]
    assert jnp.allclose(out.track_iou[0, 1], 0.5)


def test_association_skips_inactive_slots(model):
    out = _associate(
        model,
        logits=[3.0, -5.0],
        boxes=[CENTERED, CENTERED],
        detection_masks=_row_masks((0, 4), (0, 1)),
        track_masks=_row_masks((0, 4), (0, 4))[None],
        active=[[False, True]],
    )
    assert out.iom[0].tolist() == [0.0, 1.0]
    assert out.iom[1].tolist() == [0.0, 0.0]
    assert jnp.all(out.track_iou[0] == 0)


@GPU
def test_cudnn_propagation_matches_xla_under_the_bucket_vmap():
    model = _to_bfloat16(_tiny_model(32))
    fused = with_attention_implementation(model, "cudnn")
    features = model.encode_image(_to_bfloat16(IMAGE))
    state = pad_tracker_buckets(model.tracker.empty_state(1), 2)
    state, _ = _step(
        model,
        state,
        features,
        jnp.asarray(0, jnp.int32),
        _add_slots(state, 0, (0, 0), (1, 1)),
    )
    frame = jnp.asarray(1, jnp.int32)
    reference = eqx.filter_jit(model.propagate)(state, features, frame)
    outputs = eqx.filter_jit(fused.propagate)(state, features, frame)
    assert jnp.allclose(
        outputs.mask_logits, reference.mask_logits, rtol=0.05, atol=0.05
    )


REFERENCE_DATA = Path(__file__).parent / "data"
_PARITY = {"rtol": 1e-5, "atol": 1e-5}


@pytest.fixture(scope="module")
def tiny_reference():
    return np.load(REFERENCE_DATA / "sam3_tiny_reference.npz")


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        "sam31_converter", Path(__file__).resolve().parents[1] / "models" / "sam31.py"
    )
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    return converter


def _checkpoint_state(tiny_reference) -> dict[str, np.ndarray]:
    return {
        name.removeprefix("state."): tiny_reference[name]
        for name in tiny_reference.files
        if name.startswith("state.")
    }


def _reference_config() -> dict:
    path = REFERENCE_DATA / "sam3_tiny_reference.json"
    return json.loads(path.read_text())["config"]


def _reference_template() -> Sam3:
    config = _reference_config()
    return Sam3(
        key=KEY,
        in_channels=3,
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_features=config["mlp_features"],
        window_size=config["window_size"],
        global_blocks=tuple(config["global_blocks"]),
        dim=config["dim"],
        head_num_heads=config["head_num_heads"],
        ffn_features=config["ffn_features"],
        geometry_depth=config["geometry_depth"],
        fusion_depth=config["fusion_depth"],
        decoder_depth=config["decoder_depth"],
        num_queries=config["num_queries"],
        two_way_depth=config["two_way_depth"],
        memory_depth=config["memory_depth"],
        memory_heads=config["memory_heads"],
        num_slots=config["num_slots"],
        memory_start_channels=config["memory_start_channels"],
        memory_fuser_depth=1,
    )


@pytest.fixture(scope="module")
def reference_model(tiny_reference) -> Sam3:
    return _load_converter().apply_state(
        _reference_template(), _checkpoint_state(tiny_reference), "multiplex"
    )


def _assert_close(actual, expected) -> None:
    np.testing.assert_allclose(np.asarray(actual), expected, **_PARITY)


def test_tiny_reference_state_fills_every_parameter_or_is_listed(tiny_reference):
    converter = _load_converter()
    state = _checkpoint_state(tiny_reference)
    classification = converter.classify_keys(state)
    leaves = jax.tree_util.tree_flatten_with_path(
        eqx.filter(_reference_template(), eqx.is_inexact_array)
    )[0]
    template = {
        stringify_name(path): tuple(leaf.shape)
        for path, leaf in leaves
        if leaf is not None
    }
    missing = converter.check_destinations(
        classification,
        {name: tuple(value.shape) for name, value in state.items()},
        template,
        "multiplex",
    )
    assert missing == []


def test_trunk_matches_the_reference(reference_model, tiny_reference):
    output = reference_model.trunk(
        jnp.asarray(tiny_reference["inputs.trunk.args.0"][0])
    )
    _assert_close(output, tiny_reference["outputs.trunk.0"][0])


def test_neck_matches_the_reference(reference_model, tiny_reference):
    features = reference_model.neck(jnp.asarray(tiny_reference["outputs.trunk.0"][0]))
    levels = {"detector": 0, "interactive": 2, "propagation": 4}
    for name, index in levels.items():
        for level, value in enumerate(getattr(features, name)):
            _assert_close(value, tiny_reference[f"outputs.neck.{index}.{level}"][0])
            size = value.shape[-1]
            _assert_close(
                sine_position_encoding(size, size, value.shape[0] // 2),
                tiny_reference[f"outputs.neck.{index + 1}.{level}"][0],
            )


def test_geometry_encoder_matches_the_reference_without_exemplars(
    reference_model, tiny_reference
):
    tokens = tiny_reference["inputs.detector.geometry.kwargs.img_feats.0"][:, 0]
    position = tiny_reference["inputs.detector.geometry.kwargs.img_pos_embeds.0"][:, 0]
    prompt = ConceptPrompt.create(dim=DIM, max_text_tokens=1, max_boxes=1, max_points=1)
    output, valid = reference_model.detector.geometry(
        jnp.asarray(tokens), jnp.asarray(position), GRID, prompt
    )
    assert valid.tolist() == [False, False, True]
    _assert_close(output[-1:], tiny_reference["outputs.detector.geometry.0"][:, 0])


def test_fusion_encoder_matches_the_reference(reference_model, tiny_reference):
    fusion = "inputs.detector.head.fusion.kwargs"
    memory = reference_model.detector.head.fusion(
        flatten_map(jnp.asarray(tiny_reference[f"{fusion}.src.0"][0])),
        flatten_map(jnp.asarray(tiny_reference[f"{fusion}.src_pos.0"][0])),
        jnp.asarray(tiny_reference[f"{fusion}.prompt"][:, 0]),
        ~jnp.asarray(tiny_reference[f"{fusion}.prompt_key_padding_mask"][0]),
    )
    _assert_close(memory, tiny_reference["outputs.detector.head.fusion.memory"][:, 0])


def test_decoder_matches_the_reference(reference_model, tiny_reference):
    inputs = "inputs.detector.head.decoder.kwargs"
    decoder = reference_model.detector.head.decoder
    _assert_close(decoder.queries, tiny_reference[f"{inputs}.tgt"][:, 0])
    tokens, boxes, presence = decoder(
        jnp.asarray(tiny_reference[f"{inputs}.memory"][:, 0]),
        jnp.asarray(tiny_reference[f"{inputs}.pos"][:, 0]),
        jnp.asarray(tiny_reference[f"{inputs}.memory_text"][:, 0]),
        ~jnp.asarray(tiny_reference[f"{inputs}.text_attention_mask"][0]),
        GRID,
    )
    _assert_close(tokens, tiny_reference["outputs.detector.head.decoder.0"][:, :, 0])
    reference_boxes = tiny_reference["outputs.detector.head.decoder.1"]
    _assert_close(jax.nn.sigmoid(decoder.reference_boxes), reference_boxes[0, :, 0])
    _assert_close(boxes[0], reference_boxes[1, :, 0])
    _assert_close(presence, tiny_reference["outputs.detector.head.decoder.2"][:, 0, 0])


def test_scoring_matches_the_reference(reference_model, tiny_reference):
    inputs = "inputs.detector.head.scoring.args"
    logits = jax.vmap(reference_model.detector.head.scoring, in_axes=(0, None, None))(
        jnp.asarray(tiny_reference[f"{inputs}.0"][:, 0]),
        jnp.asarray(tiny_reference[f"{inputs}.1"][:, 0]),
        ~jnp.asarray(tiny_reference[f"{inputs}.2"][0]),
    )
    _assert_close(logits, tiny_reference["outputs.detector.head.scoring"][:, 0, :, 0])


def test_segmentation_matches_the_reference(reference_model, tiny_reference):
    inputs = "inputs.detector.head.segmentation.kwargs"
    head = reference_model.detector.head.segmentation
    pyramid = tuple(
        jnp.asarray(tiny_reference[f"{inputs}.backbone_feats.{level}"][0])
        for level in range(3)
    )
    memory = jnp.asarray(tiny_reference[f"{inputs}.encoder_hidden_states"][:, 0])
    prompt = jnp.asarray(tiny_reference[f"{inputs}.prompt"][:, 0])
    valid = ~jnp.asarray(tiny_reference[f"{inputs}.prompt_mask"][0])
    queries = jnp.asarray(tiny_reference[f"{inputs}.obj_queries"][-1, 0])

    masks = head(pyramid, memory, queries, prompt, valid)
    pixels = head.pixel_embedding(pyramid, memory, prompt, valid)

    outputs = "outputs.detector.head.segmentation"
    _assert_close(masks, tiny_reference[f"{outputs}.pred_masks"][0])
    _assert_close(
        head.semantic_logits(pixels), tiny_reference[f"{outputs}.semantic_seg"][0]
    )


def test_prompt_encoder_matches_the_reference_for_masks_and_boxes(
    reference_model, tiny_reference
):
    encoder = reference_model.interactive.prompt_encoder
    points = "inputs.interactive.prompt_encoder.kwargs.points"
    coordinates = tiny_reference[f"{points}.0"]
    labels = tiny_reference[f"{points}.1"]
    masks = tiny_reference["inputs.interactive.prompt_encoder.kwargs.masks"]
    for index in range(coordinates.shape[0]):
        prompt = InteractivePrompt.create(
            points=jnp.asarray(coordinates[index]),
            point_labels=jnp.asarray(labels[index]),
            mask_input=jnp.asarray(masks[index, 0]),
            mask_size=masks.shape[-1],
            max_points=1,
        )
        tokens, valid, dense = encoder(prompt)
        _assert_close(
            tokens[valid], tiny_reference["outputs.interactive.prompt_encoder.0"][index]
        )
        _assert_close(
            dense, tiny_reference["outputs.interactive.prompt_encoder.1"][index]
        )

    corners = tiny_reference["inputs.interactive.box_prompt.points"]
    prompt = InteractivePrompt.create(
        box=jnp.asarray(corners[0].ravel()),
        mask_size=masks.shape[-1],
        max_points=1,
    )
    tokens, valid, dense = encoder(prompt)
    _assert_close(
        tokens[valid], tiny_reference["outputs.interactive.box_prompt.sparse"][0]
    )
    _assert_close(dense, tiny_reference["outputs.interactive.box_prompt.dense"][0])


def test_interactive_decoder_matches_the_reference(reference_model, tiny_reference):
    inputs = "inputs.interactive.mask_decoder.kwargs"
    outputs = "outputs.interactive.mask_decoder"
    image_pe = tiny_reference[f"{inputs}.image_pe"][0]
    _assert_close(
        reference_model.interactive.prompt_encoder.dense_position_encoding(), image_pe
    )
    sparse = tiny_reference[f"{inputs}.sparse_prompt_embeddings"]
    dense = tiny_reference[f"{inputs}.dense_prompt_embeddings"]
    for index in range(sparse.shape[0]):
        masks, iou, object_score, tokens = reference_model.interactive.mask_decoder(
            jnp.asarray(tiny_reference[f"{inputs}.image_embeddings"][0]),
            jnp.asarray(image_pe),
            jnp.asarray(sparse[index]),
            jnp.ones(sparse.shape[1], bool),
            jnp.asarray(dense[index]),
            (
                jnp.asarray(tiny_reference[f"{inputs}.high_res_features.0"][0]),
                jnp.asarray(tiny_reference[f"{inputs}.high_res_features.1"][0]),
            ),
        )
        # The reference call asks for one mask, so stability picks which.
        selected = _select_mask(masks, iou, jnp.asarray(2), jnp.asarray(True))
        _assert_close(masks[selected], tiny_reference[f"{outputs}.0"][index, 0])
        _assert_close(iou[selected], tiny_reference[f"{outputs}.1"][index, 0])
        _assert_close(tokens[0], tiny_reference[f"{outputs}.2"][index, 0])
        _assert_close(object_score, tiny_reference[f"{outputs}.3"][index, 0])


def test_multiplex_decoder_matches_the_reference(reference_model, tiny_reference):
    inputs = "inputs.tracker.mask_decoder.kwargs"
    outputs = "outputs.tracker.mask_decoder"
    decoder = reference_model.tracker.mask_decoder
    extra = tiny_reference[f"{inputs}.extra_per_object_embeddings"][0]
    is_valid = np.isclose(extra, np.asarray(decoder.valid_embed), rtol=1e-6).all(-1)
    is_invalid = np.isclose(extra, np.asarray(decoder.invalid_embed), rtol=1e-6).all(-1)
    assert np.logical_xor(is_valid, is_invalid).all()
    image_pe = tiny_reference[f"{inputs}.image_pe"][0]
    _assert_close(reference_model.tracker.image_position(jnp.float32), image_pe)

    output = decoder(
        jnp.asarray(tiny_reference[f"{inputs}.image_embeddings"][0]),
        jnp.asarray(image_pe),
        (
            jnp.asarray(tiny_reference[f"{inputs}.high_res_features.0"][0]),
            jnp.asarray(tiny_reference[f"{inputs}.high_res_features.1"][0]),
        ),
        jnp.asarray(is_valid),
    )
    scores = tiny_reference[f"{outputs}.object_score_logits"][0, :, 0]
    expected_masks = np.where(
        scores[:, None, None, None] > 0, tiny_reference[f"{outputs}.masks"][0], -1024.0
    )
    _assert_close(output.mask_logits, expected_masks)
    _assert_close(output.iou, tiny_reference[f"{outputs}.iou_pred"][0])
    _assert_close(output.object_score_logits, scores)
    _assert_close(output.mask_tokens, tiny_reference[f"{outputs}.sam_tokens_out"][0])


def test_memory_encoder_matches_the_reference(reference_model, tiny_reference):
    output = reference_model.tracker.memory_encoder(
        jnp.asarray(tiny_reference["inputs.tracker.memory_encoder.args.0"][0]),
        jnp.asarray(tiny_reference["inputs.tracker.memory_encoder.args.1"][0]),
    )
    _assert_close(
        output, tiny_reference["outputs.tracker.memory_encoder.vision_features"][0]
    )


def test_memory_attention_matches_the_reference(reference_model, tiny_reference):
    inputs = "inputs.tracker.memory_attention.kwargs"
    memory = jnp.asarray(tiny_reference[f"{inputs}.memory"][:, 0])
    memory_pos = jnp.asarray(tiny_reference[f"{inputs}.memory_pos"][:, 0])
    memory_image = jnp.asarray(tiny_reference[f"{inputs}.memory_image"][:, 0])
    memory_image_pos = jnp.asarray(tiny_reference[f"{inputs}.memory_image_pos"][:, 0])
    spatial = memory_image.shape[0]
    pointers = memory.shape[0] - spatial
    output = reference_model.tracker.memory_attention(
        jnp.asarray(tiny_reference[f"{inputs}.image"][:, 0]),
        jnp.asarray(tiny_reference[f"{inputs}.src_pos"][:, 0]),
        jnp.concatenate((memory_image, jnp.zeros((pointers, DIM)))),
        memory,
        jnp.concatenate((memory_image_pos, memory_pos[spatial:])),
        num_spatial_keys=jnp.asarray(spatial),
        key_lengths=jnp.asarray(memory.shape[0]),
    )
    _assert_close(
        output, tiny_reference["outputs.tracker.memory_attention.memory"][:, 0]
    )


@pytest.mark.parametrize("case", ["full", "padded"])
def test_geometry_encoder_matches_the_reference_with_exemplars(
    reference_model, tiny_reference, case
):
    image = "inputs.detector.geometry.kwargs"
    prefix = f"inputs.detector.geometry_prompt.{case}"
    padding = tiny_reference[f"{prefix}.padding"][0]
    labels = tiny_reference[f"{prefix}.labels"][:, 0]
    prompt = ConceptPrompt(
        text_features=jnp.zeros((1, DIM)),
        text_padding_mask=jnp.ones(1, bool),
        boxes=jnp.asarray(tiny_reference[f"{prefix}.boxes"][:, 0]),
        box_labels=jnp.asarray(labels, jnp.int32),
        box_padding_mask=jnp.asarray(padding),
        points=jnp.asarray(tiny_reference[f"{prefix}.points"][:, 0]),
        point_labels=jnp.asarray(labels, jnp.int32),
        point_padding_mask=jnp.asarray(padding),
    )

    tokens, valid = reference_model.detector.geometry(
        jnp.asarray(tiny_reference[f"{image}.img_feats.0"][:, 0]),
        jnp.asarray(tiny_reference[f"{image}.img_pos_embeds.0"][:, 0]),
        GRID,
        prompt,
    )

    # The reference packs the tokens in use first; Equimo keeps padded slots.
    reference_tokens = tiny_reference[f"outputs.detector.geometry_prompt.{case}.0"][
        :, 0
    ]
    reference_padding = tiny_reference[f"outputs.detector.geometry_prompt.{case}.1"][0]
    assert jnp.sum(valid) == np.sum(~reference_padding)
    _assert_close(tokens[valid], reference_tokens[~reference_padding])


def test_joint_score_and_boxes_match_the_reference(reference_model, tiny_reference):
    head = reference_model.detector.head
    inputs = "inputs.detector.head.decoder.kwargs"
    tokens, boxes, presence = head.decoder(
        jnp.asarray(tiny_reference[f"{inputs}.memory"][:, 0]),
        jnp.asarray(tiny_reference[f"{inputs}.pos"][:, 0]),
        jnp.asarray(tiny_reference[f"{inputs}.memory_text"][:, 0]),
        ~jnp.asarray(tiny_reference[f"{inputs}.text_attention_mask"][0]),
        GRID,
    )
    logits = head.scoring(
        tokens[-1],
        jnp.asarray(tiny_reference["inputs.detector.head.scoring.args.1"][:, 0]),
        ~jnp.asarray(tiny_reference["inputs.detector.head.scoring.args.2"][0]),
    )

    joint = joint_logit(logits, presence[-1])

    _assert_close(joint, tiny_reference["outputs.detector.joint.pred_logits"][0, :, 0])
    _assert_close(boxes[-1], tiny_reference["outputs.detector.joint.pred_boxes"][0])
    _assert_close(
        presence[-1],
        tiny_reference["outputs.detector.joint.presence_logit_dec"].reshape(()),
    )


def test_mask_nms_keeps_what_the_reference_keeps(tiny_reference):
    probabilities = jnp.asarray(tiny_reference["inputs.nms.probabilities"])
    masks = jnp.asarray(tiny_reference["inputs.nms.masks"]) > 0
    candidates = probabilities > 0.4

    batched = greedy_iom_nms(masks, probabilities, candidates, 0.1)
    np.testing.assert_array_equal(batched, tiny_reference["outputs.nms.keep_batched"])

    order = jnp.argsort(-probabilities, stable=True)
    single = (
        jnp.zeros_like(candidates)
        .at[order]
        .set(candidates[order] & single_frame_nms(masks[order]))
    )
    np.testing.assert_array_equal(single, tiny_reference["outputs.nms.keep"])
