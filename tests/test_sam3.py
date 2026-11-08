"""SAM 3.1 image and tracker modules on a tiny random-weight configuration."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from equimo.core.layers.attention import with_attention_implementation
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
    ConceptPrompt,
    DetectorOutput,
    InteractivePrompt,
    Sam3,
    decode_concepts,
)
from equimo.vision.models.sam3 import tracker as tracker_module
from equimo.vision.models.sam3.common import exchanged_rope_factors
from equimo.vision.models.sam3.decoding import greedy_iom_nms, single_frame_nms
from equimo.vision.models.sam3.detector import joint_logit
from equimo.vision.models.sam3.encoder import ViTDetBlock
from equimo.vision.models.sam3.interactive import _select_mask
from equimo.vision.models.sam3.tracker import _memory_key_rotary

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
    sparse, dense = tracker.prompt_encoder.empty()
    multiplex = tracker.mask_decoder(
        low,
        tracker.prompt_encoder.dense_position_encoding(),
        sparse,
        dense,
        features.propagation[:2],
        jnp.asarray([True, False]),
    )
    attended = tracker.memory_attention(
        jnp.moveaxis(low, 0, -1).reshape(GRID * GRID, dim),
        memory,
        jnp.zeros_like(memory),
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
    assert _select_mask(logits, iou, jnp.asarray(1)) == 2
    assert _select_mask(logits, iou, jnp.asarray(2)) == 0

    unstable = jnp.zeros((4, 4))
    unstable = unstable.at[0, 0].set(1.0)
    logits = jnp.stack([unstable] * 4)
    assert _select_mask(logits, iou, jnp.asarray(2)) == 2


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
    b[0, 3:6] = 1.0  # one pixel of its three overlaps ``a`` only in the first row
    c[3, 6:8] = 1.0
    out = _output_with_masks(
        jnp.asarray(np.stack((c, a, b)), jnp.float32),
        jnp.asarray([-3.0, 2.0, 0.5], jnp.float32),
    )
    decoded = decode_concepts(out, output_size=(4, 8), max_detections=3)

    # ``c`` falls below the threshold; ``a`` outranks ``b``, which overlaps it by a
    # third of its area and survives NMS only because overlap uses ``a``'s area.
    assert decoded.keep.tolist() == [True, True, False]
    assert decoded.query_index[:2].tolist() == [1, 2]
    assert jnp.allclose(
        decoded.scores[:2], jax.nn.sigmoid(jnp.asarray([2.0, 0.5])), rtol=1e-6
    )
    assert jnp.allclose(decoded.presence, jax.nn.sigmoid(jnp.asarray(3.0)))
    assert decoded.boxes[0].tolist() == [0.0, 0.0, 3.0, 3.0]
    assert decoded.boxes[1].tolist() == [3.0, 0.0, 5.0, 0.0]
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
    base = attention(
        x,
        MEMORY,
        jnp.zeros_like(MEMORY),
        num_spatial_keys=jnp.asarray(GRID * GRID),
        key_lengths=lengths,
    )
    changed_memory = MEMORY.at[GRID * GRID + 1 :].set(100.0)
    other = attention(
        x,
        changed_memory,
        jnp.zeros_like(MEMORY),
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
    whole = attention(x, MEMORY, jnp.zeros_like(MEMORY), **kwargs)
    monkeypatch.setattr(tracker_module, "_ATTENTION_CHUNK_QUERIES", 3)
    chunked = attention(x, MEMORY, jnp.zeros_like(MEMORY), **kwargs)
    assert jnp.allclose(whole, chunked, rtol=1e-6, atol=1e-6)


def test_slot_without_an_object_gets_the_no_object_logit(model):
    tracker = model.tracker
    low = jr.normal(jr.PRNGKey(8), (DIM, GRID, GRID))
    high = jr.normal(jr.PRNGKey(9), (DIM, MASK_SIZE, MASK_SIZE))
    middle = jr.normal(jr.PRNGKey(10), (DIM, 2 * GRID, 2 * GRID))
    sparse, dense = tracker.prompt_encoder.empty()
    output = tracker.mask_decoder(
        low,
        tracker.prompt_encoder.dense_position_encoding(),
        sparse,
        dense,
        (high, middle),
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
    sparse, dense = tracker.prompt_encoder.empty()

    def decode(valid):
        return tracker.mask_decoder(
            low,
            tracker.prompt_encoder.dense_position_encoding(),
            sparse,
            dense,
            (high, middle),
            jnp.asarray(valid),
        )

    a = decode([True, False])
    b = decode([False, True])
    assert not jnp.array_equal(a.mask_tokens[0], b.mask_tokens[0])


def test_fourier_matrices_are_never_trained(model):
    mask = trainable_mask(model)
    assert mask.interactive.prompt_encoder.fourier_matrix is False
    assert mask.tracker.prompt_encoder.fourier_matrix is False
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
