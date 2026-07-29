"""PEFT method coverage."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


def test_lora_fa_identity_and_trainable_B_only(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_lora_fa(
        tiny_vision_transformer,
        eqft.LoRAFAConfig(rank=2, alpha=4.0),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="lora_fa",
            train_head=False,
        ),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert plan.trainable.blocks[0].attn.qkv.lora_fa_B is not None
    assert plan.trainable.blocks[0].attn.qkv.frozen_A is None
    assert plan.trainable.blocks[0].attn.qkv.correction_matrix is None
    assert set(plan.report.trainable_by_label) == {"lora_fa_B_decay"}


def test_adalora_orthogonality_aux_loss_spec_is_explicit(tiny_vision_transformer):
    model = eqft.apply_adalora(
        tiny_vision_transformer,
        eqft.AdaLoRAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(11),
    )
    module_loss = model.blocks[0].attn.proj.orthogonality_loss()
    spec = eqft.adalora_orthogonality_aux_loss_spec(coefficient_hint=0.25)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="adalora",
            train_head=False,
        ),
        aux_losses=(spec,),
    )

    assert jnp.allclose(eqft.adalora_orthogonality_loss(model), module_loss)
    assert jnp.allclose(
        eqft.adalora_orthogonality_loss(model, coefficient=0.25),
        module_loss * 0.25,
    )
    assert plan.aux_losses == (spec,)
    assert spec.registry_key == "equimo.adalora_orthogonality_loss"
    assert spec.reduction == "sum"


def test_adalora_rank_pattern_zeros_singulars_without_persistent_mask(
    tiny_vision_transformer,
):
    model = eqft.apply_adalora(
        tiny_vision_transformer,
        eqft.AdaLoRAConfig(
            rank=3,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(12),
    )
    path, module = eqft.iter_adalora_modules(model)[0]
    singular = jnp.asarray([1.0, 2.0, 3.0], dtype=module.singular.dtype)
    model = eqx.tree_at(
        lambda tree: eqft.iter_adalora_modules(tree)[0][1].singular,
        model,
        singular,
    )
    rank_groups = eqft.lora_rank_groups(model)
    name = eqft.path_to_str(path)

    masked = eqft.apply_lora_rank_pattern(
        model,
        {name: jnp.asarray([True, False, True])},
    )
    masked_module = eqft.iter_adalora_modules(masked)[0][1]

    assert rank_groups == {name: 3}
    assert masked_module.final_mask is None
    assert jnp.allclose(masked_module.singular, jnp.asarray([1.0, 0.0, 3.0]))

    final = eqft.apply_lora_rank_pattern(
        model,
        {name: jnp.asarray([False, True, True])},
        final=True,
    )
    final_module = eqft.iter_adalora_modules(final)[0][1]

    assert jnp.array_equal(final_module.final_mask, jnp.asarray([False, True, True]))
    assert jnp.allclose(final_module.singular, jnp.asarray([0.0, 2.0, 3.0]))


def test_lora_fa_custom_vjp_freezes_A_and_corrects_B_gradient(tiny_vision_transformer):
    x = jnp.ones((4,), dtype=jnp.float32)
    model = eqft.apply_lora_fa(
        tiny_vision_transformer,
        eqft.LoRAFAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(0),
    )
    module = model.blocks[0].attn.proj

    grads = eqx.filter_grad(lambda m: jnp.sum(m(x)))(module)
    z = module.frozen_A @ x
    raw_grad_B = jnp.outer(jnp.ones((4,), dtype=jnp.float32), z) * module.scaling
    expected_B = (raw_grad_B @ module.correction_matrix) / (module.scaling**2)

    assert jnp.array_equal(grads.frozen_A, jnp.zeros_like(module.frozen_A))
    assert jnp.allclose(grads.lora_fa_B, expected_B, atol=1e-6)


def test_fourierft_identity_trainables_and_merge_equivalence(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_fourierft(
        tiny_vision_transformer,
        eqft.FourierFTConfig(
            num_coefficients=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(1),
    )
    module = model.blocks[0].attn.proj
    trained_module = eqx.tree_at(
        lambda m: m.coefficients_real,
        module,
        jnp.ones_like(module.coefficients_real) * 0.05,
    )
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_fourierft(trained)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="fourierft",
            train_head=False,
        ),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert module.frequency_selection == "random"
    assert (
        module.transform_normalization
        == "jax.numpy.fft.ifft default 1/n inverse scaling"
    )
    assert module.reshape_convention == "row_major_flatten_out_in"
    assert plan.trainable.blocks[0].attn.proj.coefficients_real is not None
    assert plan.trainable.blocks[0].attn.proj.frequency_indices is None
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


def test_fourierft_deduplicates_conjugate_frequencies(tiny_vision_transformer):
    model = eqft.apply_fourierft(
        tiny_vision_transformer,
        eqft.FourierFTConfig(
            num_coefficients=2,
            frequency_selection="explicit",
            frequency_indices=(1, -1),
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj

    assert module.frequency_indices.shape == (1,)
    assert module.frequency_selection == "explicit"
    assert (
        module.delta_weight().dtype
        == tiny_vision_transformer.blocks[0].attn.proj.weight.dtype
    )


def test_eva_initializes_lora_A_from_activation_artifacts(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    artifacts = {
        "blocks.0.attn.proj": jnp.eye(4, dtype=jnp.float32),
        "blocks.1.attn.proj": jnp.eye(4, dtype=jnp.float32),
    }
    model = eqft.apply_eva_lora(
        tiny_vision_transformer,
        eqft.EVAInitializerConfig(rank_budget=4, per_layer_min_rank=1),
        activation_artifacts=artifacts,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(2),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert model.blocks[0].attn.proj.lora_A.shape[0] >= 1
    assert jnp.array_equal(
        model.blocks[0].attn.proj.lora_B,
        jnp.zeros_like(model.blocks[0].attn.proj.lora_B),
    )


def test_eva_accepts_valid_calibration_artifacts(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    artifacts = {
        "blocks.0.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.0.attn.proj",),
            statistics=jnp.eye(4, dtype=jnp.float32),
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
        "blocks.1.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.1.attn.proj",),
            statistics=jnp.eye(4, dtype=jnp.float32),
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
    }
    model = eqft.apply_eva_lora(
        tiny_vision_transformer,
        eqft.EVAInitializerConfig(
            rank_budget=4,
            per_layer_min_rank=1,
            calibration=eqft.CalibrationSpec(
                artifact_kind="activation_svd",
                sample_count=8,
                data_fingerprint="dataset-a",
            ),
        ),
        activation_artifacts=artifacts,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(22),
    )

    metadata = dict(model.blocks[0].attn.proj.metadata)
    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert metadata["method"] == "eva"
    assert metadata["calibration_sample_count"] == "8"
    assert metadata["calibration_data_fingerprint"] == "dataset-a"


def test_eva_consumes_activation_svd_statistics(tiny_vision_transformer):
    svd_payload = {
        "right_singular_vectors": jnp.eye(4, dtype=jnp.float32),
        "singular_values": jnp.asarray([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32),
    }
    artifacts = {
        "blocks.0.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.0.attn.proj",),
            statistics=svd_payload,
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
        "blocks.1.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.1.attn.proj",),
            statistics=svd_payload,
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
    }

    model = eqft.apply_eva_lora(
        tiny_vision_transformer,
        eqft.EVAInitializerConfig(
            rank_budget=2,
            calibration=eqft.CalibrationSpec(artifact_kind="activation_svd"),
        ),
        activation_artifacts=artifacts,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(25),
    )

    assert model.blocks[0].attn.proj.lora_A.shape == (1, 4)
    assert model.blocks[1].attn.proj.lora_A.shape == (1, 4)


def test_eva_rank_allocation_ties_use_logical_parameter_id(tiny_vision_transformer):
    svd_payload = {
        "right_singular_vectors": jnp.eye(4, dtype=jnp.float32),
        "singular_values": jnp.asarray([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32),
    }
    artifacts = {
        "blocks.0.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.0.attn.proj",),
            statistics=svd_payload,
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
        "blocks.1.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.1.attn.proj",),
            statistics=svd_payload,
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
    }

    model = eqft.apply_eva_lora(
        tiny_vision_transformer,
        eqft.EVAInitializerConfig(
            rank_budget=3,
            calibration=eqft.CalibrationSpec(artifact_kind="activation_svd"),
        ),
        activation_artifacts=artifacts,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(26),
    )

    assert model.blocks[0].attn.proj.lora_A.shape == (2, 4)
    assert model.blocks[1].attn.proj.lora_A.shape == (1, 4)


def test_eva_rejects_non_artifact_when_calibration_is_pinned(tiny_vision_transformer):
    artifacts = {
        "blocks.0.attn.proj": jnp.eye(4, dtype=jnp.float32),
        "blocks.1.attn.proj": jnp.eye(4, dtype=jnp.float32),
    }

    with pytest.raises(ValueError, match="CalibrationArtifact"):
        eqft.apply_eva_lora(
            tiny_vision_transformer,
            eqft.EVAInitializerConfig(
                rank_budget=4,
                calibration=eqft.CalibrationSpec(artifact_kind="activation_svd"),
            ),
            activation_artifacts=artifacts,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
            key=jr.PRNGKey(23),
        )


def test_eva_rejects_calibration_fingerprint_mismatch(tiny_vision_transformer):
    artifacts = {
        "blocks.0.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.0.attn.proj",),
            statistics=jnp.eye(4, dtype=jnp.float32),
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
        "blocks.1.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.1.attn.proj",),
            statistics=jnp.eye(4, dtype=jnp.float32),
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
    }

    with pytest.raises(ValueError, match="data_fingerprint mismatch"):
        eqft.apply_eva_lora(
            tiny_vision_transformer,
            eqft.EVAInitializerConfig(
                rank_budget=4,
                calibration=eqft.CalibrationSpec(
                    artifact_kind="activation_svd",
                    data_fingerprint="dataset-b",
                ),
            ),
            activation_artifacts=artifacts,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
            key=jr.PRNGKey(24),
        )


def test_orthogonal_adapter_identity_trainables_and_merge(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_orthogonal_adapters(
        tiny_vision_transformer,
        eqft.OrthogonalAdapterConfig(
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj
    skew = module.skew.at[0, 1].set(0.01)
    trained_module = eqx.tree_at(lambda m: m.skew, module, skew)
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_orthogonal_adapters(trained)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="orthogonal",
            train_head=False,
        ),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert module.orthogonality_error() < 1e-8
    assert plan.trainable.blocks[0].attn.proj.skew is not None
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


def test_boft_blockwise_forward_matches_dense_merge(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_orthogonal_adapters(
        tiny_vision_transformer,
        eqft.OrthogonalAdapterConfig(
            parameterization="butterfly_cayley",
            block_size=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj
    skew = module.skew.at[0, 0, 1].set(0.02)
    trained_module = eqx.tree_at(lambda m: m.skew, module, skew)
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_orthogonal_adapters(trained)

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


def test_boft_adapter_delta_round_trips_with_ordering_metadata(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_orthogonal_adapters(
        tiny_vision_transformer,
        eqft.OrthogonalAdapterConfig(
            parameterization="butterfly_cayley",
            block_size=2,
            num_factors=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj
    skew = module.skew.at[0, 0, 0, 1].set(0.02)
    skew = skew.at[1, 0, 0, 1].set(0.03)
    trained_module = eqx.tree_at(lambda m: m.skew, module, skew)
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_orthogonal_adapters(trained)

    bundle = eqft.extract_adapter_delta(trained)
    loaded = eqft.load_adapter_delta(tiny_vision_transformer, bundle)
    entry = next(
        entry
        for entry in bundle.adapter_config["entries"]
        if entry["class"] == "OrthogonalLinear"
    )
    serialization = entry["orthogonal"]["serialization"]

    assert serialization["layout"] == "sparse_butterfly_cayley"
    assert serialization["block_order"] == (0, 1)
    assert serialization["factor_order"] == (0, 1)
    assert serialization["butterfly_permutation"] == ((0, 1, 2, 3), (0, 2, 1, 3))
    assert jnp.allclose(trained(x), loaded(x), atol=1e-6)
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


@pytest.mark.parametrize(
    ("parameterization", "block_size", "num_factors", "skew_index"),
    (
        ("cayley", None, 1, (0, 1)),
        ("butterfly_cayley", 2, 2, (0, 0, 0, 1)),
    ),
)
def test_orthogonal_adapter_delta_round_trips_from_merged_model(
    tmp_path,
    tiny_vision_transformer,
    parameterization,
    block_size,
    num_factors,
    skew_index,
):
    x = jnp.ones((2, 3))
    model = eqft.apply_orthogonal_adapters(
        tiny_vision_transformer,
        eqft.OrthogonalAdapterConfig(
            parameterization=parameterization,
            block_size=block_size,
            num_factors=num_factors,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj
    trained_module = eqx.tree_at(
        lambda m: m.skew,
        module,
        module.skew.at[skew_index].set(0.05),
    )
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_orthogonal_adapters(trained)

    unmerged_bundle = eqft.save_delta(
        trained,
        tmp_path / "unmerged.eqft",
        method="adapter",
    )
    merged_bundle = eqft.save_delta(
        merged,
        tmp_path / "merged.eqft",
        method="adapter",
    )
    loaded = eqft.load_delta(tiny_vision_transformer, tmp_path / "merged.eqft")
    merged_entry = next(
        entry
        for entry in merged_bundle.adapter_config["entries"]
        if entry["class"] == "OrthogonalLinear"
    )

    assert merged_entry["orthogonal"]["merged"] is False
    assert jnp.allclose(loaded(x), merged(x), atol=1e-6)
    assert not jnp.allclose(loaded(x), tiny_vision_transformer(x), atol=1e-6)
    assert merged_bundle.base_checkpoint_id == unmerged_bundle.base_checkpoint_id
    assert (
        merged_bundle.lineage.base_checkpoint_hash
        == unmerged_bundle.lineage.base_checkpoint_hash
    )
    assert (
        merged_bundle.lineage.base_value_hash == unmerged_bundle.lineage.base_value_hash
    )


def test_boft_requires_explicit_block_size(tiny_vision_transformer):
    with pytest.raises(ValueError, match="block_size"):
        eqft.apply_orthogonal_adapters(
            tiny_vision_transformer,
            eqft.OrthogonalAdapterConfig(
                parameterization="butterfly_cayley",
                target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
            ),
        )


def test_randlora_dense_rank_and_serialized_random_bases(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_randlora(
        tiny_vision_transformer,
        eqft.RandLoRAConfig(
            rank=1,
            basis_count=4,
            alpha=1.0,
            seed=123,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(6),
    )
    module = model.blocks[0].attn.proj
    trained_module = eqx.tree_at(
        lambda m: m.basis_scales,
        module,
        jnp.ones_like(module.basis_scales),
    )
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_randlora(trained)
    bundle = eqft.extract_lora_delta(trained)
    loaded = eqft.load_lora_delta(tiny_vision_transformer, bundle)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="randlora",
            train_head=False,
        ),
    )
    entry = next(
        entry
        for entry in bundle.adapter_config["entries"]
        if entry["class"] == "RandLoRALinear"
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert plan.trainable.blocks[0].attn.proj.basis_scales is not None
    assert plan.trainable.blocks[0].attn.proj.random_A is None
    assert plan.trainable.blocks[0].attn.proj.random_B is None
    assert int(jnp.linalg.matrix_rank(trained_module.delta_weight())) > 1
    assert entry["seed"] == 123
    assert entry["random_A"].shape == (4, 1, 4)
    assert entry["random_B"].shape == (4, 4, 1)
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)
    assert jnp.allclose(trained(x), loaded(x), atol=1e-6)
