"""Fine-tuning scaffolding tests."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

import equimo.finetune as eqft

from fixtures import (
    EXPECTED_PARAM_COUNTS,
    TinyASTLikeEncoder,
    TinyConvNeXtLike,
    TinyLinearMLP,
    TinyTextEncoder,
    TinyVisionTransformer,
    count_params,
    extract_paths,
)


def test_import_equimo_finetune():
    assert eqft.TargetSpec(tags_any=("attention.qkv",)).tags_any == ("attention.qkv",)
    assert eqft.TrainableSpec(mode="head").mode == "head"
    assert eqft.LLRDConfig().decay == 0.75


def test_required_public_api_exports():
    required = {
        "AdapterBankConfig",
        "AdapterFusion",
        "AdapterFusionConfig",
        "BitFitConfig",
        "ContinuedSSLAdaptationConfig",
        "ContinuedSSLPlan",
        "PTuningV2Config",
        "DenseFeatureAdapter",
        "EWCConfig",
        "FineTuneStage",
        "FineTuneBundleError",
        "GlobalAveragePool",
        "GreedySoupConfig",
        "HeadPlusNormConfig",
        "LoRAPlusLabelConfig",
        "MixoutConfig",
        "PEFTConfig",
        "OutputAdapterModule",
        "PartialUnfreezeConfig",
        "ParallelAdapterConfig",
        "PrefixProjection",
        "StagePolicy",
        "SoftPromptConfig",
        "SupervisedAfterSSLConfig",
        "adapter_fusion_trainable_spec",
        "adapter_norm_loss",
        "apply_adapter_fusion",
        "audio",
        "bitfit_trainable_spec",
        "configure_adapter_bank",
        "continued_ssl_adaptation",
        "delta_attention_aux_loss_spec",
        "delta_attention_loss",
        "disable_dropout",
        "disable_stochastic_depth",
        "ewc_loss",
        "iter_ia3_modules",
        "iter_scale_shift_wrappers",
        "language",
        "lora_transformer",
        "lora_transformer_all_linear",
        "merge_dora",
        "merge_ia3",
        "merge_scale_shift",
        "merge_vera",
        "add_adapter",
        "set_dropout_rate",
        "set_stochastic_depth_rate",
        "set_active_adapter",
        "save_finetune_bundle",
        "load_finetune_bundle",
        "mixout_leaf",
        "mixout_tree",
        "partial_unfreeze",
        "prepare_lpft_stage1_model",
        "prepare_lpft_stage2_model",
        "recipes",
        "task_adapter_bank",
        "transfer_head",
        "task_vector_norm_loss",
        "vision",
        "vpt_deep",
    }

    assert not {name for name in required if not hasattr(eqft, name)}


def test_recipe_namespaces_export_direct_helpers():
    assert eqft.recipes.lora_transformer is eqft.lora_transformer
    assert hasattr(eqft.recipes, "lora_transformer_all_linear")
    assert hasattr(eqft.recipes, "vpt_deep")
    assert hasattr(eqft.recipes, "task_adapter_bank")
    assert hasattr(eqft.vision, "dense_feature_adapter")
    assert hasattr(eqft.vision, "DenseProbe")
    assert hasattr(eqft.vision, "make_dense_probe")
    assert hasattr(eqft.audio, "adapter_ast")
    assert hasattr(eqft.language, "prefix_encoder")


def test_tiny_fixture_param_counts(finetune_key):
    models = {
        "tiny_ast_like_encoder": TinyASTLikeEncoder(key=finetune_key),
        "tiny_convnext_like": TinyConvNeXtLike(key=finetune_key),
        "tiny_linear_mlp": TinyLinearMLP(key=finetune_key),
        "tiny_text_encoder": TinyTextEncoder(key=finetune_key),
        "tiny_vision_transformer": TinyVisionTransformer(key=finetune_key),
    }

    assert {
        name: count_params(model) for name, model in models.items()
    } == EXPECTED_PARAM_COUNTS


def test_tiny_fixture_paths_are_predictable(tiny_vision_transformer, tiny_text_encoder):
    vit_paths = extract_paths(tiny_vision_transformer)
    text_paths = extract_paths(tiny_text_encoder)

    assert "patch_embed.proj.weight" in vit_paths
    assert "blocks.0.attn.qkv.weight" in vit_paths
    assert "blocks.1.mlp.fc2.bias" in vit_paths
    assert "head.bias" in vit_paths
    assert "token_embed.weight" in text_paths
    assert "blocks.0.attn.proj.weight" in text_paths


class _TiedLeaves(eqx.Module):
    left: jnp.ndarray
    right: jnp.ndarray


def test_param_identities_mark_tied_alias_groups():
    shared = jnp.ones((2,), dtype=jnp.float32)
    model = _TiedLeaves(shared, shared)

    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
    )

    assert plan.identities.left.alias_group is not None
    assert plan.identities.left.alias_group == plan.identities.right.alias_group


def test_finetune_core_does_not_import_optimizer_libraries():
    finetune_root = Path(eqft.__file__).parent
    source = "\n".join(path.read_text() for path in finetune_root.rglob("*.py"))

    assert "import optax" not in source
    assert "import rollfast" not in source
