"""Behavior matrix for the private fine-tuning serialization codecs."""

from __future__ import annotations

from dataclasses import replace
import json
import tarfile
from typing import get_args

import jax.numpy as jnp
import jax.random as jr
import lz4.frame
import pytest

import equimo.finetune as eqft
from equimo.finetune._serialization_codecs import build_codec_registry
from equimo.finetune.peft import PEFTConfig
from equimo.finetune.peft.lora import architecture_hash
import equimo.finetune.serialization as serialization


METHODS = (
    "lora",
    "dora",
    "adapter",
    "prompt",
    "prefix",
    "scale_shift",
    "ia3",
    "vera",
)


def _apply_method(method, base_model):
    target = eqft.TargetSpec(tags_any=("attention.proj",))
    if method == "lora":
        return eqft.apply_lora(
            base_model,
            eqft.LoRAConfig(rank=2, alpha=4.0, target=target),
            key=jr.PRNGKey(0),
        )
    if method == "dora":
        return eqft.apply_dora(
            base_model,
            eqft.DoRAConfig(rank=2, alpha=4.0, target=target),
            key=jr.PRNGKey(0),
        )
    if method == "adapter":
        return eqft.apply_adapters(
            base_model,
            eqft.AdapterConfig(bottleneck=2),
            key=jr.PRNGKey(0),
        )
    if method == "prompt":
        return eqft.apply_prompts(
            base_model,
            eqft.PromptConfig(num_tokens=2),
            key=jr.PRNGKey(0),
        )
    if method == "prefix":
        return eqft.apply_prefixes(
            base_model,
            eqft.PrefixConfig(num_prefix_tokens=2),
            key=jr.PRNGKey(0),
        )
    if method == "scale_shift":
        return eqft.apply_scale_shift(
            base_model,
            eqft.ScaleShiftConfig(
                target=eqft.TargetSpec(include=("*.norm",)),
            ),
        )
    if method == "ia3":
        return eqft.apply_ia3(base_model, eqft.IA3Config(target=target))
    if method == "vera":
        return eqft.apply_vera(
            base_model,
            eqft.VeRAConfig(rank=2, target=target),
            key=jr.PRNGKey(0),
        )
    raise AssertionError(f"Missing test factory for {method!r}.")


def test_codec_registry_is_closed_complete_and_config_aligned():
    assert tuple(serialization._CODECS) == METHODS
    assert {
        config_type
        for codec in serialization._CODECS.values()
        for config_type in codec.config_types
    } == set(get_args(PEFTConfig))

    codec = serialization._CODECS["lora"]
    with pytest.raises(ValueError, match="Duplicate delta codec method 'lora'"):
        build_codec_registry((codec, replace(codec, extract=codec.extract)))
    with pytest.raises(TypeError):
        serialization._CODECS["other"] = codec


def test_unsupported_method_errors_remain_stable(
    tmp_path,
    tiny_vision_transformer,
):
    with pytest.raises(
        ValueError,
        match=(
            "^Unsupported delta method 'other'; currently 'lora', 'dora', "
            "'adapter', 'prompt', 'prefix', 'scale_shift', 'ia3', or 'vera'\\.$"
        ),
    ):
        eqft.save_delta(tiny_vision_transformer, tmp_path / "bad.eqft", method="other")

    bundle = eqft.FineTuneBundle(method="other", schema_version=1)
    with pytest.raises(
        eqft.FineTuneBundleError,
        match="^Unsupported delta method 'other'\\.$",
    ):
        eqft.load_delta(tiny_vision_transformer, bundle)


@pytest.mark.parametrize(
    ("method", "mergeable"),
    (
        ("lora", True),
        ("dora", True),
        ("adapter", False),
        ("prompt", False),
        ("prefix", False),
        ("scale_shift", True),
        ("ia3", True),
        ("vera", True),
    ),
)
def test_codec_schema1_roundtrip_behavior_matrix(
    method,
    mergeable,
    tmp_path,
    tiny_vision_transformer,
):
    model = _apply_method(method, tiny_vision_transformer)
    codec = serialization._CODECS[method]

    extracted = codec.extract(model)
    stripped = codec.strip_to_base(model)
    assert extracted.method == method
    assert extracted.schema_version == 1
    assert extracted.architecture_hash == architecture_hash(stripped)
    assert codec.can_infer_exact_base_checkpoint(model) is True
    assert codec.is_mergeable(extracted) is mergeable
    assert codec.is_merged(extracted) is False
    entries = extracted.adapter_config.get("entries", ())
    if entries:
        changed_entries = [dict(entry) for entry in entries]
        changed_entries[0]["merged"] = True
        changed_entries[0]["mergeable"] = False
        changed = replace(
            extracted,
            adapter_config={**extracted.adapter_config, "entries": changed_entries},
        )
        assert codec.is_merged(changed) is True
        assert codec.is_mergeable(changed) is (method == "dora")

    path = tmp_path / f"{method}.eqft"
    saved = eqft.save_delta(model, path, method=method)
    loaded_bundle = eqft.load_finetune_bundle(path)
    loaded_model = eqft.load_delta(tiny_vision_transformer, path)

    assert saved.schema_version == 1
    assert saved.base_checkpoint_id == serialization._checkpoint_hash(stripped)
    assert loaded_bundle.method == method
    assert loaded_bundle.schema_version == 1
    assert codec.extract(loaded_model).method == method
    assert jnp.allclose(
        loaded_model(jnp.ones((2, 3))),
        model(jnp.ones((2, 3))),
        atol=1e-6,
    )

    with lz4.frame.open(path, "rb") as archive:
        with tarfile.open(fileobj=archive, mode="r") as tar:
            assert set(tar.getnames()) == {"manifest.json", "arrays.eqx"}
            manifest_file = tar.extractfile("manifest.json")
            assert manifest_file is not None
            manifest = json.loads(manifest_file.read().decode())
    assert manifest["format"] == "equimo.finetune.bundle"
    assert manifest["format_version"] == 1
    assert manifest["bundle"]["method"] == method
    assert manifest["bundle"]["schema_version"] == 1


@pytest.mark.parametrize("method", METHODS)
def test_codec_malformed_entry_error_matrix(method, tiny_vision_transformer):
    model = _apply_method(method, tiny_vision_transformer)
    codec = serialization._CODECS[method]
    bundle = codec.extract(model)
    adapter_config = dict(bundle.adapter_config)
    if method == "prompt":
        del adapter_config["prompts"]
        expected_error = KeyError
        expected_message = "prompts"
    elif method == "prefix":
        del adapter_config["prefixes"]
        expected_error = KeyError
        expected_message = "prefixes"
    else:
        entries = [dict(entry) for entry in adapter_config["entries"]]
        entries[0]["path"] = "missing.path"
        adapter_config["entries"] = entries
        expected_error = eqft.FineTuneBundleError
        expected_message = "no matching leaf"
    malformed = replace(bundle, adapter_config=adapter_config)

    with pytest.raises(expected_error, match=expected_message):
        codec.load(tiny_vision_transformer, malformed)


def test_lora_codec_preserves_inexact_base_checkpoint_rule(
    tmp_path,
    tiny_vision_transformer,
):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.PiSSAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(0),
    )
    codec = serialization._CODECS["lora"]

    assert codec.can_infer_exact_base_checkpoint(model) is False
    bundle = eqft.save_delta(model, tmp_path / "pissa.eqft")
    assert bundle.base_checkpoint_id is None


FEATURE_SPECS = (
    eqft.FeatureSpec("features", "BNC", "all", None, normalize="none"),
    eqft.FeatureSpec("forward_features", "BNC", "all", "native", normalize="l2"),
    eqft.FeatureSpec("features", "BNC", "all", "cls", normalize="standardize"),
    eqft.FeatureSpec("features", "BNC", "all", "cls_patch_mean"),
    eqft.FeatureSpec("features", "BCHW", "all", "global_avg"),
    eqft.FeatureSpec(
        "features",
        "BTC",
        "all",
        "mean_token",
        mask_field="padding_mask",
        preprocessing_fingerprint="sha256:preprocessing",
    ),
    eqft.FeatureSpec(
        "features", "BNC", "patches", "mean_patch", exclude_prompt_tokens=False
    ),
    eqft.FeatureSpec("features", "BCT", "frames", "mean_frame"),
    eqft.FeatureSpec("features", "BTC", "all", "attention"),
    eqft.FeatureSpec("features", "BCHW", "all", "gem"),
    eqft.FeatureSpec("features", "BTC", "all", "last_token", mask_field="padding_mask"),
    eqft.FeatureSpec("features", "BTC", "cls", None),
    eqft.FeatureSpec(
        "features",
        "BTC",
        "last_valid",
        None,
        mask_field="padding_mask",
        layer_aggregation={"method": "last"},
    ),
    eqft.FeatureSpec(
        "intermediate_features",
        "BTC",
        "all",
        "mean_token",
        layer_aggregation={"method": "mean"},
    ),
    eqft.FeatureSpec(
        "intermediate_features",
        "BC",
        "all",
        None,
        layer_aggregation={"method": "concat"},
    ),
)


@pytest.mark.parametrize("feature_spec", FEATURE_SPECS)
def test_feature_spec_codec_roundtrips_supported_contract(feature_spec, tmp_path):
    path = tmp_path / "feature-spec.eqft"
    bundle = eqft.FineTuneBundle(method="lora", feature_spec=feature_spec)

    eqft.save_finetune_bundle(path, bundle)
    loaded = eqft.load_finetune_bundle(path)

    assert loaded.feature_spec == feature_spec


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(version=999), "codec version"),
        (lambda payload: payload["value"].update(unknown=True), "unknown"),
        (
            lambda payload: payload["value"].update(output_layout="BXYZ"),
            "output_layout",
        ),
    ),
)
def test_feature_spec_codec_rejects_unknown_versions_fields_and_values(
    mutation, message
):
    payload = serialization._feature_spec_to_payload(FEATURE_SPECS[0])
    mutation(payload)

    with pytest.raises(eqft.FineTuneBundleError, match=message):
        serialization._feature_spec_from_payload(payload)


def test_save_delta_carries_feature_spec_and_preprocessing_lineage(
    tmp_path, tiny_vision_transformer
):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    feature_spec = eqft.FeatureSpec(
        "features",
        "BNC",
        "all",
        "cls",
        preprocessing_fingerprint="sha256:preprocessing",
    )

    saved = eqft.save_delta(
        model,
        tmp_path / "with-feature-spec.eqft",
        feature_spec=feature_spec,
    )
    loaded = eqft.load_finetune_bundle(tmp_path / "with-feature-spec.eqft")

    assert saved.feature_spec == feature_spec
    assert loaded.feature_spec == feature_spec
    assert loaded.lineage.preprocessing_fingerprint == "sha256:preprocessing"


def test_feature_spec_codec_rejects_lineage_fingerprint_mismatch(tmp_path):
    feature_spec = eqft.FeatureSpec(
        "features",
        "BC",
        "all",
        None,
        preprocessing_fingerprint="sha256:features",
    )
    bundle = eqft.FineTuneBundle(
        method="lora",
        feature_spec=feature_spec,
        lineage=eqft.ModelLineage(preprocessing_fingerprint="sha256:lineage"),
    )

    with pytest.raises(eqft.FineTuneBundleError, match="bundle lineage"):
        eqft.save_finetune_bundle(tmp_path / "mismatch.eqft", bundle)
