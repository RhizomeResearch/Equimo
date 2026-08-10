"""Fine-tuning strategy inventories shared by integration tests."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax

import equimo.finetune as eqft
from cases.model_cases import MODEL_CASES, ModelInvocation


def build_model_invocation(name: str, key: jax.Array) -> ModelInvocation:
    return next(case for case in MODEL_CASES if case.registry_name == name).build(key)


def _identity(model: Any, key: jax.Array) -> Any:
    del key
    return model


def _scale_shift(model: Any, key: jax.Array) -> Any:
    del key
    return eqft.apply_scale_shift(
        model,
        eqft.ScaleShiftConfig(
            target=eqft.TargetSpec(tags_any=("norm",)),
            train_head=False,
        ),
    )


def _lora(model: Any, key: jax.Array) -> Any:
    return eqft.apply_lora(
        model,
        eqft.LoRAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=key,
    )


@dataclass(frozen=True)
class FineTuneModeCase:
    mode: str
    model_name: str
    spec: eqft.TrainableSpec
    adapt: Callable[[Any, jax.Array], Any] = _identity


FINETUNE_MODE_CASES = (
    FineTuneModeCase(
        "frozen", "text_transformer_encoder", eqft.TrainableSpec(mode="frozen")
    ),
    FineTuneModeCase("head", "tabpfn", eqft.TrainableSpec(mode="head")),
    FineTuneModeCase(
        "head_plus_norm", "ast", eqft.TrainableSpec(mode="head_plus_norm")
    ),
    FineTuneModeCase(
        "norm",
        "text_transformer_encoder",
        eqft.TrainableSpec(mode="norm", train_head=False),
    ),
    FineTuneModeCase(
        "bias", "convnext", eqft.TrainableSpec(mode="bias", train_head=False)
    ),
    FineTuneModeCase(
        "scale_shift",
        "vit",
        eqft.TrainableSpec(mode="scale_shift", train_head=False),
        _scale_shift,
    ),
    FineTuneModeCase(
        "partial",
        "convnext",
        eqft.TrainableSpec(mode="partial", depth_range=(0, 1), train_head=False),
    ),
    FineTuneModeCase(
        "surgical",
        "ast",
        eqft.TrainableSpec(mode="surgical", method_name="input", train_head=False),
    ),
    FineTuneModeCase("full", "t0", eqft.TrainableSpec(mode="full")),
    FineTuneModeCase(
        "peft",
        "vit",
        eqft.TrainableSpec(mode="peft", method_name="lora", train_head=False),
        _lora,
    ),
)


@dataclass(frozen=True)
class PEFTCase:
    config_type: type[Any]
    config: Any
    apply: Callable[[Any, Any, jax.Array], Any]


def _with_key(function):
    return lambda model, config, key: function(model, config, key=key)


def _without_key(function):
    return lambda model, config, key: function(model, config)


PEFT_CASES = (
    PEFTCase(eqft.LoRAConfig, eqft.LoRAConfig(rank=2), _with_key(eqft.apply_lora)),
    PEFTCase(eqft.DoRAConfig, eqft.DoRAConfig(rank=2), _with_key(eqft.apply_dora)),
    PEFTCase(
        eqft.AdapterConfig,
        eqft.AdapterConfig(bottleneck=2),
        _with_key(eqft.apply_adapters),
    ),
    PEFTCase(
        eqft.OrthogonalAdapterConfig,
        eqft.OrthogonalAdapterConfig(),
        _without_key(eqft.apply_orthogonal_adapters),
    ),
    PEFTCase(
        eqft.FourierFTConfig,
        eqft.FourierFTConfig(
            num_coefficients=4,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        _with_key(eqft.apply_fourierft),
    ),
    PEFTCase(
        eqft.PromptConfig,
        eqft.PromptConfig(num_tokens=2),
        _with_key(eqft.apply_prompts),
    ),
    PEFTCase(
        eqft.PrefixConfig,
        eqft.PrefixConfig(num_prefix_tokens=2, prefix_projection=False),
        _with_key(eqft.apply_prefixes),
    ),
    PEFTCase(
        eqft.ScaleShiftConfig,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(tags_any=("norm",))),
        _without_key(eqft.apply_scale_shift),
    ),
    PEFTCase(eqft.IA3Config, eqft.IA3Config(), _without_key(eqft.apply_ia3)),
    PEFTCase(eqft.VeRAConfig, eqft.VeRAConfig(rank=2), _with_key(eqft.apply_vera)),
)
