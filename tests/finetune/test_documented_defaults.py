"""Regression tests for behavior-affecting documented defaults."""

from __future__ import annotations

from pathlib import Path

import equimo.finetune as eqft


DEFAULTS_DOC = Path(__file__).parents[2] / "docs" / "finetuning" / "method_defaults.md"


def _method_row(method: str) -> str:
    prefix = f"| {method} |"
    rows = [
        line
        for line in DEFAULTS_DOC.read_text().splitlines()
        if line.startswith(prefix)
    ]
    assert len(rows) == 1
    return rows[0]


def test_prompt_config_defaults_are_documented():
    config = eqft.PromptConfig()

    assert config.num_tokens == 10
    assert config.depth == "shallow"
    assert config.init_std == 0.02

    row = _method_row("Prompt tuning (`PromptConfig`)")
    assert f"{config.num_tokens} tokens" in row
    assert config.depth in row
    assert f"std {config.init_std:g}" in row


def test_l2_sp_config_defaults_are_documented():
    config = eqft.L2SPConfig()

    assert config.alpha == 1e-3
    assert config.beta == 0.0
    assert config.reduction == "sum"
    assert config.library_variant == "paper_objective"

    row = _method_row("L2-SP (`L2SPConfig`)")
    assert f"alpha {config.alpha:.0e}".replace("e-0", "e-") in row
    assert f"beta {config.beta:g}" in row
    assert f"{config.reduction} reduction" in row
    assert f"half-scaled {config.library_variant.replace('_', ' ')}" in row
