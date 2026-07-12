import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ("tips.py", "tips_text.py", "eupe.py")
HELP_SCRIPTS = (
    "audio_preprocessing.py",
    "ast.py",
    "dinov2.py",
    "dinov3.py",
    "eupe.py",
    "siglip2.py",
    "tabpfn3.py",
    "tips.py",
    "tips_text.py",
    "torch_models.py",
    "validate_references.py",
)


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, ROOT / "models" / script, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("script", HELP_SCRIPTS)
def test_conversion_help_is_offline(script):
    result = run_script(script, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_conversion_requires_explicit_inputs(script):
    result = run_script(script)

    assert result.returncode == 2
    assert "required" in result.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_conversion_rejects_unknown_variant(script, tmp_path):
    source_dir = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    source_dir.mkdir()
    checkpoint.touch()
    result = run_script(
        script,
        "unknown",
        "--source-dir",
        str(source_dir),
        "--checkpoint",
        str(checkpoint),
        "--dry-run",
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_tips_image_dry_run_resolves_checkpoint_root(tmp_path):
    source_dir = tmp_path / "tips-source"
    checkpoint_root = tmp_path / "checkpoints"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    checkpoint_root.mkdir()
    checkpoint = checkpoint_root / "tips_oss_s14_highres_distilled_vision.npz"
    checkpoint.touch()

    result = run_script(
        "tips.py",
        "tips_vits14_hr",
        "--source-dir",
        str(source_dir),
        "--checkpoint-root",
        str(checkpoint_root),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert f"checkpoint={checkpoint.resolve()}" in result.stdout
    assert f"output={(output_dir / 'tips_vits14_hr').resolve()}" in result.stdout


def test_tips_dry_run_defaults_to_all_variants(tmp_path):
    source_dir = tmp_path / "tips-source"
    checkpoint_root = tmp_path / "checkpoints"
    source_dir.mkdir()
    checkpoint_root.mkdir()
    for filename in (
        "tips_oss_s14_highres_distilled_vision.npz",
        "tips_oss_b14_highres_distilled_vision.npz",
        "tips_oss_l14_highres_distilled_vision.npz",
        "tips_oss_so400m14_highres_largetext_distilled_vision.npz",
        "tips_oss_g14_lowres_vision.npz",
        "tips_oss_g14_highres_vision.npz",
    ):
        (checkpoint_root / filename).touch()

    result = run_script(
        "tips.py",
        "--source-dir",
        str(source_dir),
        "--checkpoint-root",
        str(checkpoint_root),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("checkpoint=") == 6


def test_tips_text_dry_run_accepts_single_checkpoint(tmp_path):
    source_dir = tmp_path / "tips-source"
    checkpoint = tmp_path / "text.npz"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    checkpoint.touch()

    result = run_script(
        "tips_text.py",
        "tips_vitb14_hr_text",
        "--source-dir",
        str(source_dir),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert f"checkpoint={checkpoint.resolve()}" in result.stdout
    assert f"output={(output_dir / 'tips_vitb14_hr_text').resolve()}" in result.stdout


def test_tips_single_checkpoint_requires_one_variant(tmp_path):
    source_dir = tmp_path / "tips-source"
    checkpoint = tmp_path / "vision.npz"
    source_dir.mkdir()
    checkpoint.touch()

    result = run_script(
        "tips.py",
        "tips_vits14_hr",
        "tips_vitb14_hr",
        "--source-dir",
        str(source_dir),
        "--checkpoint",
        str(checkpoint),
        "--dry-run",
    )

    assert result.returncode == 2
    assert "exactly one variant" in result.stderr


def test_eupe_dry_run_resolves_explicit_paths(tmp_path):
    source_dir = tmp_path / "eupe-source"
    checkpoint = tmp_path / "EUPE-ViT-T.pt"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    checkpoint.touch()

    result = run_script(
        "eupe.py",
        "eupe_vitt16",
        "--source-dir",
        str(source_dir),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert f"source={source_dir.resolve()}" in result.stdout
    assert f"checkpoint={checkpoint.resolve()}" in result.stdout
    assert f"output={(output_dir / 'eupe_vitt16').resolve()}" in result.stdout


def test_conversion_scripts_have_no_machine_specific_paths():
    contents = "\n".join((ROOT / "models" / script).read_text() for script in SCRIPTS)

    assert "/mnt/hdd" not in contents
    assert "/home/" not in contents


def test_torch_reference_dry_run_uses_pinned_revision(tmp_path):
    result = run_script(
        "torch_models.py",
        "dinov3_vits16",
        "--output-dir",
        str(tmp_path),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "revision=114c1379950215c8b35dfcd4e90a5c251dde0d32" in result.stdout
    assert "seed=42" in result.stdout
    assert str(tmp_path / "dinov3_vits16_reference.npz") in result.stdout


def test_ast_dry_run_uses_output_dir_and_revision(tmp_path):
    result = run_script(
        "ast.py",
        "ast_base_patch16_speechcommands_v2_10_10_0_9812",
        "--references-only",
        "--output-dir",
        str(tmp_path),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "revision=315b0b847a3ca207e68b718503ad72066612eacd" in result.stdout
    assert (
        str(tmp_path / "ast_base_patch16_speechcommands_v2_10_10_0_9812_reference.npz")
        in result.stdout
    )


def test_tabpfn_dry_run_resolves_checkpoint(tmp_path):
    checkpoint = tmp_path / "tabpfn-v3-classifier-v3_default.ckpt"
    checkpoint.touch()
    result = run_script(
        "tabpfn3.py",
        "tabpfn_v3_classifier_default",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(tmp_path),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert f"checkpoint={checkpoint.resolve()}" in result.stdout
    assert "upstream_revision=e923ba9be85784206c9e2f43b0035c84d5fd5747" in result.stdout
