import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ("tips.py", "tips_text.py", "eupe.py")


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, ROOT / "models" / script, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("script", SCRIPTS)
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
