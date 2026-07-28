"""Guard PyTree structure stability for checkpoint compatibility.

``load_weights`` validates ``model_class`` (module.qualname) and
``model_signature`` (sha256 over the ``jtu.keystr(path)`` and shape of every
array leaf).  Any refactor that renames a class, renames a field, or nests
existing fields under a new submodule silently breaks every saved checkpoint.

This test pins both values for small, fixed-hyperparameter instances of every
model (and structure-sensitive layer) so such breaks fail loudly, without
needing cached pretrained checkpoints.

To regenerate goldens after an *intentional* structure change::

    uv run python tests/test_checkpoint_signature_stability.py
"""

import json
from pathlib import Path

import jax.random as jr
import pytest

import equimo.audio.models as am
import equimo.vision.models as em
from equimo.serialization import _model_class, _model_signature
from equimo.tabular.models.tabpfn import TabPFN
from equimo.vision.layers.attention import HATBlock
from equimo.vision.layers.convolution import GenericGhostModule, GhostBottleneck
from equimo.vision.models.fastervit import FasterViT
from equimo.vision.models.mlla import Mlla
from equimo.vision.models.partialformer import PartialFormer
from equimo.vision.models.shvit import SHViT
from equimo.vision.models.vssd import Vssd

GOLDEN_PATH = Path(__file__).parent / "data" / "model_structure_signatures.json"

KEY = jr.PRNGKey(0)


def _vit(**overrides):
    cfg = dict(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=8,
        num_heads=[2],
        depths=[2],
        num_classes=10,
        key=KEY,
    )
    cfg.update(overrides)
    return em.VisionTransformer(**cfg)


def _parcae(**overrides):
    cfg = dict(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=8,
        num_heads=2,
        num_classes=10,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=1,
        n_layers_in_coda=1,
        mean_recurrence=2,
        mean_backprop_depth=1,
        max_recurrence=2,
        key=KEY,
    )
    cfg.update(overrides)
    return em.VisionParcae(**cfg)


BUILDERS = {
    "vit_tiny": _vit,
    "vit_tiny_untied_cls_norm": lambda: _vit(untie_global_and_local_cls_norm=True),
    "parcae_tiny": _parcae,
    "parcae_tiny_untied_cls_norm": lambda: _parcae(
        untie_global_and_local_cls_norm=True
    ),
    "mlla_tiny": lambda: Mlla(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=4,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 4, 8],
        num_classes=10,
        key=KEY,
    ),
    "vssd_tiny": lambda: Vssd(
        img_size=64,
        in_channels=3,
        dim=32,
        patch_size=4,
        depths=[1, 1, 2, 1],
        num_heads=[1, 2, 4, 8],
        num_classes=10,
        key=KEY,
    ),
    "convnext_tiny_2stage": lambda: em.ConvNeXt(
        in_channels=3,
        depths=[1, 1],
        dims=[32, 64],
        num_classes=10,
        key=KEY,
    ),
    "shvit_tiny": lambda: SHViT(
        in_channels=3,
        dim=[32, 64],
        pdim=[8, 16],
        qk_dim=[8, 8],
        depths=[1, 1],
        block_type=["s", "s"],
        num_classes=10,
        key=KEY,
    ),
    "partialformer_tiny": lambda: PartialFormer(
        img_size=64,
        in_channels=3,
        dim=32,
        num_heads=[1, 2],
        depths=[1, 1],
        foreground_ratios=0.5,
        patch_size=4,
        num_classes=10,
        key=KEY,
    ),
    "fastervit_tiny": lambda: FasterViT(
        img_size=64,
        in_channels=3,
        dim=32,
        in_dim=16,
        num_heads=1,
        hat=False,
        depths=[1, 1],
        window_size=4,
        ct_size=2,
        num_classes=10,
        key=KEY,
    ),
    "iformer_t": lambda: em.iformer_t(in_channels=3, num_classes=10, key=KEY),
    "lowformer_b0": lambda: em.lowformer_backbone_b0(
        in_channels=3, num_classes=10, attention_type="softmax", key=KEY
    ),
    "reduceformer_b1": lambda: em.reduceformer_backbone_b1(
        in_channels=3, num_classes=10, key=KEY
    ),
    "attnet_xxs": lambda: em.attnet_xxs(key=KEY),
    "mobilenetv3_small": lambda: em.mobilenetv3_small(key=KEY),
    "deq_convnext_t": lambda: em.deq_convnext_t(in_channels=3, num_classes=10, key=KEY),
    "ast_tiny": lambda: am.AudioSpectrogramTransformer(
        input_fdim=32,
        input_tdim=64,
        dim=32,
        patch_size=16,
        fstride=16,
        tstride=16,
        num_heads=4,
        depths=[1],
        num_classes=10,
        key=KEY,
    ),
    "tabpfn_tiny": lambda: TabPFN(
        num_classes=4,
        dim=16,
        depths=(1, 2, 2),
        num_heads=(2, 2, 2),
        num_inducing_points=4,
        feature_group_size=3,
        num_cls_tokens=2,
        num_kv_heads_test=1,
        decoder_head_dim=8,
        decoder_num_heads=2,
        mlp_ratio=2.0,
        key=KEY,
    ),
    "hat_block": lambda: HATBlock(
        dim=64, num_heads=2, window_size=4, sr_ratio=2, key=KEY
    ),
    "generic_ghost_module": lambda: GenericGhostModule(
        in_channels=8, out_channels=16, key=KEY
    ),
    "ghost_bottleneck": lambda: GhostBottleneck(
        in_channels=8, mid_channels=16, out_channels=16, key=KEY
    ),
}


def _load_goldens() -> dict:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_structure_signature_is_stable(name):
    goldens = _load_goldens()
    assert name in goldens, (
        f"No golden for {name!r}. Regenerate with "
        "`uv run python tests/test_checkpoint_signature_stability.py`."
    )
    model = BUILDERS[name]()
    expected = goldens[name]
    assert _model_class(model) == expected["model_class"], (
        f"{name}: model class changed — this breaks loading of every saved "
        "checkpoint of this model."
    )
    assert _model_signature(model) == expected["model_signature"], (
        f"{name}: PyTree structure (array-leaf paths/shapes) changed — this "
        "breaks loading of every saved checkpoint of this model."
    )


def test_goldens_have_no_stale_entries():
    stale = set(_load_goldens()) - set(BUILDERS)
    assert not stale, f"Goldens without builders: {sorted(stale)}"


if __name__ == "__main__":
    goldens = {
        name: {
            "model_class": _model_class(model),
            "model_signature": _model_signature(model),
        }
        for name, model in ((n, b()) for n, b in sorted(BUILDERS.items()))
    }
    GOLDEN_PATH.write_text(json.dumps(goldens, indent=2) + "\n")
    print(f"Wrote {len(goldens)} goldens to {GOLDEN_PATH}")
