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
from equimo.language.models import CausalTextTransformer
from equimo.serialization import (
    _layout_for_signature,
    _model_class,
    _model_signature,
)
from equimo.tabular.models.tabpfn import TabPFN
from equimo.timeseries.models.t0 import T0
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
    "eomt_tiny": lambda: em.EoMT(
        _vit(num_classes=0, dynamic_img_size=True),
        num_classes=3,
        num_queries=2,
        num_blocks=1,
        key=KEY,
    ),
    "vit_tiny_rope_mode": lambda: _vit(
        num_heads=2,
        use_local_pos_embed=True,
        local_pos_embed_config_patch={
            "strategy": "mode",
            "freqs_for": "lang",
            "theta": 10_000,
            "pt_seq_len": 8,
        },
    ),
    "vit_tiny_rope_period": lambda: _vit(
        num_heads=2,
        use_local_pos_embed=True,
    ),
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
    "causal_text_transformer_tiny": lambda: CausalTextTransformer(
        dim=32,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        vocab_size=64,
        context_length=8,
        out_dim=16,
        key=KEY,
    ),
    "sam3_tiny": lambda: em.Sam3(
        key=KEY,
        in_channels=3,
        img_size=56,
        patch_size=14,
        embed_dim=16,
        depth=2,
        num_heads=2,
        mlp_features=32,
        window_size=2,
        global_blocks=(1,),
        dim=16,
        head_num_heads=2,
        ffn_features=32,
        geometry_depth=1,
        fusion_depth=1,
        decoder_depth=1,
        num_queries=2,
        two_way_depth=1,
        memory_depth=1,
        memory_heads=2,
        memory_start_channels=1,
        memory_fuser_depth=1,
        num_slots=2,
    ),
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
    "t0_tiny": lambda: T0(
        embed_dim=16,
        num_layers=3,
        num_heads=2,
        mlp_hidden_dim=32,
        patch_size=4,
        group_every_n=3,
        dropout=0.0,
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


# Checkpoints written before Equimo 2.8 also stored values these models now
# compute, such as RoPE frequencies; ``load_weights`` reads them through
# ``equimo._legacy_layouts``.
PRE_2_8_SIGNATURES = {
    "hat_block": (
        "sha256:46c8b66cf51e699439b3de91048aff5007efe93ba291a9744a4870ccf92f6a00"
    ),
    "mlla_tiny": (
        "sha256:1d75588db207b17d6f3f0cd080e6f8dc26cbd6f7c69185ef6721aa8af485b953"
    ),
    "vit_tiny_rope_mode": (
        "sha256:f83d08857438a53329124a9693d7dcfa40815babefc3b12b5dea62bec8599b4e"
    ),
    "vit_tiny_rope_period": (
        "sha256:f83d08857438a53329124a9693d7dcfa40815babefc3b12b5dea62bec8599b4e"
    ),
}


@pytest.mark.parametrize("name", sorted(PRE_2_8_SIGNATURES))
def test_pre_2_8_signature_is_still_accepted(name):
    model = BUILDERS[name]()
    assert _layout_for_signature(model, PRE_2_8_SIGNATURES[name]) is not model


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
