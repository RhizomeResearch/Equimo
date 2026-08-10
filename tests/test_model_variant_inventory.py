"""Integrity checks for every named model registry without large construction."""

from cases.model_variant_cases import MODEL_VARIANT_REGISTRIES
from equimo._pretrained import PRETRAINED_ARCHIVE_SHA256


def test_every_named_variant_has_a_factory_and_two_part_configuration():
    for case in MODEL_VARIANT_REGISTRIES:
        assert case.registry
        for variant, configuration in case.registry.items():
            assert variant == variant.lower()
            assert callable(getattr(case.module, variant, None)), (
                f"{case.family}:{variant} has no same-named factory"
            )
            assert isinstance(configuration, tuple) and len(configuration) == 2
            assert all(isinstance(part, dict) for part in configuration)
            merged = configuration[0] | configuration[1]
            assert merged, f"{case.family}:{variant} has an empty configuration"


def test_trusted_archives_map_to_named_factories_or_the_explicit_tips_text_gap():
    named_variants = {
        variant for case in MODEL_VARIANT_REGISTRIES for variant in case.registry
    }
    missing = set(PRETRAINED_ARCHIVE_SHA256) - named_variants

    assert missing
    assert all(
        identifier.startswith("tips_") and identifier.endswith("_text")
        for identifier in missing
    )
