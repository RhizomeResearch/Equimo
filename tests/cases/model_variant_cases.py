"""Named model-variant registries that are authoritative factory inventories."""

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

ast = import_module("equimo.audio.models.ast")
tabpfn = import_module("equimo.tabular.models.tabpfn")
t0 = import_module("equimo.timeseries.models.t0")
attnet = import_module("equimo.vision.models.attnet")
convnext = import_module("equimo.vision.models.convnext")
deq = import_module("equimo.vision.models.deq")
iformer = import_module("equimo.vision.models.iformer")
lowformer = import_module("equimo.vision.models.lowformer")
mobilenet = import_module("equimo.vision.models.mobilenet")
parcae = import_module("equimo.vision.models.parcae")
reduceformer = import_module("equimo.vision.models.reduceformer")
vit = import_module("equimo.vision.models.vit")


@dataclass(frozen=True)
class ModelVariantRegistryCase:
    family: str
    module: ModuleType
    registry: dict[str, tuple[dict, dict]]


MODEL_VARIANT_REGISTRIES = (
    ModelVariantRegistryCase("ast", ast, ast._AST_REGISTRY),
    ModelVariantRegistryCase("tabpfn", tabpfn, tabpfn._TABPFN_REGISTRY),
    ModelVariantRegistryCase("t0", t0, t0._T0_REGISTRY),
    ModelVariantRegistryCase("attnet", attnet, attnet._ATTNET_REGISTRY),
    ModelVariantRegistryCase("convnext", convnext, convnext._CONVNEXT_REGISTRY),
    ModelVariantRegistryCase("deq", deq, deq._DEQ_REGISTRY),
    ModelVariantRegistryCase("iformer", iformer, iformer._IFORMER_REGISTRY),
    ModelVariantRegistryCase("lowformer", lowformer, lowformer._LOWFORMER_REGISTRY),
    ModelVariantRegistryCase("mobilenetv3", mobilenet, mobilenet._MOBILENET_REGISTRY),
    ModelVariantRegistryCase("vision_parcae", parcae, parcae._VISION_PARCAE_REGISTRY),
    ModelVariantRegistryCase(
        "reduceformer", reduceformer, reduceformer._REDUCEFORMER_REGISTRY
    ),
    ModelVariantRegistryCase("vit", vit, vit._VIT_REGISTRY),
)
