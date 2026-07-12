"""Generate the checked-in ``equimo.finetune`` API reference."""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import equimo.finetune as ft


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs" / "finetuning"
OUTPUT_DIR = DOCS_DIR / "api"
PUBLIC_MARKER = re.compile(r"<!-- equimo\.finetune:([A-Za-z_][A-Za-z0-9_]*) -->")
LINK = re.compile(r"(?<!!)\[[^]]*]\(([^)]+)\)")
ANCHOR = re.compile(r'<a id="([^"]+)"></a>')
ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
PRIVATE_MODULE = re.compile(r"equimo\.finetune(?:\.[a-z_][a-z0-9_]*)+\.")


GROUPS = (
    (
        "configs-plans",
        "Configs and plans",
        "Configuration records, plan metadata, and the core public typing aliases.",
    ),
    (
        "selectors-tags",
        "Selectors, paths, labels, and tags",
        "Utilities for locating, describing, labeling, and filtering PyTree leaves.",
    ),
    (
        "heads-pooling",
        "Heads, pooling, and feature extraction",
        "Task heads, pooling modules, probes, and feature-extraction helpers.",
    ),
    (
        "peft",
        "Parameter-efficient fine-tuning",
        "Public adapters, LoRA-family methods, prompts, prefixes, and related surgery.",
    ),
    (
        "recipes-profiles",
        "Recipes and profiles",
        "Fine-tuning recipes, staged workflows, and declared fidelity profiles.",
    ),
    (
        "regularization",
        "Regularization",
        "Regularization configurations, losses, and feature-tap helpers.",
    ),
    (
        "merging",
        "Model merging",
        "Model soups, task vectors, and modern model-merging methods.",
    ),
    (
        "serialization",
        "Serialization",
        "Fine-tuning bundle and delta persistence helpers.",
    ),
    (
        "integrations",
        "Modality and integration namespaces",
        "Public audio, language, tabular, and vision modality namespaces.",
    ),
    (
        "inspection-surgery",
        "Inspection and model surgery",
        "Plan inspection and general model-surgery utilities.",
    ),
)

MODULE_GROUPS = {
    "equimo.finetune.config": "configs-plans",
    "equimo.finetune.paths": "selectors-tags",
    "equimo.finetune.selectors": "selectors-tags",
    "equimo.finetune.labels": "selectors-tags",
    "equimo.finetune.masks": "selectors-tags",
    "equimo.finetune.tags": "selectors-tags",
    "equimo.finetune.heads": "heads-pooling",
    "equimo.finetune.pooling": "heads-pooling",
    "equimo.finetune.feature_extraction": "heads-pooling",
    "equimo.finetune.profiles": "recipes-profiles",
    "equimo.finetune.recipes": "recipes-profiles",
    "equimo.finetune.continued_ssl": "recipes-profiles",
    "equimo.finetune.regularization": "regularization",
    "equimo.finetune.merging": "merging",
    "equimo.finetune.serialization": "serialization",
    "equimo.finetune.audio": "integrations",
    "equimo.finetune.language": "integrations",
    "equimo.finetune.tabular": "integrations",
    "equimo.finetune.vision": "integrations",
    "equimo.finetune.inspection": "inspection-surgery",
    "equimo.finetune.surgery": "inspection-surgery",
}

# Type aliases do not retain their assignment location at runtime. The constant is
# defined in tags.py but similarly has no ``__module__`` attribute.
GROUP_OVERRIDES = {
    "CANONICAL_TAGS": "selectors-tags",
    "FilterSpec": "configs-plans",
    "LeafPredicate": "configs-plans",
    "Path": "configs-plans",
    "PEFTConfig": "peft",
    "PyTree": "configs-plans",
}
TYPE_ALIASES = frozenset(
    {"FilterSpec", "LeafPredicate", "Path", "PEFTConfig", "PyTree"}
)


def _module_name(obj: Any) -> str | None:
    if isinstance(obj, ModuleType):
        return obj.__name__
    return getattr(obj, "__module__", None)


def _group_for(name: str, obj: Any) -> str:
    if name in GROUP_OVERRIDES:
        return GROUP_OVERRIDES[name]
    module = _module_name(obj)
    if module is not None and module.startswith("equimo.finetune.peft."):
        return "peft"
    try:
        return MODULE_GROUPS[module]
    except KeyError as exc:
        raise ValueError(
            f"No API reference group for {name!r} from {module!r}"
        ) from exc


def _normalize(text: str) -> str:
    text = ADDRESS.sub("", text)
    text = text.replace("typing.", "").replace("collections.abc.", "")
    return PRIVATE_MODULE.sub("", text)


def _alias_expression(obj: Any) -> str:
    return _normalize(inspect.formatannotation(obj))


def _declaration(name: str, obj: Any) -> str:
    qualified_name = f"equimo.finetune.{name}"
    if name in TYPE_ALIASES:
        return f"type {qualified_name} = {_alias_expression(obj)}"
    if isinstance(obj, ModuleType):
        return f"module {qualified_name}"
    if inspect.isclass(obj):
        try:
            signature = _normalize(str(inspect.signature(obj)))
        except (TypeError, ValueError):
            bases = ", ".join(base.__name__ for base in obj.__bases__)
            signature = f"({bases})"
        return f"class {qualified_name}{signature}"
    if callable(obj):
        return f"{qualified_name}{_normalize(str(inspect.signature(obj)))}"
    return f"{qualified_name} = {_normalize(repr(obj))}"


def _anchor(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"equimo-finetune-{slug}"


def _docstring(name: str, obj: Any) -> str | None:
    if name in TYPE_ALIASES:
        return None
    if isinstance(obj, ModuleType) or inspect.isclass(obj) or callable(obj):
        return inspect.getdoc(obj)
    return None


def _render_docstring(docstring: str) -> str:
    return "\n".join(
        ">" if not line else f"> {line}" for line in docstring.splitlines()
    )


def _render_page(title: str, introduction: str, names: list[str]) -> str:
    lines = [
        "<!-- Generated by docs/finetuning/generate_api_reference.py. Do not edit. -->",
        "",
        f"# {title}",
        "",
        introduction,
        "",
        "[Fine-tuning overview](../index.md) | [API reference index](index.md)",
        "",
        "## Symbols",
        "",
    ]
    lines.extend(f"- [`{name}`](#{_anchor(name)})" for name in names)
    for name in names:
        obj = getattr(ft, name)
        lines.extend(
            (
                "",
                f"<!-- equimo.finetune:{name} -->",
                f'<a id="{_anchor(name)}"></a>',
                f"## `{name}`",
                "",
                "```python",
                _declaration(name, obj),
                "```",
            )
        )
        module = _module_name(obj)
        if module is not None:
            lines.extend(("", f"Defined in `{module}`."))
        docstring = _docstring(name, obj)
        if docstring:
            lines.extend(("", _render_docstring(docstring)))
    lines.append("")
    return "\n".join(lines)


def _render_index(grouped: dict[str, list[str]]) -> str:
    total = sum(len(names) for names in grouped.values())
    lines = [
        "<!-- Generated by docs/finetuning/generate_api_reference.py. Do not edit. -->",
        "",
        "# `equimo.finetune` API reference",
        "",
        f"This reference covers all {total} names exported by `equimo.finetune.__all__`.",
        "Signatures and defaults are generated from the installed source objects.",
        "See the [fine-tuning overview](../index.md) for task-oriented guides.",
        "",
    ]
    for slug, title, introduction in GROUPS:
        names = grouped[slug]
        lines.extend(
            (
                f"## [{title}]({slug}.md)",
                "",
                f"{introduction} ({len(names)} symbols)",
                "",
            )
        )
    return "\n".join(lines)


def render_reference() -> dict[Path, str]:
    names = tuple(ft.__all__)
    duplicates = sorted(name for name, count in Counter(names).items() if count != 1)
    if duplicates:
        raise ValueError(f"Duplicate names in equimo.finetune.__all__: {duplicates}")

    grouped = {slug: [] for slug, _, _ in GROUPS}
    for name in names:
        if not hasattr(ft, name):
            raise ValueError(f"Public export {name!r} does not resolve")
        grouped[_group_for(name, getattr(ft, name))].append(name)
    for group_names in grouped.values():
        group_names.sort(key=str.casefold)

    rendered = {OUTPUT_DIR / "index.md": _render_index(grouped)}
    for slug, title, introduction in GROUPS:
        rendered[OUTPUT_DIR / f"{slug}.md"] = _render_page(
            title, introduction, grouped[slug]
        )
    _validate_inventory(rendered, names)
    _validate_links(rendered)
    return rendered


def _validate_inventory(rendered: dict[Path, str], names: tuple[str, ...]) -> None:
    documented = [
        name
        for path, content in rendered.items()
        if path.name != "index.md"
        for name in PUBLIC_MARKER.findall(content)
    ]
    counts = Counter(documented)
    missing = sorted(set(names) - counts.keys())
    unexpected = sorted(counts.keys() - set(names))
    duplicated = sorted(name for name, count in counts.items() if count != 1)
    if missing or unexpected or duplicated:
        raise ValueError(
            "Invalid API inventory: "
            f"missing={missing}, unexpected={unexpected}, duplicated={duplicated}"
        )


def _validate_links(rendered: dict[Path, str]) -> None:
    documents = dict(rendered)
    overview = DOCS_DIR / "index.md"
    documents[overview] = overview.read_text()
    known_paths = set(documents)
    known_paths.update(ROOT.rglob("*.md"))

    for source, content in documents.items():
        for raw_target in LINK.findall(content):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if "://" in target or target.startswith(("mailto:", "#")):
                if target.startswith("#"):
                    _validate_anchor(source, target[1:], content)
                continue
            path_text, separator, fragment = target.partition("#")
            destination = (source.parent / path_text).resolve()
            if destination not in known_paths:
                raise ValueError(f"Broken link in {source.relative_to(ROOT)}: {target}")
            if separator:
                destination_content = documents.get(destination)
                if destination_content is None:
                    destination_content = destination.read_text()
                _validate_anchor(destination, fragment, destination_content)


def _validate_anchor(path: Path, fragment: str, content: str) -> None:
    anchors = ANCHOR.findall(content)
    if len(anchors) != len(set(anchors)):
        raise ValueError(f"Duplicate explicit anchor in {path.relative_to(ROOT)}")
    if fragment not in anchors:
        raise ValueError(f"Broken anchor in {path.relative_to(ROOT)}: #{fragment}")


def _check(rendered: dict[Path, str]) -> int:
    expected_paths = set(rendered)
    actual_paths = set(OUTPUT_DIR.glob("*.md")) if OUTPUT_DIR.exists() else set()
    stale = sorted(
        path.relative_to(ROOT)
        for path, content in rendered.items()
        if not path.exists() or path.read_text() != content
    )
    unexpected = sorted(
        path.relative_to(ROOT) for path in actual_paths - expected_paths
    )
    if stale or unexpected:
        for label, paths in (("stale or missing", stale), ("unexpected", unexpected)):
            if paths:
                print(f"{label} reference files:", file=sys.stderr)
                for path in paths:
                    print(f"  {path}", file=sys.stderr)
        print(
            "Run `uv run python docs/finetuning/generate_api_reference.py`.",
            file=sys.stderr,
        )
        return 1
    print(f"Fine-tuning API reference is current ({len(ft.__all__)} exports).")
    return 0


def _write(rendered: dict[Path, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in set(OUTPUT_DIR.glob("*.md")) - set(rendered):
        path.unlink()
    for path, content in rendered.items():
        path.write_text(content)
    print(f"Generated {len(ft.__all__)} exports across {len(GROUPS)} reference pages.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in reference differs from the live API",
    )
    args = parser.parse_args(argv)
    rendered = render_reference()
    if args.check:
        return _check(rendered)
    _write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
