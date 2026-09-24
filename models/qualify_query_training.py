"""Generate a small task-loss fixture from digest-verified official sources.

Supply a directory containing the four source files listed in SOURCES. This
maintainer-only command executes selected, unmodified upstream functions and
needs the optional PyTorch reference environment. It never imports Equimo.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

if __name__ == "__main__" and sys.path and sys.path[0].endswith("/models"):
    sys.path.pop(0)

import numpy as np


SOURCES = {
    "criterion.py": {
        "url": "https://raw.githubusercontent.com/facebookresearch/Mask2Former/161be514814ea63440fae520d2ec72707b62ba54/mask2former/modeling/criterion.py",
        "sha256": "b80c583c1dcfc63f9fbdaea3d0509a7908d161fc348aefdbf6746ffff53a88da",
    },
    "matcher.py": {
        "url": "https://raw.githubusercontent.com/facebookresearch/Mask2Former/161be514814ea63440fae520d2ec72707b62ba54/mask2former/modeling/matcher.py",
        "sha256": "3e291b1a04aec536a32c277668256749d99e5bfc66b89ef0cd8cd3942046d905",
    },
    "point_features.py": {
        "url": "https://raw.githubusercontent.com/facebookresearch/detectron2/9f8d35e653674932fbe7a579feee36ed9dc8e8c5/projects/PointRend/point_rend/point_features.py",
        "sha256": "81f16bc695110dd45575bb64d8b1d84efbed4d13ab1acb44d2b0d00e88f2daae",
    },
    "_hungarian_algorithm.py": {
        "url": "https://raw.githubusercontent.com/google-deepmind/optax/225a7079f4630bf75bee94ec78de9aa69c60fab3/optax/assignment/_hungarian_algorithm.py",
        "sha256": "996656415890fa8017a3ddf1c6b95cacab090b13913563a73ceda218ef89adbe",
    },
}


def _functions(directory: Path, filename: str, names: tuple[str, ...], namespace):
    source = (directory / filename).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCES[filename]["sha256"]:
        raise ValueError(f"Unexpected upstream source digest: {filename}")
    # Load only the named functions, without optional framework imports and
    # unrelated models. Their bodies, defaults, and annotations are unchanged.
    selected = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing reference functions in {filename}")
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace
    )


def generate(reference_dir: Path, output: Path):
    import jax
    import jax.numpy as jnp
    import scipy
    from scipy.optimize import linear_sum_assignment
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    namespace = {
        "torch": torch,
        "F": functional,
        "jax": jax,
        "lax": jax.lax,
        "jnp": jnp,
    }
    _functions(
        reference_dir, "criterion.py", ("dice_loss", "sigmoid_ce_loss"), namespace
    )
    _functions(
        reference_dir,
        "matcher.py",
        ("batch_dice_loss", "batch_sigmoid_ce_loss"),
        namespace,
    )
    _functions(reference_dir, "point_features.py", ("point_sample",), namespace)
    _functions(
        reference_dir,
        "_hungarian_algorithm.py",
        ("hungarian_algorithm", "_masked_argmin"),
        namespace,
    )
    rng = np.random.default_rng(23)
    classes = torch.tensor(
        rng.normal(size=(3, 3)), dtype=torch.float32, requires_grad=True
    )
    masks = torch.tensor(
        rng.normal(size=(3, 3, 4)), dtype=torch.float32, requires_grad=True
    )
    targets = torch.tensor(rng.integers(0, 2, size=(2, 4, 6)), dtype=torch.float32)
    labels = torch.tensor([0, 1])
    # Interior coordinates isolate loss/interpolation parity from our documented
    # border-extension and validity-normalization changes.
    points = torch.tensor(rng.uniform(0.3, 0.7, size=(7, 2)), dtype=torch.float32)
    sampled = namespace["point_sample"](
        masks[:, None], points.expand(3, -1, -1), align_corners=False
    )[:, 0]
    truth = namespace["point_sample"](
        targets[:, None], points.expand(2, -1, -1), align_corners=False
    )[:, 0]
    class_cost = -classes.softmax(-1)[:, labels]
    mask_cost = namespace["batch_sigmoid_ce_loss"](sampled, truth)
    dice_cost = namespace["batch_dice_loss"](sampled, truth)
    costs = 2 * class_cost + 5 * mask_cost + 5 * dice_cost
    rows, cols = linear_sum_assignment(costs.detach().numpy())
    assigned = torch.full((3,), 2, dtype=torch.long)
    assigned[rows] = labels[cols]
    class_loss = functional.cross_entropy(
        classes, assigned, weight=torch.tensor([1.0, 1.0, 0.1])
    )
    mask_loss = namespace["sigmoid_ce_loss"](sampled[rows], truth[cols], 2.0)
    dice_loss = namespace["dice_loss"](sampled[rows], truth[cols], 2.0)
    total = 2 * class_loss + 5 * mask_loss + 5 * dice_loss
    total.backward()
    assignment_costs = rng.normal(size=(5, 3)).astype(np.float32)
    i, j = namespace["hungarian_algorithm"](jnp.asarray(assignment_costs))
    values = {
        "class_logits": classes,
        "mask_logits": masks,
        "target_masks": targets,
        "points": points,
        "class_cost": class_cost,
        "mask_cost": mask_cost,
        "dice_cost": dice_cost,
        "class_loss": class_loss,
        "mask_loss": mask_loss,
        "dice_loss": dice_loss,
        "total": total,
        "class_gradient": classes.grad,
        "mask_gradient": masks.grad,
    }
    fixture = {name: value.detach().numpy().tolist() for name, value in values.items()}
    fixture.update(
        {
            "query_indices": rows[np.argsort(cols)].tolist(),
            "assignment_costs": assignment_costs.tolist(),
            "optax_query_indices": np.asarray(i)[np.argsort(np.asarray(j))].tolist(),
            "sources": SOURCES,
            "versions": {
                "torch": torch.__version__,
                "jax": jax.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "seed": 23,
            "generator": "uv run --group reference python models/qualify_query_training.py --reference-dir /path/to/pinned/sources --output tests/data/query_training_reference.json",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    generate(arguments.reference_dir, arguments.output)
