"""Offline performance audit workloads; run before and after a source change.

Example: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run benchmarks/performance.py
Model arrays are runtime arguments, and every timed result is synchronized.
"""

import argparse
import ast
import importlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from equimo.finetune import LoRALinear, TargetSpec, is_linear, resolve_target
from equimo.finetune.merging import _solve_regmean_system
from equimo.finetune.peft.base import map_wrappers
from equimo.tabular.layers.attention import InContextAttention, SoftmaxScaling
from equimo.vision.layers.attention import PartialFormerBlock


def _cast(model, dtype):
    return jax.tree.map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, model
    )


def workloads(only):
    if only in (None, "attention"):
        for rows, n_train, dim in ((128, 128, 128), (1024, 128, 128), (4096, 128, 256)):
            heads = dim // 32
            for dtype in (jnp.float32, jnp.bfloat16):
                for kv_heads in (None, 1):
                    scaling = SoftmaxScaling(heads, 32, hidden_dim=16, key=jr.key(2))
                    model = _cast(
                        InContextAttention(
                            dim,
                            heads,
                            key=jr.key(0),
                            num_kv_heads_test=kv_heads,
                            softmax_scaling=scaling,
                        ),
                        dtype,
                    )
                    x = jr.normal(jr.key(1), (rows, dim), dtype)
                    name = f"attention/{rows}/{n_train}/{dim}/kv={kv_heads}/{jnp.dtype(dtype)}"
                    yield name, lambda m, x, n: m(x, n), (model, x, n_train), True

    if only in (None, "partialformer"):
        for patches in (49, 196, 1024):
            idx = jr.permutation(jr.key(0), patches)
            x = jr.normal(jr.key(1), (patches, 4, 64))
            yield (
                f"inverse-sort/{patches}",
                lambda idx, x: x[jnp.argsort(idx)],
                (idx, x),
                True,
            )
            yield (
                f"inverse-scatter/{patches}",
                lambda idx, x: x[
                    jnp.zeros_like(idx)
                    .at[idx]
                    .set(
                        jnp.arange(idx.shape[0], dtype=idx.dtype),
                        unique_indices=True,
                    )
                ],
                (idx, x),
                True,
            )
        for side in (14, 28):
            model = PartialFormerBlock(32, 2, 0.5, 2, key=jr.key(0))
            x = jr.normal(jr.key(1), (side * side, 32))
            qa = jr.normal(jr.key(2), (1, 32))
            yield (
                f"partialformer/{side}",
                lambda m, x, qa: m(x, qa, inference=True),
                (model, x, qa),
                True,
            )

    if only in (None, "regmean"):
        for width in (64, 256):
            a = jr.normal(jr.key(0), (width, width))
            system = a @ a.T / width + 0.1 * jnp.eye(width)
            rhs = jr.normal(jr.key(1), (width // 2, width))
            yield (
                f"regmean/{width}",
                lambda s, w: _solve_regmean_system(s, w, solver="cholesky"),
                (system, rhs),
                True,
            )

    if only in (None, "selectors"):
        linear = eqx.nn.Linear(4, 4, key=jr.key(0))
        for count in (64, 256, 1024):
            model = tuple(linear for _ in range(count))
            yield (
                f"selectors/{count}",
                lambda m: resolve_target(m, TargetSpec(predicate=is_linear)),
                (model,),
                False,
            )

    if only in (None, "wrappers"):
        wrapper = LoRALinear(
            eqx.nn.Linear(4, 4, key=jr.key(0)),
            rank=2,
            alpha=4.0,
            scaling="alpha_over_r",
            dropout=0.0,
            train_base=False,
            mergeable=True,
            key=jr.key(1),
        )
        wrapper = eqx.tree_at(
            lambda m: m.lora_B, wrapper, jnp.ones_like(wrapper.lora_B)
        )
        for count in (16, 64, 256):
            model = tuple(wrapper for _ in range(count))
            yield (
                f"wrappers/{count}",
                lambda m: map_wrappers(m, LoRALinear, lambda w: w.merge()),
                (model,),
                False,
            )


def baseline_functions(source_root):
    """Load only the audited callables, avoiding duplicate model registration.

    The source snapshot must use the same module layouts and dependencies as the
    installed package. It is executed as Python code and must be trusted.
    """
    specs = (
        (
            "attention",
            "equimo.tabular.layers.attention",
            "InContextAttention",
            ("__call__",),
        ),
        (
            "partialformer",
            "equimo.vision.layers.attention",
            "PartialFormerBlock",
            ("__call__",),
        ),
        ("regmean", "equimo.finetune.merging", None, ("_solve_regmean_system",)),
        (
            "selectors",
            "equimo.finetune.selectors",
            None,
            ("_resolve_predicate_paths", "resolve_target"),
        ),
        ("wrappers", "equimo.finetune.peft.base", None, ("map_wrappers",)),
    )
    functions = {}
    for label, module_name, class_name, names in specs:
        path = source_root.joinpath(*module_name.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = [
            node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        body = tree.body
        if class_name is not None:
            body = next(
                node.body
                for node in body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
        selected = [
            node
            for node in body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        if len(selected) != len(names):
            raise ValueError(f"Baseline {path} does not contain {names}")
        namespace = vars(importlib.import_module(module_name)).copy()
        exec(
            compile(
                ast.Module(body=imports + selected, type_ignores=[]), str(path), "exec"
            ),
            namespace,
        )
        functions[label] = namespace[names[-1]]
    return {
        "attention": functions["attention"],
        "partialformer": lambda m, x, qa: functions["partialformer"](
            m, x, qa, inference=True
        ),
        "regmean": lambda s, w: functions["regmean"](s, w, solver="cholesky"),
        "selectors": lambda m: functions["selectors"](
            m, TargetSpec(predicate=is_linear)
        ),
        "wrappers": lambda m: functions["wrappers"](m, LoRALinear, lambda w: w.merge()),
    }


def prepare(fn, args, compiled):
    compilation_ms = None
    temporary_bytes = None
    flops = None
    if compiled:
        start = time.perf_counter()
        fn = eqx.filter_jit(fn).lower(*args).compile()
        compilation_ms = (time.perf_counter() - start) * 1000
        memory = fn.compiled.memory_analysis()
        if memory is not None:
            temporary_bytes = memory.temp_size_in_bytes
        costs = fn.compiled.cost_analysis()
        if costs is not None:
            flops = costs.get("flops")
    return fn, {
        "compile_ms": compilation_ms,
        "temporary_bytes": temporary_bytes,
        "flops": flops,
    }


def measure(fn, args, *, baseline, compiled, repeats, rounds):
    jax.block_until_ready(args)
    functions = (
        {"baseline": baseline, "candidate": fn} if baseline else {"candidate": fn}
    )
    prepared = {name: prepare(call, args, compiled) for name, call in functions.items()}
    functions = {name: call for name, (call, _) in prepared.items()}
    results = {name: stats for name, (_, stats) in prepared.items()}
    if not compiled:
        # Whole-tree baseline replacements are deliberately quadratic and slow.
        repeats = min(repeats, 3)
    if baseline:
        before, after = (call(*args) for call in functions.values())
        low_precision = any(
            getattr(x, "dtype", None) == jnp.bfloat16 for x in jax.tree.leaves(after)
        )
        tolerance = 0.02 if low_precision else 1e-5
        assert eqx.tree_equal(before, after, atol=tolerance, rtol=tolerance), (
            "Baseline output mismatch"
        )
    for _ in range(3):
        for call in functions.values():
            jax.block_until_ready(call(*args))
    medians = {name: [] for name in functions}
    for round_index in range(rounds):
        timings = {name: [] for name in functions}
        for repeat_index in range(repeats):
            order = list(functions)
            if (round_index + repeat_index) % 2:
                order.reverse()
            for name in order:
                start = time.perf_counter()
                jax.block_until_ready(functions[name](*args))
                timings[name].append((time.perf_counter() - start) * 1000)
        for name in functions:
            medians[name].append(statistics.median(timings[name]))
    for name, result in results.items():
        result.update(
            median_ms=statistics.median(medians[name]),
            round_medians_ms=medians[name],
            repeats=repeats,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="Trusted baseline source directory containing equimo/",
    )
    parser.add_argument(
        "--only",
        choices=("attention", "partialformer", "regmean", "selectors", "wrappers"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 1:
        parser.error("repeats and rounds must be positive")
    baseline = baseline_functions(args.baseline_source) if args.baseline_source else {}
    report = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "equinox": eqx.__version__,
            "platform": platform.platform(),
            "devices": [str(d) + ": " + d.device_kind for d in jax.devices()],
            "x64": jax.config.jax_enable_x64,
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "thread_settings": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "XLA_FLAGS")
            },
        },
        "results": {},
    }
    for name, fn, inputs, compiled in workloads(args.only):
        result = measure(
            fn,
            inputs,
            baseline=baseline.get(name.split("/")[0]),
            compiled=compiled,
            repeats=args.repeats,
            rounds=args.rounds,
        )
        report["results"][name] = result
        summary = ", ".join(
            f"{label} {stats['median_ms']:.3f} ms" for label, stats in result.items()
        )
        print(f"{name}: {summary}", flush=True)
        jax.clear_caches()
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
