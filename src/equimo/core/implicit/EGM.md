# equimo.core.implicit

## Overview

Deep Equilibrium (DEQ) layers. `DEQCell` defines the map
`f(z, x) = stabilizer(z_in=z, z_out=strategy(blocks, z, x_ctx, injector))`. `DEQBlock` solves for the fixed point `z*`
of that cell with the third-party `banax` library: a Picard forward solver and a `NeumannPhantom`
implicit-differentiation adjoint. The cell has three pluggable parts, each with an abstract base and a string-keyed
registry:

- injectors decide how the input `x` enters the state `z`;
- stabilizers post-process the cell output so that `f` stays near-contractive;
- stack strategies decide where in the block stack injection happens.

## Interface

Re-exported by `equimo.core.implicit` ([`__init__.py`](./__init__.py)): `DEQBlock`, `DEQCell`, `get_injector`,
`get_stabilizer`, `get_strategy`. Everything else must be imported from its submodule.

- `[deq.py:DEQBlock]`: the fixed-point layer.
  - Construction: all arguments are keyword-only and required: `channels`, `depth`, `module`, `module_kwargs`,
    `injector`, `stabilizer`, `strategy`, `tol`, `max_steps`, `key`.
  - Solvers: `solver = banax.solver.Picard(atol=0.0, rtol=tol, max_steps=max_steps)` and
    `adjoint = banax.adjoint.NeumannPhantom(solver, b_solver=banax.solver.Relaxed(damp=0.5, max_steps=5))`.
  - Call: `__call__(x, z0=None, *, inference=False, key)` returns `(z_star, aux)`. `aux` has the keys `key`,
    `x_context`, `z0`, `z_star`, `trajectory`, `depth` (solver steps) and `error` (relative error).
- `[deq.py:DEQCell]`: one application of the map. It holds `blocks`, `injector`, `stabilizer` and `strategy`.
  - Construction:
    `__init__(*, channels, depth, module, module_kwargs, injector, stabilizer, strategy=EntryInjection(), key)` builds
    `depth` blocks as `module(**module_kwargs, key=k)`. `channels` is accepted but not used.
  - Precomputation: `prepare(x, key, *, inference=False)` returns `(injector.prepare(...), stabilizer.prepare(...))`.
  - Call: `__call__(z, x, *, inference=False, key)`.
- `[_base.py:AbstractInjector]`: base class for injectors.
  - `prepare(x, key)` returns an `InputContext`, which is `x` by default.
  - The abstract `__call__(z, x_ctx, *, inference=False, key)` must return an array with the same shape as `z`.
- `[_base.py:AbstractStabilizer]`: base class for stabilizers. It has `prepare(x, key)` and an abstract
  `__call__(z_in, z_out, x_ctx, *, inference=False, key)`.
- `[_base.py:AbstractStackStrategy]`: base class for stack strategies. Its abstract method is
  `__call__(blocks, z, x_ctx, injector, *, inference=False, key)`.
- `[_base.py:InputContext]`: alias of `Any`. It is the PyTree that `prepare` returns.
- `[injectors.py:get_injector]` / `[injectors.py:register_injector]`: lookup and registration for injectors. The
  registered names are:
  - `add` (`Add`)
  - `proj_add` (`ProjAdd`, a 1x1 `Conv2d` projection of `x`)
  - `prenorm_add` (`PreNormAdd`, computes `norm(z) + proj(x)`; the default norm is `LayerNorm2d`)
  - `gated` (`Gated`, a sigmoid-gated mix `(1-g)*z + g*x`)
  - `film` (`FiLM`, computes `(1+gamma)*norm(z) + beta`)
- `[stabilizers.py:get_stabilizer]` / `[stabilizers.py:register_stabilizer]`: lookup and registration for stabilizers.
  The registered names are:
  - `identity` (`Identity`)
  - `projected` (`GroupNormProject`)
  - `damped` (`Damped`, computes `(1-a)*z_in + a*z_out`)
  - `damped_projected` (`DampedProject`)
- `[strategies.py:get_strategy]` / `[strategies.py:register_strategy]`: lookup and registration for stack strategies.
  The registered names are:
  - `entry` (`EntryInjection`, injects once before the stack)
  - `per_block` (`PerBlockInjection`, injects before every block)
  - `scheduled` (`ScheduledInjection`, injects at the static `indices` in `[0, len(blocks)]`)

## Invariants

- `[deq.py:DEQBlock.__call__]`: the same `key_solve` is reused for every fixed-point iteration (through
  `eqx.Partial(self.cell, ..., key=key_solve)`). This freezes dropout and drop-path so that `f` is deterministic, which
  convergence requires. Do not split keys per iteration.
- `[deq.py:DEQBlock.__call__]`: `z0` is always initialised with `_init_z0(..., mode="mixed")`, wrapped in
  `jax.lax.stop_gradient` before the solve. `cell.prepare` runs once per forward pass, outside the solver loop.
- `[deq.py:DEQBlock]`: `tol` and `max_steps` are static fields. Gradients come from the `banax` `NeumannPhantom`
  adjoint. This package defines no `custom_vjp`.
- `[deq.py:_init_z0]`: when `z0` is given, it must match `x.shape`. A 1-D `z0` is accepted only for CHW inputs with
  `z0.shape[0] == C` and is broadcast over `h w`. These checks are `assert` statements. An unknown `mode` raises
  `ValueError` only when `z0 is None`. When `z0` is given, every mode other than `mixed` returns `z0` without checking
  the mode.
- `[deq.py:_init_z0]`: in `mixed` mode with inference enabled, it returns the base init (`z0` or zeros)
  deterministically. In training, it uses a single Bernoulli(0.5) draw to choose between that base and `N(0, 1) * 0.1`
  noise.
- `[deq.py:DEQBlock.__call__]`, `[strategies.py:EntryInjection]`: keys are split with
  `equimo.core._prng.split_for_mode`. A `key` of `None` is allowed only under static `inference=True`.
- `[deq.py:DEQCell.__call__]`: `x` must be the `(injector_ctx, stabilizer_ctx)` pair that `DEQCell.prepare` returns.
- `[_base.py:AbstractStackStrategy]`: strategies must apply the blocks as they are, without bypassing them, rewrapping
  them or re-adding residuals around them.
- `[stabilizers.py:Damped]`, `[stabilizers.py:DampedProject]`: these classes take `init_alpha` and `mode` arguments.
  - `init_alpha` must lie in `[0, 1]`, and `mode` must be `"scalar"` or `"channel"`. Both are checked with `assert`.
  - A learnable alpha is a float32 logit with shape `(1,)` or `(dim, 1, 1)`, passed through a sigmoid.
  - `fixed_alpha` and `mode` are static fields.
- `[stabilizers.py:GroupNormProject]`: the GroupNorm group count is `nearest_power_of_2_divisor(dim, num_groups)`, not
  `num_groups` itself.
- `[injectors.py:FiLM]`: `gamma_proj` is zero-initialised, so the layer starts as `norm(z) + beta`.
  `[injectors.py:Gated]` stores its gate as a float32 logit of `init_gate`, clamped to `[1e-4, 1 - 1e-4]`.
- `[strategies.py:ScheduledInjection]`: an empty `indices` raises `ValueError`. At call time, indices outside
  `[0, len(blocks)]` are dropped (the docstring says "clamped"). If none remain, the strategy falls back to entry
  injection `(0,)`.
- `[injectors.py:register_injector]`, `[stabilizers.py:register_stabilizer]`, `[strategies.py:register_strategy]`: these
  use the shared `make_register` / `make_get` from `../layers/_registry.py`. Names are lowercased, and a duplicate name
  raises `ValueError` unless `force=True`. A non-`eqx.Module` class raises `TypeError`. Unknown names raise
  `ValueError`.

## Side Effects

- `[injectors.py:_INJECTOR_REGISTRY]`, `[stabilizers.py:_STABILIZER_REGISTRY]`, `[strategies.py:_STRATEGY_REGISTRY]`:
  importing the module fills these module-global registries. The `register_*` functions mutate them at runtime, and
  `force=True` overwrites an existing entry.
- `[deq.py:_init_z0]`: draws random numbers from the supplied key in training (`mixed` and `random` modes). Its default
  `key=jr.PRNGKey(42)` is created once, at import time.
- `[deq.py:DEQCell.__init__]`: the default `strategy=EntryInjection()` is a single instance created at import and shared
  by every caller that uses the default.
- `[deq.py:DEQBlock]`: depends on the external `banax` package. There is no I/O, printing or host callback.

## Verification

- Run: `devenv shell -- uv run pytest tests/test_implicit.py -q`
