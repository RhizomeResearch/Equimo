# T0 source-provenance audit

Last reviewed: 2026-08-21

## Scope and result

This audit asks whether Equimo's T0 implementation can be treated as wholly
original MIT-licensed code or whether it remains a modified adaptation of the
Apache-2.0-licensed `tfc-t0` implementation. The comparison used the reference
dependency `tfc-t0==0.2.1`, which is pinned in Equimo's `reference` dependency
group.

The MIT-only condition is not met. Equimo expresses the upstream T0 design in
JAX/Equinox, but preserves implementation-specific structure and behavior from
the upstream source. Equimo therefore retains its existing package license
expression, `MIT AND Apache-2.0`, together with the upstream license and notice.
This is a repository provenance decision, not legal advice.

## Repository history

The relevant changes entered the repository in these revisions:

- `0627512652f5` — added the T0-alpha checkpoint converter.
- `7ec68b1e5f9d` — recorded the upstream Apache-2.0 attribution, changed the
  package expression to `MIT AND Apache-2.0`, and added `NOTICE` and the bundled
  `tfc-t0` license.
- `c7fce1ad4270` — integrated the T0-alpha forecasting implementation.

## Implementation evidence

The audit found corresponding, implementation-specific behavior rather than
only a shared high-level transformer architecture:

| Equimo area | Preserved upstream behavior |
| --- | --- |
| `src/equimo/timeseries/layers/patch_encoder.py` | Patch features concatenate values, normalized time, and validity; a residual value projection and three variate-type embeddings are applied. |
| `src/equimo/timeseries/layers/blocks.py` | Blocks alternate time and group attention, use normalized query/key projections, pre-normalized residual paths, and a gated feed-forward layer. |
| `src/equimo/timeseries/models/t0.py` scaling | Causal and global scaling, inverse hyperbolic-sine transformation, patch-edge statistics, and rescaling follow the upstream forecasting pipeline. |
| `src/equimo/timeseries/models/t0.py` forecasts | Quantile interpolation, probability-mass handling, path expansion, and autoregressive long-horizon rollout follow the upstream behavior. |
| `models/t0.py` | The converter maps the upstream checkpoint into the corresponding Equinox PyTree and validates converted arrays. |

Language, framework APIs, and module organization differ because the port is
JAX/Equinox-native. Those changes do not remove the source provenance described
above.

## Distributed notices

- [`LICENSES/tfc-t0-APACHE-2.0.txt`](../../LICENSES/tfc-t0-APACHE-2.0.txt)
  contains the Apache-2.0 text associated with the T0-derived source.
- [`NOTICE`](../../NOTICE) preserves the TFC, Datadog, and Chronos-2
  attributions.
- The separate [pretrained-model license index](../../LICENSES/pretrained/README.md)
  records the current T0-alpha weight-license link and review date.

Revisit this audit if the T0 implementation is replaced independently or if
the upstream source, provenance evidence, or applicable terms change.
