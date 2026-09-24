# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
# Copyright 2026 Equimo contributors.
# SPDX-License-Identifier: Apache-2.0
"""Padded rectangular assignment, adapted from Optax's Hungarian algorithm.

Reference: google-deepmind/optax, revision
225a7079f4630bf75bee94ec78de9aa69c60fab3,
optax/assignment/_hungarian_algorithm.py. See NOTICE for modifications.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax


class QueryAssignment(eqx.Module):
    """Fixed-capacity matches in target order.

    ``query_indices[t]`` is the query matched to target slot ``t``. Invalid
    slots have query index ``-1``. ``is_valid`` is false for nonfinite
    active costs or internal arithmetic overflow; no matches are returned in
    that case. Indices and decisions
    are nondifferentiable. Ties use the lowest available search index, keeping
    an existing predecessor on equal costs; this is not a lexicographic solver.
    """

    query_indices: jax.Array
    valid: jax.Array
    is_valid: jax.Array


def match_queries(costs: jax.Array, target_valid: jax.Array) -> QueryAssignment:
    """Minimize ``(Q,T)`` costs with ``T <= Q``, entirely on the JAX device.

    Every valid target is assigned a distinct query. Invalid target columns
    are ignored, including their contents. Complexity is O(T**2 * Q), with
    O(T * Q) cost storage. No host solver, callback, or differentiation through
    assignment is used. Nonfinite active costs or internal arithmetic overflow
    produce ``is_valid=False``. Every augmenting-path loop is explicitly bounded.
    """
    if costs.ndim != 2 or target_valid.shape != (costs.shape[1],):
        raise ValueError("Expected (queries, targets) costs and (targets,) validity.")
    queries, targets = costs.shape
    if queries == 0 or targets > queries:
        raise ValueError("Assignment requires 0 <= target capacity <= query count.")
    if target_valid.dtype != jnp.bool_:
        raise TypeError("Target validity must be boolean.")
    matrix = lax.stop_gradient(costs.astype(jnp.float32).T)
    finite = jnp.all(jnp.where(target_valid[:, None], jnp.isfinite(matrix), True))
    active = target_valid & finite
    matrix = jnp.where(active[:, None], matrix, 0)
    if targets == 0:
        return QueryAssignment(jnp.zeros((0,), dtype=jnp.int32), active, finite)

    def insert(row, state):
        def augment(state):
            row_potential, column_potential, owner, feasible = state
            owner = owner.at[0].set(row + 1)

            def search(carry):
                u, v, visited, distance, previous, column, steps = carry
                visited = visited.at[column].set(True)
                available = ~visited[1:]
                reduced = matrix[owner[column] - 1] - u[owner[column]] - v[1:]
                improves = available & (reduced < distance)
                previous = jnp.where(improves, column, previous)
                distance = jnp.where(improves, reduced, distance)
                candidate = jnp.where(available, distance, jnp.inf)
                next_column = jnp.argmin(candidate).astype(jnp.int32) + 1
                delta = candidate[next_column - 1]
                # Out-of-range scatter updates are dropped; unvisited columns
                # must not repeatedly add to the dummy row's potential.
                indices = jnp.where(visited, owner, targets + 1)
                u = u.at[indices].add(delta, mode="drop")
                v = jnp.where(visited, v - delta, v)
                distance = jnp.where(available, distance - delta, distance)
                return u, v, visited, distance, previous, next_column, steps + 1

            def searching(carry):
                *_, column, steps = carry
                return (owner[column] != 0) & (steps <= row)

            initial = (
                row_potential,
                column_potential,
                jnp.zeros((queries + 1,), dtype=jnp.bool_),
                jnp.full((queries,), jnp.inf, dtype=jnp.float32),
                jnp.zeros((queries,), dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )
            u, v, _, _, previous, column, _ = lax.while_loop(searching, search, initial)
            feasible &= (
                (owner[column] == 0)
                & jnp.all(jnp.isfinite(u))
                & jnp.all(jnp.isfinite(v))
            )

            def backtrack(carry):
                owner, column, steps = carry
                predecessor = previous[column - 1]
                return owner.at[column].set(owner[predecessor]), predecessor, steps + 1

            owner, column, _ = lax.while_loop(
                lambda carry: feasible & (carry[1] != 0) & (carry[2] <= row),
                backtrack,
                (owner, column, jnp.asarray(0, dtype=jnp.int32)),
            )
            return u, v, owner, feasible & (column == 0)

        return lax.cond(active[row] & state[-1], augment, lambda value: value, state)

    initial = (
        jnp.zeros((targets + 1,), dtype=jnp.float32),
        jnp.zeros((queries + 1,), dtype=jnp.float32),
        jnp.zeros((queries + 1,), dtype=jnp.int32),
        finite,
    )
    _, _, owner, feasible = lax.fori_loop(0, targets, insert, initial)
    indices = jnp.full((targets,), -1, dtype=jnp.int32)
    destinations = jnp.where(owner[1:] > 0, owner[1:] - 1, targets)
    indices = indices.at[destinations].set(
        jnp.arange(queries, dtype=jnp.int32), mode="drop"
    )
    return QueryAssignment(
        jnp.where(feasible, indices, -1), active & feasible, feasible
    )
