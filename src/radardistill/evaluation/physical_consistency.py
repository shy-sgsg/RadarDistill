"""Stable scene sampling and periodic-error helpers for physical diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def select_scene_ids_by_stratum(
    scene_ids: np.ndarray,
    strata: np.ndarray,
    quotas: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Select a deterministic quota of stable scene IDs within every stratum.

    Candidate IDs are sorted before the seeded draw. Therefore a permutation of
    input rows does not change the selected ID set. Returned IDs are sorted as
    well, so the manifest itself is stable across row orders.
    """
    ids = np.asarray(scene_ids).reshape(-1).astype(str)
    labels = np.asarray(strata).reshape(-1).astype(str)
    if ids.size == 0 or ids.shape != labels.shape:
        raise ValueError("scene_ids and strata must be nonempty arrays with equal length")
    if np.unique(ids).size != ids.size:
        raise ValueError("scene_ids must be globally unique")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not quotas:
        raise ValueError("at least one stratum quota is required")

    normalized_quotas = {str(name): int(count) for name, count in quotas.items()}
    if any(count < 0 for count in normalized_quotas.values()):
        raise ValueError("stratum quotas must be nonnegative")
    actual_strata = set(labels.tolist())
    if actual_strata != set(normalized_quotas):
        raise ValueError(
            "registered strata must exactly match input strata; "
            f"found={sorted(actual_strata)}, quotas={sorted(normalized_quotas)}"
        )

    selected: dict[str, np.ndarray] = {}
    for ordinal, name in enumerate(sorted(normalized_quotas)):
        candidates = np.sort(ids[labels == name])
        quota = normalized_quotas[name]
        if len(candidates) < quota:
            raise ValueError(
                f"stratum {name!r} has {len(candidates)} scenes, below quota {quota}"
            )
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(ordinal,)))
        chosen = candidates if quota == len(candidates) else rng.choice(
            candidates, size=quota, replace=False
        )
        selected[name] = np.sort(chosen.astype(str, copy=False))
    return selected


def wrap_periodic_difference(difference: np.ndarray | float, *, period: float) -> np.ndarray:
    """Wrap a scalar/vector difference to the half-open interval [-period/2, period/2)."""
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be positive and finite")
    values = np.asarray(difference, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("difference must contain only finite values")
    return np.remainder(values + period / 2.0, period) - period / 2.0

