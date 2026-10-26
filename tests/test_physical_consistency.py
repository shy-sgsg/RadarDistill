from __future__ import annotations

import numpy as np
import pytest

from radardistill.evaluation.physical_consistency import (
    select_scene_ids_by_stratum,
    wrap_periodic_difference,
)


def test_stratified_scene_selection_is_invariant_to_row_permutation() -> None:
    scene_ids = np.asarray([f"scene-{index:03d}" for index in range(60)])
    strata = np.repeat(np.asarray(["C0", "C1", "C2"]), 20)
    quotas = {"C0": 7, "C1": 5, "C2": 9}

    selected = select_scene_ids_by_stratum(scene_ids, strata, quotas, seed=101)
    permutation = np.random.default_rng(13).permutation(len(scene_ids))
    permuted = select_scene_ids_by_stratum(
        scene_ids[permutation], strata[permutation], quotas, seed=101
    )

    assert selected.keys() == permuted.keys()
    for stratum in quotas:
        assert selected[stratum].tolist() == permuted[stratum].tolist()
        assert len(selected[stratum]) == quotas[stratum]


def test_stratified_scene_selection_rejects_duplicate_ids_and_missing_quota() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_scene_ids_by_stratum(
            np.asarray(["same", "same"]), np.asarray(["C0", "C0"]), {"C0": 1}, seed=1
        )
    with pytest.raises(ValueError, match="quota"):
        select_scene_ids_by_stratum(
            np.asarray(["a", "b"]), np.asarray(["C0", "C0"]), {"C0": 3}, seed=1
        )


def test_stratified_scene_selection_rejects_unregistered_strata() -> None:
    with pytest.raises(ValueError, match="strata"):
        select_scene_ids_by_stratum(
            np.asarray(["a", "b"]), np.asarray(["C0", "C1"]), {"C0": 1}, seed=1
        )


def test_periodic_difference_wraps_to_half_open_interval() -> None:
    result = wrap_periodic_difference(np.asarray([650.0, -650.0, 1301.0]), period=1300.0)
    np.testing.assert_allclose(result, np.asarray([-650.0, -650.0, 1.0]))

