"""Frozen D2-A-R5 training-budget and repeated-scene plan helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PROTOCOL_VERSION = 1.1
COURSE_LEVELS = ("C0", "C1", "C2", "C3")


def validate_r5a_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the preregistered R5-A science/resource budget."""
    try:
        experiment = config["experiment"]
        radar = config["radar"]
        data = config["data"]
        training = config["training"]
        capacity = config["capacity"]
        metrics = config["metrics"]
        gates = config["gates"]
    except (KeyError, TypeError) as exc:
        raise ValueError("R5-A config is missing a required section") from exc

    if (
        experiment.get("id") != "D2-A-R5"
        or experiment.get("subphase") != "R5-A"
        or float(experiment.get("protocol_version", -1)) != PROTOCOL_VERSION
    ):
        raise ValueError("config must identify D2-A-R5-A protocol v1.0")
    if experiment.get("status") != "preregistered":
        raise ValueError("R5-A config must be preregistered")
    if experiment.get("final_blind_generation_authorized") is not False:
        raise ValueError("R5-A does not authorize final blind generation")
    if experiment.get("distillation_authorized") is not False:
        raise ValueError("R5-A authorizes truth-only training; distillation is forbidden")
    if (
        int(radar.get("channel_count", -1)) != 3
        or float(radar.get("prf_hz", -1.0)) != 1300.0
        or int(radar.get("pulse_count", -1)) != 130
        or int(radar.get("range_bin_count", -1)) != 32
        or radar.get("clutter") != "stationary"
    ):
        raise ValueError("R5-A radar model must preserve the frozen three-channel stationary-clutter model")

    if (
        data.get("paired_training_scenes") != 51_200
        or data.get("paired_validation_scenes") != 5_000
        or data.get("reuse_frozen_training_and_validation_ids") is not True
        or data.get("regenerate_from_frozen_source_config") is not True
        or data.get("final_blind_generation_authorized") is not False
        or data.get("development_is_final_blind") is not False
    ):
        raise ValueError("R5-A must reuse the frozen E003 train/validation scene IDs and keep blind sealed")
    if not data.get("cache_root") or not data.get("source_config"):
        raise ValueError("R5-A cache and frozen source config must be specified")

    expected = {
        "models": ["S0", "S2"],
        "seed": 101,
        "supervision": "physical_truth_only_E1",
        "optimizer": "AdamW",
        "learning_rate_initial": 0.001,
        "learning_rate_final": 0.0001,
        "extension_learning_rate_final": 0.00001,
        "learning_rate_schedule": "cosine_over_6400_updates",
        "max_updates_initial": 6400,
        "continuation_max_updates": 12800,
        "paired_scenes_per_update": 32,
        "effective_batch_size": 64,
        "validation_interval_updates": 400,
        "checkpoint_every_updates": 400,
        "precision": "FP32",
        "device": "cuda:0",
        "cuda_required": True,
        "cpu_fallback": "forbidden",
        "amp": False,
        "simultaneous_models_on_gpu": 1,
        "initial_microbatch_size": 16,
        "oom_microbatch_fallbacks": [8, 4],
        "preserve_effective_batch_on_oom": True,
        "save_optimizer_scheduler_rng_sampler_and_data_cursor": True,
    }
    for key, value in expected.items():
        if training.get(key) != value:
            if key in {"device", "cuda_required", "cpu_fallback", "simultaneous_models_on_gpu"}:
                raise ValueError(f"R5-A CUDA-only sequential training constraint changed: {key}")
            if key == "amp" or key == "precision":
                raise ValueError("R5-A precision is frozen to FP32 without AMP")
            raise ValueError(f"R5-A training setting changed: {key}")
    if int(training.get("pair_order_seed_base", -1)) < 0:
        raise ValueError("R5-A pair-order seed must be nonnegative")
    if training.get("shared_pair_permutation_by_model") is not True:
        raise ValueError("S0 and S2 must share pair order")
    if training.get("matched_shared_initialization_for_S0_S2") is not True:
        raise ValueError("S0 and S2 shared TaskStudent initialization must match")

    if capacity.get("first_comparison_only") != ["Small", "Medium"]:
        raise ValueError("R5-A first capacity comparison must be Small versus Medium")
    if capacity.get("width_by_model") != {"Small": 32, "Medium": 96}:
        raise ValueError("R5-A capacity widths are frozen to 32 and 96")
    if capacity.get("parameter_count_expected") != {"Small": 13383, "Medium": 82887}:
        raise ValueError("R5-A expected capacity parameter counts changed")
    if capacity.get("vary_width_only") is not True or capacity.get("large_model_authorized") is not False:
        raise ValueError("R5-A capacity study may vary width only; Large is not authorized")
    if metrics.get("bootstrap_unit") != "paired_base_scene_id" or int(metrics.get("bootstrap_replicates", 0)) != 2000:
        raise ValueError("R5-A paired scene bootstrap definition changed")
    if not all(name in gates for name in ("G0", "G1", "G1_overfit", "G2", "G3", "G7")):
        raise ValueError("R5-A required stage gates are missing")


def build_r5_repeated_pair_plan(
    scene_ids: Sequence[str] | np.ndarray,
    *,
    updates: int,
    pairs_per_update: int,
    seed_base: int,
) -> np.ndarray:
    """Build a matched plan with one full permutation of the fixed cohort per epoch."""
    ids = np.asarray(scene_ids).reshape(-1).astype(str)
    if ids.size == 0 or updates <= 0 or pairs_per_update <= 0 or seed_base < 0:
        raise ValueError("scene IDs, updates, batch size and seed must be valid")
    if np.unique(ids).size != ids.size:
        raise ValueError("scene IDs must be unique")
    if ids.size % pairs_per_update:
        raise ValueError("each epoch must contain complete paired-scene batches")
    updates_per_epoch = ids.size // pairs_per_update
    if updates < updates_per_epoch or updates % updates_per_epoch:
        raise ValueError("updates must comprise complete dataset passes")

    epoch_count = updates // updates_per_epoch
    plans = []
    for epoch in range(epoch_count):
        permutation = np.random.default_rng(seed_base + epoch).permutation(ids.size)
        epoch_plan = permutation.reshape(updates_per_epoch, pairs_per_update)
        if not np.array_equal(np.sort(epoch_plan.reshape(-1)), np.arange(ids.size)):
            raise RuntimeError("internal error: an R5 epoch did not consume every scene exactly once")
        plans.append(epoch_plan)
    plan = np.concatenate(plans, axis=0)
    if plan.shape != (updates, pairs_per_update):
        raise RuntimeError("internal error: R5 pair-plan shape changed")
    return plan


@dataclass(frozen=True)
class R5CosineSchedule:
    """Two-segment cosine schedule whose extension starts at the prior final LR."""

    initial_updates: int
    final_updates: int
    initial_lr: float
    initial_final_lr: float
    extension_final_lr: float

    def __post_init__(self) -> None:
        if self.initial_updates <= 1 or self.final_updates <= self.initial_updates:
            raise ValueError("R5 schedule update bounds are invalid")
        rates = (self.initial_lr, self.initial_final_lr, self.extension_final_lr)
        if not all(math.isfinite(rate) and rate > 0.0 for rate in rates):
            raise ValueError("R5 learning rates must be positive and finite")
        if not (self.initial_lr > self.initial_final_lr > self.extension_final_lr):
            raise ValueError("R5 cosine rates must decrease across both segments")

    def lr_for_update(self, update: int) -> float:
        if not 1 <= update <= self.final_updates:
            raise ValueError("update must be within the frozen R5 schedule")
        if update <= self.initial_updates:
            segment_start = 1
            segment_stop = self.initial_updates
            lr_start = self.initial_lr
            lr_stop = self.initial_final_lr
        else:
            segment_start = self.initial_updates + 1
            segment_stop = self.final_updates
            lr_start = self.initial_final_lr
            lr_stop = self.extension_final_lr
        fraction = (update - segment_start) / max(segment_stop - segment_start, 1)
        return float(lr_stop + 0.5 * (lr_start - lr_stop) * (1.0 + math.cos(math.pi * fraction)))

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "R5CosineSchedule":
        return cls(
            initial_updates=int(state["initial_updates"]),
            final_updates=int(state["final_updates"]),
            initial_lr=float(state["initial_lr"]),
            initial_final_lr=float(state["initial_final_lr"]),
            extension_final_lr=float(state["extension_final_lr"]),
        )


def should_extend_r5_training(history: Sequence[Mapping[str, Any]]) -> bool:
    """Apply the frozen loss trend and per-task protection gates for continuation."""
    required_updates = (4800, 5200, 5600, 6000, 6400)
    by_update: dict[int, dict[str, float]] = {}
    required_metrics = (
        "validation_task_loss",
        "validation_presence_loss",
        "validation_range_cell_loss",
        "validation_range_offset_loss",
        "validation_doppler_loss",
        "validation_auc",
        "validation_range_rmse_cells",
        "validation_doppler_rmse_hz",
        "validation_true_cell_doppler_rmse_hz",
        "validation_range_cell_mismatch_rate",
    )
    for row in history:
        try:
            update = int(row["update"])
            metrics = {name: float(row[name]) for name in required_metrics}
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in metrics.values()) and metrics["validation_task_loss"] > 0.0:
            by_update[update] = metrics
    if any(update not in by_update for update in required_updates):
        return False
    losses = [by_update[update]["validation_task_loss"] for update in required_updates]
    relative_drop = (losses[0] - losses[-1]) / losses[0]
    intervals_not_materially_worse = sum(
        later <= earlier * 1.005 for earlier, later in zip(losses[:-1], losses[1:], strict=True)
    )
    if relative_drop < 0.01 or intervals_not_materially_worse < 3:
        return False

    first, last = by_update[4800], by_update[6400]
    error_metrics = (
        "validation_presence_loss",
        "validation_range_cell_loss",
        "validation_range_offset_loss",
        "validation_doppler_loss",
        "validation_range_rmse_cells",
        "validation_doppler_rmse_hz",
        "validation_true_cell_doppler_rmse_hz",
    )
    if any(last[name] > first[name] * 1.05 for name in error_metrics):
        return False
    if last["validation_auc"] < first["validation_auc"] - 0.01:
        return False
    if last["validation_range_cell_mismatch_rate"] > first["validation_range_cell_mismatch_rate"] + 0.01:
        return False
    return True
