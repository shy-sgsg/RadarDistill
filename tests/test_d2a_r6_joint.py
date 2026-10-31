from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from radardistill.distillation.task_losses import encode_doppler_hz
from radardistill.students.r6_joint import (
    R6D_ARMS,
    R6J1Student,
    R6J2Student,
    build_r6d_student,
    presence_logit_from_candidate_logits,
    pool_r6d_j2_parameters,
    wrap_r6d_doppler,
)
from radardistill.training.d2a_r6d_joint import (
    audit_r6d_scene_plans,
    build_r6d_paired_plan,
    r6d_learning_rate,
    r6d_loss,
)


PRF = 1300.0


def _truth(batch: int, *, all_h0: bool = False) -> dict[str, torch.Tensor]:
    presence = torch.zeros(batch) if all_h0 else torch.ones(batch)
    return {
        "truth_presence": presence,
        "truth_range_cell": torch.full((batch,), 11.25),
        "truth_doppler_hz": torch.full((batch,), 649.9),
    }


@pytest.mark.parametrize("arm", R6D_ARMS)
def test_r6d_models_return_finite_task_outputs(arm: str) -> None:
    torch.manual_seed(4)
    model = build_r6d_student(arm, width=8)
    outputs = model(torch.randn(2, 6, 130, 32))
    assert torch.isfinite(outputs["presence_logit"]).all()
    assert outputs["presence_logit"].shape == (2,)
    if arm == "j2_joint":
        assert outputs["candidate_probability"].shape == (2, 1 + 520 * 32)
        assert outputs["candidate_range"].shape == (2, 520, 32)
        assert outputs["candidate_doppler_vector"].shape == (2, 520, 32, 2)
        assert outputs["range_cell_estimate"].shape == (2,)
        assert outputs["doppler_vector"].shape == (2, 32, 2)
        assert outputs["doppler_vector_pooled"].shape == (2, 2)
        assert torch.isfinite(outputs["candidate_probability"]).all()
        assert torch.isfinite(outputs["range_cell_estimate"]).all()
        assert torch.isfinite(outputs["doppler_vector"]).all()
    else:
        assert outputs["range_logits"].shape == (2, 32)
        assert outputs["range_offset"].shape == (2, 32)
        assert outputs["doppler_vector"].shape == (2, 32, 2)


def test_j2_null_and_candidate_probabilities_normalize_and_define_presence() -> None:
    torch.manual_seed(5)
    model = R6J2Student(width=8).eval()
    outputs = model(torch.zeros(2, 6, 130, 32))
    probabilities = outputs["candidate_probability"]
    assert torch.all(probabilities >= 0)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
    p_null = probabilities[:, 0]
    p_present = probabilities[:, 1:].sum(dim=-1)
    torch.testing.assert_close(
        outputs["presence_logit"], torch.log(p_present) - torch.log(p_null), atol=1e-5, rtol=1e-5
    )
    logits = torch.full((2, 520 * 32), -1000.0)
    null = torch.full((2,), 1000.0)
    stable = presence_logit_from_candidate_logits(logits, null)
    assert torch.isfinite(stable).all()
    assert torch.all(stable < -1000)


def test_r6d_doppler_wrap_identifies_nyquist_endpoints() -> None:
    values = torch.tensor([PRF / 2, -PRF / 2, 3 * PRF / 2, -3 * PRF / 2])
    wrapped = wrap_r6d_doppler(values, PRF)
    expected = torch.full((4,), -PRF / 2)
    torch.testing.assert_close(wrapped, expected)
    torch.testing.assert_close(
        encode_doppler_hz(wrapped, PRF)[0], encode_doppler_hz(wrapped, PRF)[1], atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("arm", R6D_ARMS)
def test_loss_arm_masks_are_explicit_and_finite(arm: str) -> None:
    model = build_r6d_student(arm, width=8)
    outputs = model(torch.randn(3, 6, 130, 32))
    h0 = _truth(3, all_h0=True)
    terms = r6d_loss(outputs, h0, arm=arm, prf_hz=PRF)
    for value in terms.values():
        assert torch.isfinite(value)
    assert terms["range_offset"].item() == 0.0
    assert terms["doppler"].item() == 0.0
    if arm == "j1_detection":
        assert terms["presence"].item() > 0
        assert terms["total"].item() == pytest.approx(terms["presence"].item())
    elif arm == "j1_parameter":
        assert terms["presence"].item() == 0.0
        assert terms["total"].item() == 0.0
    else:
        assert terms["presence"].item() > 0
        if arm == "j2_joint":
            assert terms["candidate"].item() == 0.0


def test_parameter_only_loss_is_h1_conditional_and_joint_losses_are_composed() -> None:
    truth = _truth(2)
    for arm in ("j1_parameter", "j1_joint", "j2_joint"):
        model = build_r6d_student(arm, width=8)
        outputs = model(torch.randn(2, 6, 130, 32))
        terms = r6d_loss(outputs, truth, arm=arm, prf_hz=PRF)
        assert terms["range_offset"].item() > 0
        assert terms["doppler"].item() >= 0
        if arm != "j1_parameter":
            assert terms["presence"].item() > 0
        assert terms["total"].item() > 0


def test_j1_specialized_objectives_do_not_train_the_other_task_head() -> None:
    torch.manual_seed(24)
    inputs = torch.randn(4, 6, 130, 32)
    truth = _truth(4)
    detection_model = R6J1Student(width=8)
    detection_terms = r6d_loss(detection_model(inputs), truth, arm="j1_detection", prf_hz=PRF)
    detection_terms["total"].backward()
    assert detection_model.task_student.presence_per_range.weight.grad is not None
    assert detection_model.task_student.range_head.weight.grad is None
    assert detection_model.task_student.doppler_head.weight.grad is None

    parameter_model = R6J1Student(width=8)
    parameter_terms = r6d_loss(parameter_model(inputs), truth, arm="j1_parameter", prf_hz=PRF)
    parameter_terms["total"].backward()
    assert parameter_model.task_student.range_head.weight.grad is not None
    assert parameter_model.task_student.doppler_head.weight.grad is not None
    assert parameter_model.task_student.presence_per_range.weight.grad is None


def test_j1_and_j2_forward_accept_iq_only_and_are_batch_permutation_equivariant() -> None:
    assert list(inspect.signature(R6J1Student.forward).parameters) == ["self", "iq_ri"]
    assert list(inspect.signature(R6J2Student.forward).parameters) == ["self", "iq_ri"]
    torch.manual_seed(9)
    permutation = torch.tensor([2, 0, 1])
    inputs = torch.randn(3, 6, 130, 32)
    for model in (R6J1Student(width=8).eval(), R6J2Student(width=8).eval()):
        original = model(inputs)
        permuted = model(inputs[permutation])
        for key, value in original.items():
            if torch.is_tensor(value) and value.shape[0] == 3:
                torch.testing.assert_close(permuted[key], value[permutation], atol=1e-5, rtol=1e-5)


def test_registered_arms_share_identical_j1_weights_and_j2_trunk_initialization() -> None:
    j1_detection = build_r6d_student("j1_detection", width=8, seed=505)
    j1_parameter = build_r6d_student("j1_parameter", width=8, seed=505)
    j1_joint = build_r6d_student("j1_joint", width=8, seed=505)
    j2_joint = build_r6d_student("j2_joint", width=8, seed=505)
    for first, second in ((j1_detection, j1_parameter), (j1_detection, j1_joint)):
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    j1_trunk = j1_detection.task_student
    j2_trunk = j2_joint.trunk
    for j1_module, j2_module in (
        (j1_trunk.stem, j2_trunk.stem),
        (j1_trunk.temporal_blocks, j2_trunk.temporal_blocks),
    ):
        for key, value in j1_module.state_dict().items():
            torch.testing.assert_close(value, j2_module.state_dict()[key], atol=0, rtol=0)


def test_j2_detection_loss_backpropagates_through_parameter_feedback() -> None:
    torch.manual_seed(21)
    model = R6J2Student(width=8)
    outputs = model(torch.randn(2, 6, 130, 32))
    detection = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["presence_logit"], torch.ones(2)
    )
    detection.backward()
    grad = model.parameter_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad).item() > 0


def test_j2_parameter_loss_backpropagates_through_soft_candidate_posterior() -> None:
    torch.manual_seed(22)
    model = R6J2Student(width=8)
    outputs = model(torch.randn(2, 6, 130, 32))
    pooled = pool_r6d_j2_parameters(
        outputs["candidate_probability"],
        outputs["candidate_range"],
        outputs["candidate_doppler_vector"],
    )
    truth = _truth(2)
    loss = torch.nn.functional.smooth_l1_loss(pooled["range_cell"], truth["truth_range_cell"])
    loss = loss + (1.0 - (pooled["doppler_vector"] * encode_doppler_hz(truth["truth_doppler_hz"], PRF)).sum(-1)).mean()
    loss.backward()
    grad = model.candidate_score.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad).item() > 0


def test_j2_detached_controls_cut_the_expected_cross_task_paths() -> None:
    torch.manual_seed(23)
    model = R6J2Student(width=8)
    outputs = model(torch.randn(2, 6, 130, 32))

    detached_presence = presence_logit_from_candidate_logits(
        outputs["candidate_logits_base"] + outputs["parameter_feedback"].detach(),
        outputs["null_logit"],
    )
    detection = torch.nn.functional.binary_cross_entropy_with_logits(detached_presence, torch.ones(2))
    detection.backward(retain_graph=True)
    feedback_grad = model.parameter_head.weight.grad
    assert feedback_grad is None or torch.count_nonzero(feedback_grad).item() == 0

    model.zero_grad(set_to_none=True)
    detached_pool = pool_r6d_j2_parameters(
        outputs["candidate_probability"].detach(),
        outputs["candidate_range"],
        outputs["candidate_doppler_vector"],
    )
    truth = _truth(2)
    parameter_loss = torch.nn.functional.smooth_l1_loss(
        detached_pool["range_cell"], truth["truth_range_cell"]
    )
    parameter_loss.backward()
    score_grad = model.candidate_score.weight.grad
    assert score_grad is None or torch.count_nonzero(score_grad).item() == 0


def test_r6d_registered_paired_plans_are_deterministic_balanced_and_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "experiments/D2-A-R6/R6-D/config.yaml").read_text())
    radar = yaml.safe_load((root / "experiments/D2-A-R6/R6-A/config.yaml").read_text())["radar"]
    conditions = radar["conditions"]
    train_a = build_r6d_paired_plan(config["data"], conditions, "train")
    train_b = build_r6d_paired_plan(config["data"], conditions, "train")
    validation = build_r6d_paired_plan(config["data"], conditions, "validation")
    assert train_a == train_b
    result = audit_r6d_scene_plans(
        {"train": train_a, "validation": validation},
        expected_counts={"train": 51200, "validation": 5000},
        prior_scene_ids=set(),
    )
    assert result["overlap_count"] == 0
    assert len(result["scene_ids"]["train"]) == 51200
    assert len(result["scene_ids"]["validation"]) == 5000
    assert all(int(row["count"]) == 16 for row in train_a)


def test_r6d_budget_cosine_schedule_spans_registered_updates() -> None:
    values = [r6d_learning_rate(step) for step in (1, 6400, 12800, 19200, 25600)]
    assert values[0] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-5)
    assert all(left > right for left, right in zip(values, values[1:]))
