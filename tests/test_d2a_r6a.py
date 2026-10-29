import numpy as np
import pytest

from radardistill.evaluation.d2a_r6a import calibrate_threshold, matched_predictions, wrapped_error


def test_methods_receive_same_iq_without_truth():
    iq = np.ones((3, 130, 32), dtype=np.complex64)
    seen = []
    def method(x):
        seen.append(x)
        return {"score": float(np.abs(x).max())}
    out = matched_predictions(iq, {"a": method, "b": method})
    assert seen[0] is iq and seen[1] is iq
    assert out == {"a": {"score": 1.0}, "b": {"score": 1.0}}


def test_calibration_higher_quantile_and_ties():
    result = calibrate_threshold(np.arange(100.0), 0.05)
    assert result["threshold"] == 95.0
    assert result["pfa_gt"] == 0.04
    assert result["pfa_ge"] == 0.05
    assert result["tie_count"] == 1


def test_wrap_at_nyquist_and_reject_invalid_iq():
    assert wrapped_error(-649.0, 649.0, 1300.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        matched_predictions(np.zeros((3, 130, 32)), {"a": lambda x: x})


def test_candidate_frequency_uses_selected_range_gate_and_wraps():
    from radardistill.evaluation.d2a_r6a import candidate_frequency_estimates
    import yaml
    with open('experiments/D2-A-E001/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    n = np.arange(130)
    iq = np.zeros((2, 3, 130, 32), dtype=np.complex128)
    iq[0, 0, :, 7] = np.exp(2j * np.pi * 123.4 * n / 1300)
    iq[1, 0, :, 9] = np.exp(2j * np.pi * -648.0 * n / 1300)
    estimates = candidate_frequency_estimates(iq, np.array([7, 9]), config)
    assert set(estimates) == {'fft_peak', 'fft_interpolation', 'fft_zero_pad_4x', 'lag_one_phase_difference', 'local_profile_ml'}
    assert np.max(np.abs(wrapped_error(estimates['local_profile_ml'], [123.4, -648.0], 1300))) < 0.01
    assert np.max(np.abs(wrapped_error(estimates['lag_one_phase_difference'], [123.4, -648.0], 1300))) < 0.01
    with pytest.raises(ValueError):
        candidate_frequency_estimates(iq, np.array([7, 32]), config)


def test_classical_detectors_match_existing_baselines():
    import yaml
    from radardistill.d2a_baselines import run_b3
    from radardistill.evaluation.d2a_r6a import classical_detector_predictions
    with open('experiments/D2-A-E001/config.yaml', encoding='utf-8') as f:
        physics = yaml.safe_load(f)
    with open('experiments/D2-A-R3/config.yaml', encoding='utf-8') as f:
        r3 = yaml.safe_load(f)
    rng = np.random.default_rng(82)
    iq = (rng.standard_normal((1, 3, 130, 32)) + 1j * rng.standard_normal((1, 3, 130, 32)))
    out = classical_detector_predictions(iq, physics, r3)
    assert set(out) == {'B0', 'B1', 'B2', 'B3', 'R3_T1', 'R3_T2'}
    original = run_b3(iq, 1300., physics)
    np.testing.assert_allclose(out['B3'].score, original.score)
    np.testing.assert_allclose(out['B3'].range_cell, original.range_cell)
    assert all(np.isfinite(np.asarray(value.score)).all() for value in out.values())


def test_frozen_medium_checkpoint_cuda_export_preserves_task_outputs():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA device unavailable')
    from radardistill.evaluation.d2a_r6a import load_frozen_medium, predict_medium
    model = load_frozen_medium('experiments/D2-A-R5/R5-A/capacity/runs/seed_101/Medium/best_checkpoint.pt', 'experiments/D2-A-R5/R5-A/capacity/runs/seed_101/Medium/training_record.json')
    assert sum(p.numel() for p in model.parameters()) == 70277
    rng = np.random.default_rng(123)
    iq = (rng.normal(size=(2, 3, 130, 32)) + 1j*rng.normal(size=(2, 3, 130, 32))).astype(np.complex64)
    prediction = predict_medium(model, iq)
    assert set(prediction) == {'score', 'predicted_range_cell', 'predicted_range_bin', 'predicted_doppler_hz'}
    assert all(np.asarray(value).shape == (2,) and np.isfinite(value).all() for value in prediction.values())
    assert next(model.parameters()).is_cuda
    from radardistill.students.task_student import TaskStudent
    original = TaskStudent(width=96, include_spectrum_head=True).cuda().eval()
    payload = torch.load('experiments/D2-A-R5/R5-A/capacity/runs/seed_101/Medium/best_checkpoint.pt', map_location='cpu', weights_only=False)
    original.load_state_dict(payload['model_state_dict'], strict=True)
    tensor = torch.from_numpy(np.concatenate((iq.real, iq.imag), axis=1)).cuda()
    with torch.inference_mode():
        reference = original(tensor, compute_spectrum=False)
        exported = model(tensor, compute_spectrum=False)
    for key in reference:
        torch.testing.assert_close(reference[key], exported[key], rtol=0, atol=0)


def test_registered_split_plan_counts_and_disjoint_seed_blocks():
    import yaml
    from radardistill.evaluation.d2a_r6a import build_split_plan
    with open('experiments/D2-A-R6/R6-A/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    plans = {name: build_split_plan(config, name) for name in config['data']['splits']}
    assert {k: sum(row['count'] for row in v) for k, v in plans.items()} == {'h0_calibration': 12000, 'h0_verification': 12000, 'h1_development': 6144}
    all_seeds = [row['seed'] for plan in plans.values() for row in plan]
    assert len(all_seeds) == len(set(all_seeds))
    assert all(row['count'] <= 16 for plan in plans.values() for row in plan)
    bad = yaml.safe_load(yaml.safe_dump(config))
    bad['data']['splits']['h0_verification']['seed_base'] = bad['data']['splits']['h0_calibration']['seed_base']
    with pytest.raises(ValueError):
        build_split_plan(bad, 'h0_verification')


def test_streamed_chunk_preserves_scene_ids_and_presence():
    import yaml
    from radardistill.evaluation.d2a_r6a import iter_scene_chunks
    with open('experiments/D2-A-R6/R6-A/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open('experiments/D2-A-E001/config.yaml', encoding='utf-8') as f:
        physics = yaml.safe_load(f)
    for index, split in enumerate(config['data']['splits'].values()):
        split['count_per_condition'] = 1
        split['scene_count'] = 12
        split['seed_base'] = 2000000000 + index * 20000000
    h0 = next(iter_scene_chunks(config, physics, 'h0_calibration'))
    h1 = next(iter_scene_chunks(config, physics, 'h1_development'))
    assert h0['iq'].shape == h1['iq'].shape == (1, 3, 130, 32)
    assert h0['truth'] is None
    assert h1['truth']['range_cell'].shape == (1,)
    assert set(h0['scene_ids']).isdisjoint(h1['scene_ids'])


def test_fixed_threshold_summary_excludes_verification_from_calibration():
    from radardistill.evaluation.d2a_r6a import summarize_workpoint
    cal = np.arange(100., dtype=float)
    verification = np.array([96., 95., 0., -1.])
    scores = np.array([99., 90., 99.])
    result = summarize_workpoint(cal, verification, scores, np.array([3., 3., 6.]), np.array([649., 0., 10.]), np.array([3., 3., 3.]), np.array([-649., 0., 10.]), 0.05, 1300.)
    assert result['calibration']['threshold'] == 95.
    assert result['verification']['pfa'] == .5
    assert result['h1']['pd'] == pytest.approx(2/3)
    assert result['h1']['joint_success'] == pytest.approx(1/3)
    assert result['h1']['doppler_rmse_hz'] == pytest.approx(np.sqrt(4/3))
    changed = summarize_workpoint(cal, verification + 1000, scores, np.array([3., 3., 6.]), np.array([649., 0., 10.]), np.array([3., 3., 3.]), np.array([-649., 0., 10.]), 0.05, 1300.)
    assert changed['calibration']['threshold'] == 95.


def test_matched_detector_predictions_and_separate_oracle_frequency():
    import yaml
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA device unavailable')
    from radardistill.evaluation.d2a_r6a import load_frozen_medium, predict_observed, oracle_frequency_estimates
    with open('experiments/D2-A-E001/config.yaml', encoding='utf-8') as f:
        physics = yaml.safe_load(f)
    with open('experiments/D2-A-R3/config.yaml', encoding='utf-8') as f:
        r3 = yaml.safe_load(f)
    model = load_frozen_medium('experiments/D2-A-R5/R5-A/capacity/runs/seed_101/Medium/best_checkpoint.pt', 'experiments/D2-A-R5/R5-A/capacity/runs/seed_101/Medium/training_record.json')
    rng = np.random.default_rng(987)
    iq = (rng.normal(size=(1, 3, 130, 32)) + 1j*rng.normal(size=(1, 3, 130, 32))).astype(np.complex64)
    results = predict_observed(iq, physics, r3, model, estimate_doppler=True)
    assert set(results) == {'B0', 'B1', 'B2', 'B3', 'R3_T1', 'R3_T2', 'Medium'}
    assert all(item['score'].shape == (1,) and item['range_cell'].shape == (1,) for item in results.values())
    assert all(item['frequency_estimates']['local_profile_ml'].shape == (1,) for item in results.values())
    oracle = oracle_frequency_estimates(iq, np.array([7.8]), physics)
    assert oracle['local_profile_ml'].shape == (1,)


def test_oracle_range_gate_uses_nearest_available_range_center(monkeypatch):
    import radardistill.evaluation.d2a_r6a as r6a
    captured = {}

    def fake_candidate_estimates(iq, bins, physics):
        captured['bins'] = np.asarray(bins).copy()
        names = ('fft_peak', 'fft_interpolation', 'fft_zero_pad_4x',
                 'lag_one_phase_difference', 'local_profile_ml')
        return {name: np.zeros(len(iq)) for name in names}

    monkeypatch.setattr(r6a, 'candidate_frequency_estimates', fake_candidate_estimates)
    iq = np.zeros((3, 3, 130, 32), dtype=np.complex128)
    truth_range = np.array([10.8787, 10.3, 31.8])

    result = r6a.oracle_frequency_estimates(iq, truth_range, {})

    np.testing.assert_array_equal(captured['bins'], [11, 10, 31])
    assert result['local_profile_ml'].shape == (3,)


def test_truth_range_to_bin_rounds_half_up_and_clips_to_axis():
    from radardistill.evaluation.d2a_r6a import truth_range_to_bin
    np.testing.assert_array_equal(
        truth_range_to_bin(np.array([.5, 1.5, -2., 31.8]), 32),
        [1, 2, 0, 31],
    )
    with pytest.raises(ValueError):
        truth_range_to_bin(np.array([np.nan]), 32)
    with pytest.raises(ValueError):
        truth_range_to_bin(np.array([1.]), 0)


def test_seed_audit_rejects_previous_or_reserved_seed_blocks():
    import yaml
    from radardistill.evaluation.d2a_r6a import audit_seed_isolation
    with open('experiments/D2-A-R6/R6-A/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open('experiments/D2-A-R6/R6-B/config.yaml', encoding='utf-8') as f:
        r6b = yaml.safe_load(f)
    result = audit_seed_isolation(config, 'experiments/D2-A-R5/R5-C-teaching/seed_manifest.csv', r6b)
    assert result['r5c_seed_overlap'] == 0 and result['r6b_reserved_overlap'] == 0
    bad = yaml.safe_load(yaml.safe_dump(config))
    bad['data']['splits']['h0_calibration']['seed_base'] = 1008000000
    with pytest.raises(ValueError, match='R5-C'):
        audit_seed_isolation(bad, 'experiments/D2-A-R5/R5-C-teaching/seed_manifest.csv', r6b)
    bad = yaml.safe_load(yaml.safe_dump(config))
    bad['data']['splits']['h0_calibration']['seed_base'] = 1508000000
    with pytest.raises(ValueError, match='R6-B'):
        audit_seed_isolation(bad, 'experiments/D2-A-R5/R5-C-teaching/seed_manifest.csv', r6b)


def test_cuda_r3_statistics_match_frozen_cpu_implementations():
    import yaml
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA device unavailable')
    from radardistill.evaluation.d2a_r6a_cuda import cuda_r3_predictions
    with open('experiments/D2-A-E001/config.yaml', encoding='utf-8') as f:
        physics = yaml.safe_load(f)
    with open('experiments/D2-A-R3/config.yaml', encoding='utf-8') as f:
        r3 = yaml.safe_load(f)
    from radardistill.evaluation.d2a_r6a import classical_detector_predictions
    rng = np.random.default_rng(981)
    iq = rng.normal(size=(2, 3, 130, 32)) + 1j*rng.normal(size=(2, 3, 130, 32))
    reference = classical_detector_predictions(iq, physics, r3)
    result = cuda_r3_predictions(iq, physics, r3)
    for name in ('B3', 'R3_T1', 'R3_T2'):
        np.testing.assert_allclose(result[name].score, reference[name].score, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(result[name].range_cell, reference[name].range_cell)
        np.testing.assert_array_equal(result[name].doppler_hz, reference[name].doppler_hz)


def test_chunk_prediction_keeps_truth_outside_observed_methods(monkeypatch):
    from radardistill.evaluation import d2a_r6a
    iq = np.ones((2, 3, 130, 32), dtype=np.complex64)
    truth = {'range_cell': np.array([1.1, 2.2]), 'doppler_hz': np.array([5., 7.])}
    def fake_observed(x, physics, r3, model, *, estimate_doppler, timings=None):
        assert x is iq and estimate_doppler
        return {'X': {'score': np.array([1., 2.]), 'range_cell': np.array([1., 2.]), 'range_bin': np.array([1, 2]), 'doppler_hz': np.array([5., 6.]), 'frequency_estimates': {'local_profile_ml': np.array([5., 6.])}}}
    def fake_oracle(x, ranges, physics):
        np.testing.assert_array_equal(ranges, truth['range_cell'])
        return {'local_profile_ml': np.array([5.1, 7.1])}
    monkeypatch.setattr(d2a_r6a, 'predict_observed', fake_observed)
    monkeypatch.setattr(d2a_r6a, 'oracle_frequency_estimates', fake_oracle)
    chunk = {'spec': {'split': 'h1_development', 'condition_index': 3, 'seed': 11}, 'iq': iq, 'scene_ids': np.array(['a', 'b']), 'truth': truth}
    rows = d2a_r6a.evaluate_chunk(chunk, {}, {}, None)
    assert len(rows) == 2 and rows[0]['scene_id'] == 'a'
    assert rows[0]['oracle_local_profile_ml'] == 5.1
    assert rows[0]['truth_range_cell'] == 1.1


def test_scene_id_audit_checks_known_manifests(tmp_path):
    import yaml
    from radardistill.evaluation.d2a_r6a import audit_scene_ids
    with open('experiments/D2-A-R6/R6-A/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    manifest = tmp_path / 'old.csv'
    manifest.write_text('base_scene_id\nd2a-scene-1008000000-0\n', encoding='utf-8')
    assert audit_scene_ids(config, [manifest])['overlap'] == 0
    manifest.write_text('base_scene_id\nd2a-scene-1208000000-0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='scene ID'):
        audit_scene_ids(config, [manifest])


def test_csv_checkpoint_recovers_only_uncommitted_partial_chunk(tmp_path):
    from radardistill.evaluation.d2a_r6a import ResumableCSV
    path = tmp_path / 'scores.csv'
    writer = ResumableCSV(path, ['scene_id', 'score'], 'abc', 'h0_calibration')
    writer.append([{'scene_id': 'a', 'score': 1.}], 0)
    with path.open('ab') as f:
        f.write(b'partial-corrupt-row')
    recovered = ResumableCSV(path, ['scene_id', 'score'], 'abc', 'h0_calibration')
    assert recovered.chunks_done == 1
    assert b'partial-corrupt-row' not in path.read_bytes()
    recovered.append([{'scene_id': 'b', 'score': 2.}], 1)
    assert path.read_text().splitlines() == ['scene_id,score', 'a,1.0', 'b,2.0']
    with pytest.raises(ValueError, match='identity'):
        ResumableCSV(path, ['scene_id', 'score'], 'wrong', 'h0_calibration')


def test_parameter_estimator_reports_all_detection_and_candidate_conditioning():
    from radardistill.evaluation.d2a_r6a import summarize_parameter_estimator
    result = summarize_parameter_estimator(
        predicted_range=np.array([2., 5.]), truth_range=np.array([2., 4.]),
        predicted_doppler=np.array([649., -649.]), truth_doppler=np.array([-649., 649.]),
        detection_mask=np.array([True, False]), prf_hz=1300.,
    )
    assert result['all_h1']['n'] == 2
    assert result['all_h1']['range_rmse_cells'] == pytest.approx(np.sqrt(.5))
    assert result['all_h1']['doppler_rmse_hz'] == pytest.approx(2.)
    assert result['detection_hit']['n'] == 1
    assert result['correct_candidate']['n'] == 2
    assert result['detected_correct_candidate']['n'] == 1
    assert result['joint_success'] == pytest.approx(.5)


def test_detection_summary_uses_global_calibration_and_reports_condition_migration():
    from radardistill.evaluation.d2a_r6a import summarize_detection_method
    result = summarize_detection_method(
        calibration_scores=np.array([0., 1., 2., 3., 4.]),
        calibration_conditions=np.array([0, 0, 1, 1, 1]),
        verification_scores=np.array([3., 4., 2., 1.]),
        verification_conditions=np.array([0, 0, 1, 1]),
        h1_scores=np.array([5., 2., 4., 0.]),
        h1_conditions=np.array([0, 0, 1, 1]),
        target_pfas=[.4], bootstrap_seed=7,
    )
    wp = result['workpoints']['0.4']
    assert wp['calibration']['threshold'] == 3.
    assert wp['verification']['pfa'] == .5
    assert wp['verification']['pfa_gt'] == .25
    assert wp['verification_by_condition']['0']['pfa'] == 1.
    assert wp['verification_by_condition']['0']['pfa_gt'] == .5
    assert wp['verification_by_condition']['1']['pfa'] == 0.
    assert wp['h1']['pd'] == .5
    assert result['roc_auc_global'] == pytest.approx(.5625)


def test_runner_imports_the_registered_streamed_scene_iterator():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('r6a_runner_test', Path('scripts/run_d2a_r6a.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert callable(runner.iter_scene_chunks)
    assert callable(runner._bootstrap_mean_interval)


def test_postrun_summary_accepts_separate_analysis_sha_but_run_does_not():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('r6a_runner_sha_test', Path('scripts/run_d2a_r6a.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with pytest.raises(RuntimeError, match='differs'):
        runner._validate_preflight_head('prediction-sha', 'analysis-sha')
    assert runner._validate_preflight_head(
        'prediction-sha', 'analysis-sha', allow_analysis_code_update=True
    ) == {'prediction_run_git_sha': 'prediction-sha', 'analysis_git_sha': 'analysis-sha'}


def test_joint_success_figure_reads_condition_specific_metrics():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('r6a_runner_joint_test', Path('scripts/run_d2a_r6a.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    metrics = {'workpoints': {'0.05': {'h1': {'parameter_metrics_by_condition': {
        '0': {'estimators': {'medium_cnn_direct': {'joint_success': 0.25}}},
        '1': {'estimators': {'medium_cnn_direct': {'joint_success': 0.75}}},
    }}}}}
    assert runner._joint_success_by_condition(metrics, 'Medium', ['0', '1']) == [0.25, 0.75]


def test_condition_plot_labels_distinguish_snr_repeats():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('r6a_runner_label_test', Path('scripts/run_d2a_r6a.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    labels = runner._condition_plot_labels([
        {'course': 'C3', 'scenario_id': 'partial_overlap', 'snr_db': 0.0},
        {'course': 'C3', 'scenario_id': 'partial_overlap', 'snr_db': 10.0},
    ])
    assert len(set(labels)) == 2
    assert 'SNR 0 dB' in labels[0]
    assert 'SNR 10 dB' in labels[1]
