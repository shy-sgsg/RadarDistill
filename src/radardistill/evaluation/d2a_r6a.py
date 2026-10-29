"""Shared-input evaluation primitives for the preregistered R6-A baseline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def matched_predictions(iq: np.ndarray, methods: Mapping[str, Callable[[np.ndarray], Any]]) -> dict[str, Any]:
    """Call every method with the same observable IQ; truth is never supplied."""
    values = np.asarray(iq)
    if values.ndim not in (3, 4) or not np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("IQ must be a finite complex CPI or batch")
    if not methods:
        raise ValueError("at least one method is required")
    return {name: method(values) for name, method in methods.items()}


def calibrate_threshold(scores: np.ndarray, target_pfa: float) -> dict[str, float | int]:
    """Freeze a higher empirical quantile from calibration scores alone."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a finite nonempty vector")
    if not 0.0 < target_pfa < 1.0:
        raise ValueError("target_pfa must lie strictly between zero and one")
    threshold = float(np.quantile(values, 1.0 - target_pfa, method="higher"))
    return {"threshold": threshold, "pfa_gt": float(np.mean(values > threshold)),
            "pfa_ge": float(np.mean(values >= threshold)), "tie_count": int(np.sum(values == threshold))}


def wrapped_error(predicted: Any, truth: Any, prf_hz: float) -> Any:
    """Signed PRF-periodic prediction error in [-PRF/2, PRF/2)."""
    if not np.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    return (np.asarray(predicted) - np.asarray(truth) + prf_hz / 2) % prf_hz - prf_hz / 2


def candidate_frequency_estimates(iq: np.ndarray, range_bins: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Estimate Doppler in explicitly supplied candidate bins of the DPCA residual.

    The caller must construct these bins from detector output; oracle bins may
    only be passed by a separately labelled offline analysis.
    """
    from radardistill.baselines import estimate_fft_interpolated, estimate_fft_peak, estimate_mle_local, estimate_phase_difference
    from radardistill.d2a_baselines import dpca_difference

    values = np.asarray(iq)
    if values.ndim != 4 or not np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("IQ must be a finite complex batch [B,C,N,R]")
    bins = np.asarray(range_bins)
    if bins.shape != (len(values),) or not np.isfinite(bins).all() or np.any(bins != np.floor(bins)):
        raise ValueError("range_bins must contain one integer bin per CPI")
    bins = bins.astype(np.int64)
    if np.any(bins < 0) or np.any(bins >= values.shape[-1]):
        raise ValueError("range bin outside IQ")
    residual = dpca_difference(values, dict(config))
    samples = residual[np.arange(len(values)), :, bins]
    prf = float(config["radar"]["prf_hz"])
    nfft = 4 * samples.shape[-1]
    power = np.abs(np.fft.fft(samples, n=nfft, axis=-1)) ** 2
    frequencies = np.fft.fftfreq(nfft, d=1.0 / prf)
    return {
        "fft_peak": np.asarray(estimate_fft_peak(samples, prf)),
        "fft_interpolation": np.asarray(estimate_fft_interpolated(samples, prf)),
        "fft_zero_pad_4x": frequencies[np.argmax(power, axis=-1)],
        "lag_one_phase_difference": np.asarray(estimate_phase_difference(samples, prf)),
        "local_profile_ml": np.asarray(estimate_mle_local(samples, prf)),
    }


def classical_detector_predictions(iq: np.ndarray, physics: Mapping[str, Any], r3_config: Mapping[str, Any], *, use_cuda_r3: bool = False) -> dict[str, Any]:
    """Evaluate the frozen B0–B3 and R3 statistics on the same IQ batch."""
    from radardistill.d2a_baselines import run_b0, run_b1, run_b2, run_b3
    from radardistill.d2a_r3_baselines import run_t1_global_normalized, run_t2_local_cfar

    prf = float(physics['radar']['prf_hz'])
    options = r3_config['detection']
    methods = {
        'B0': lambda x: run_b0(x, prf),
        'B1': lambda x: run_b1(x, prf),
        'B2': lambda x: run_b2(x, prf, dict(physics)),
    }
    if not use_cuda_r3:
        methods.update({
            'B3': lambda x: run_b3(x, prf, dict(physics)),
            'R3_T1': lambda x: run_t1_global_normalized(x, prf, dict(physics), options['t1_global_robust_background']),
            'R3_T2': lambda x: run_t2_local_cfar(x, prf, dict(physics), options['t2_local_cfar_like']),
        })
    results = matched_predictions(iq, methods)
    if use_cuda_r3:
        from radardistill.evaluation.d2a_r6a_cuda import cuda_r3_predictions
        results.update(cuda_r3_predictions(iq, physics, r3_config))
    return results


def load_frozen_medium(checkpoint_path: str, record_path: str):
    """Load the registered R5-A Medium weights and export only the task model."""
    import json
    from pathlib import Path
    import torch
    from radardistill.students.task_student import TaskStudent, export_inference_model

    if not torch.cuda.is_available() or '3050' not in torch.cuda.get_device_name(0):
        raise RuntimeError('R6-A requires the RTX 3050 CUDA device')
    record = json.loads(Path(record_path).read_text(encoding='utf-8'))
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    expected = {'experiment': 'D2-A-R5', 'subphase': 'R5-A', 'arm': 'S0', 'width': 96, 'global_update': 12800}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('registered Medium checkpoint identity does not match')
    if payload.get('r5_config', {}).get('training', {}).get('seed') != 101:
        raise ValueError('registered Medium checkpoint training seed does not match')
    if (record.get('source_git_sha') != payload.get('source_git_sha') or record.get('width') != 96
            or record.get('selected_update') != 12800 or record.get('parameter_count') != 82887):
        raise ValueError('Medium checkpoint training record does not match')
    model = TaskStudent(width=96, include_spectrum_head=True)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 82887:
        raise ValueError('training parameter count does not match protocol')
    exported = export_inference_model(model)
    if sum(p.numel() for p in exported.parameters()) != 70277:
        raise ValueError('inference parameter count does not match protocol')
    return exported.to(torch.device('cuda:0')).eval()


def predict_medium(model: Any, iq: np.ndarray) -> dict[str, np.ndarray]:
    """Predict from observable IQ on the registered CUDA device only."""
    import torch
    from radardistill.evaluation.d2a_r5c_teaching_pilot import predict_task_student_chunk

    if next(model.parameters()).device != torch.device('cuda:0'):
        raise RuntimeError('student is not resident on cuda:0')
    return predict_task_student_chunk(model, iq, torch.device('cuda:0'), prf_hz=1300.0)


def build_split_plan(config: Mapping[str, Any], split_name: str) -> list[dict[str, Any]]:
    """Derive bounded scene chunks and reject overlaps between registered seed blocks."""
    data = config['data']
    splits = data['splits']
    if split_name not in splits:
        raise ValueError('unknown split')
    conditions = config['radar']['conditions']
    stride = int(data['condition_seed_stride'])
    chunk_size = int(data['chunk_size'])
    chunk_stride = int(data['chunk_seed_stride'])
    if len(conditions) != int(data['condition_count']) or min(stride, chunk_size, chunk_stride) <= 0:
        raise ValueError('invalid condition or chunk configuration')
    intervals = []
    for name, split in splits.items():
        base = int(split['seed_base'])
        intervals.append((base, base + len(conditions) * stride, name))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError('registered split seed blocks overlap')
    split = splits[split_name]
    per_condition = int(split['count_per_condition'])
    if per_condition <= 0 or per_condition * len(conditions) != int(split['scene_count']):
        raise ValueError('split counts do not match conditions')
    rows = []
    for index, condition in enumerate(conditions):
        remaining = per_condition
        chunk = 0
        while remaining:
            count = min(chunk_size, remaining)
            if chunk * chunk_stride >= stride:
                raise ValueError('chunk seeds exceed condition block')
            rows.append({'split': split_name, 'presence': split['presence'], 'condition_index': index,
                         'condition': condition, 'count': count,
                         'seed': int(split['seed_base']) + index * stride + chunk * chunk_stride})
            remaining -= count
            chunk += 1
    return rows


def iter_scene_chunks(config: Mapping[str, Any], physics: Mapping[str, Any], split_name: str):
    """Generate registered scene chunks; truth stays separate from observable IQ."""
    from radardistill.simulation.d2a import generate_d2a_paired_batch

    for spec in build_split_plan(config, split_name):
        condition = spec['condition']
        batch = generate_d2a_paired_batch(
            spec['count'], spec['seed'], condition['snr_db'], condition['cnr_db'],
            dict(physics), scenario_id=condition['scenario_id'],
        )
        if spec['presence'] not in ('H0', 'H1'):
            raise ValueError('split presence must be H0 or H1')
        truth = None
        if spec['presence'] == 'H1':
            truth = {'range_cell': np.asarray(batch.truth_target_range_cell),
                     'doppler_hz': np.asarray(batch.truth_target_doppler_hz)}
        yield {'spec': spec, 'iq': batch.iq_h0 if truth is None else batch.iq_h1,
               'scene_ids': np.asarray(batch.base_scene_id).astype(str), 'truth': truth}


def summarize_workpoint(cal_scores: np.ndarray, verification_scores: np.ndarray, h1_scores: np.ndarray,
                        predicted_range: np.ndarray, predicted_doppler: np.ndarray,
                        truth_range: np.ndarray, truth_doppler: np.ndarray,
                        target_pfa: float, prf_hz: float) -> dict[str, Any]:
    """Apply a calibration-only threshold and evaluate independent H0/H1 samples."""
    from scipy.stats import beta
    calibration = calibrate_threshold(cal_scores, target_pfa)
    threshold = calibration['threshold']
    h0 = np.asarray(verification_scores, dtype=np.float64)
    arrays = [np.asarray(v, dtype=np.float64) for v in (h1_scores, predicted_range, predicted_doppler, truth_range, truth_doppler)]
    if h0.ndim != 1 or h0.size == 0 or not np.isfinite(h0).all():
        raise ValueError('independent H0 scores must be finite and nonempty')
    if any(a.ndim != 1 or a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays) or arrays[0].size == 0:
        raise ValueError('H1 arrays must be finite, aligned, and nonempty')
    score, prange, pdoppler, trange, tdoppler = arrays
    n = h0.size
    false_alarms = int(np.count_nonzero(h0 >= threshold))
    upper = 1.0 if false_alarms == n else float(beta.ppf(.95, false_alarms + 1, n - false_alarms))
    hit = score >= threshold
    range_error = prange - trange
    doppler_error = wrapped_error(pdoppler, tdoppler, prf_hz)
    correct = np.abs(range_error) <= 1.0
    joint = hit & correct & (np.abs(doppler_error) <= 20.0)
    def errors(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {'n': 0, 'range_rmse_cells': None, 'doppler_rmse_hz': None}
        return {'n': int(mask.sum()), 'range_rmse_cells': float(np.sqrt(np.mean(range_error[mask] ** 2))),
                'doppler_rmse_hz': float(np.sqrt(np.mean(doppler_error[mask] ** 2)))}
    return {'calibration': calibration,
            'verification': {'n': int(n), 'false_alarms': false_alarms, 'pfa': false_alarms / n,
                             'pfa_ci95_exact_upper': upper, 'tie_count': int(np.count_nonzero(h0 == threshold))},
            'h1': {'n': int(len(score)), 'pd': float(np.mean(hit)), 'joint_success': float(np.mean(joint)),
                   'doppler_rmse_hz': float(np.sqrt(np.mean(doppler_error ** 2))),
                   'all_h1': errors(np.ones(len(score), dtype=bool)),
                   'detection_hit': errors(hit), 'correct_candidate': errors(hit & correct)}}


def predict_observed(iq: np.ndarray, physics: Mapping[str, Any], r3_config: Mapping[str, Any],
                     model: Any, *, estimate_doppler: bool) -> dict[str, dict[str, Any]]:
    """Predict all detection methods from one IQ batch without truth labels."""
    classical = classical_detector_predictions(iq, physics, r3_config, use_cuda_r3=True)
    observed: dict[str, dict[str, Any]] = {}
    for name, output in classical.items():
        observed[name] = {'score': np.asarray(output.score), 'range_cell': np.asarray(output.range_cell),
                          'range_bin': np.asarray(output.range_cell, dtype=np.int64),
                          'doppler_hz': np.asarray(output.doppler_hz)}
    student = predict_medium(model, iq)
    observed['Medium'] = {'score': student['score'], 'range_cell': student['predicted_range_cell'],
                          'range_bin': student['predicted_range_bin'], 'doppler_hz': student['predicted_doppler_hz']}
    if estimate_doppler:
        for item in observed.values():
            item['frequency_estimates'] = candidate_frequency_estimates(iq, item['range_bin'], physics)
    return observed


def oracle_frequency_estimates(iq: np.ndarray, truth_range: np.ndarray, physics: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Offline-only oracle gate; must never be used to select a runtime candidate."""
    truth = np.asarray(truth_range, dtype=np.float64)
    if truth.ndim != 1 or not np.isfinite(truth).all():
        raise ValueError('oracle truth ranges must be finite vector')
    return candidate_frequency_estimates(iq, np.floor(truth).astype(np.int64), physics)


def audit_seed_isolation(config: Mapping[str, Any], r5c_manifest: str, r6b_config: Mapping[str, Any]) -> dict[str, int]:
    """Reject registered seeds used by R5-C or reserved for R6-B."""
    import csv
    plans = [row for name in config['data']['splits'] for row in build_split_plan(config, name)]
    seeds = {row['seed'] for row in plans}
    if len(seeds) != len(plans):
        raise ValueError('R6-A chunk seeds overlap')
    with open(r5c_manifest, newline='', encoding='utf-8') as handle:
        prior = {int(row['seed']) for row in csv.DictReader(handle)}
    overlap = seeds & prior
    if overlap:
        raise ValueError('R5-C seed overlap')
    r6b_data = r6b_config['data']
    stride = int(r6b_data['condition_seed_stride'])
    count = int(r6b_data['condition_count'])
    reserved = []
    for split in r6b_data['splits'].values():
        if 'seed_base' in split:
            base = int(split['seed_base'])
            reserved.append((base, base + count * stride))
        for base in split.get('seed_bases_by_course', []):
            reserved.append((int(base), int(base) + 5 * stride))
    if any(lo <= seed < hi for seed in seeds for lo, hi in reserved):
        raise ValueError('R6-B reserved seed overlap')
    return {'r6a_chunk_seeds': len(seeds), 'r5c_known_seeds': len(prior),
            'r5c_seed_overlap': 0, 'r6b_reserved_overlap': 0}


FREQUENCY_NAMES = ('fft_peak', 'fft_interpolation', 'fft_zero_pad_4x', 'lag_one_phase_difference', 'local_profile_ml')


def evaluate_chunk(chunk: Mapping[str, Any], physics: Mapping[str, Any], r3_config: Mapping[str, Any], model: Any) -> list[dict[str, Any]]:
    """Create compact matched rows; oracle calculations occur only after prediction."""
    iq, truth, spec = chunk['iq'], chunk['truth'], chunk['spec']
    observed = predict_observed(iq, physics, r3_config, model, estimate_doppler=truth is not None)
    oracle = oracle_frequency_estimates(iq, truth['range_cell'], physics) if truth is not None else {}
    ids = np.asarray(chunk['scene_ids'])
    if ids.shape != (len(iq),) or len(np.unique(ids)) != len(ids):
        raise ValueError('invalid or repeated scene IDs')
    rows = []
    for method, prediction in observed.items():
        for i, scene_id in enumerate(ids):
            row = {'scene_id': str(scene_id), 'split': spec['split'], 'seed': spec['seed'],
                   'condition_index': spec['condition_index'], 'method': method,
                   'score': float(prediction['score'][i]), 'range_cell': float(prediction['range_cell'][i]),
                   'range_bin': int(prediction['range_bin'][i]), 'doppler_hz': float(prediction['doppler_hz'][i]),
                   'truth_range_cell': None if truth is None else float(truth['range_cell'][i]),
                   'truth_doppler_hz': None if truth is None else float(truth['doppler_hz'][i])}
            for name in FREQUENCY_NAMES:
                actual = prediction.get('frequency_estimates', {}).get(name)
                row[name] = None if actual is None else float(actual[i])
                row['oracle_' + name] = None if name not in oracle else float(oracle[name][i])
            rows.append(row)
    return rows
