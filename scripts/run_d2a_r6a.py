#!/usr/bin/env python3
"""Run and summarize the preregistered D2-A R6-A shared-scene benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
STAGE = ROOT / 'experiments/D2-A-R6/R6-A'
METHODS = ('B0', 'B1', 'B2', 'B3', 'R3_T1', 'R3_T2', 'Medium')
BASE_FIELDS = ('scene_id', 'split', 'seed', 'condition_index', 'method', 'score',
               'range_cell', 'range_bin', 'doppler_hz', 'truth_range_cell', 'truth_doppler_hz')
FREQUENCY_NAMES = ('fft_peak', 'fft_interpolation', 'fft_zero_pad_4x',
                   'lag_one_phase_difference', 'local_profile_ml')
PREDICTION_FIELDS = list(BASE_FIELDS) + [field for name in FREQUENCY_NAMES
                                         for field in (name, 'oracle_' + name)]
ORACLE_FIELDS = ['scene_id', 'split', 'seed', 'condition_index', 'truth_range_cell',
                 'oracle_range_bin', 'truth_doppler_hz', *FREQUENCY_NAMES]

from radardistill.evaluation.d2a_r6a import (
    ResumableCSV,
    _bootstrap_mean_interval,
    audit_scene_ids,
    audit_seed_isolation,
    build_split_plan,
    candidate_frequency_estimates,
    evaluate_chunk,
    iter_scene_chunks,
    load_frozen_medium,
    summarize_detection_method,
    summarize_parameter_estimator,
    truth_range_to_bin,
    wrapped_error,
)


def read_yaml(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _git(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def preflight() -> dict:
    config = read_yaml('experiments/D2-A-R6/R6-A/config.yaml')
    r6b = read_yaml('experiments/D2-A-R6/R6-B/config.yaml')
    head = _git('rev-parse', 'HEAD')
    branch = _git('branch', '--show-current')
    remote_lines = _git('ls-remote', 'origin', 'refs/heads/main').split()
    remote = remote_lines[0] if remote_lines else ''
    tracked_changes = _git('status', '--porcelain', '--untracked-files=no')
    if branch != 'main' or head != remote or tracked_changes:
        raise RuntimeError('G0 requires committed, pushed main and no tracked changes')
    if not torch.cuda.is_available() or '3050' not in torch.cuda.get_device_name(0):
        raise RuntimeError('G0 requires RTX 3050 cuda:0')
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    if free_vram < 512 * 1024**2:
        raise RuntimeError('insufficient free CUDA memory')
    disk_free = shutil.disk_usage(ROOT).free
    if disk_free < 1024**3:
        raise RuntimeError('insufficient disk space for compact predictions')
    sources = [
        ROOT / 'experiments/D2-A-E003/scene_manifest.csv',
        ROOT / 'experiments/D2-A-R5/R5-A/scene_manifest.csv',
        ROOT / 'experiments/D2-A-R5/R5-C/selected_scene_ids.csv',
        ROOT / 'experiments/D2-A-R5/R5-C-teaching/scene_manifest.csv',
    ]
    for split in ('h0_calibration', 'h0_verification', 'c3_h0_verification', 'h1_development'):
        sources.append(ROOT / 'experiments/D2-A-R5/R5-C-teaching/predictions/T0' / f'{split}.csv')
    seed_audit = audit_seed_isolation(
        config, str(ROOT / 'experiments/D2-A-R5/R5-C-teaching/seed_manifest.csv'), r6b)
    id_audit = audit_scene_ids(config, sources)
    model = load_frozen_medium(str(ROOT / config['methods']['checkpoint_source']),
                               str(ROOT / config['methods']['checkpoint_record']))
    parameter_count = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    check = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_d2a_r6a.py', '-q'],
                           cwd=ROOT, env=env, capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f'G0 tests failed: {check.stdout}\n{check.stderr}')
    result = {
        'status': 'PASS_WITH_RECORDED_PROTOCOL_ORDER_DEVIATION',
        'git_sha': head,
        'gpu': torch.cuda.get_device_name(0),
        'cuda_free_bytes': int(free_vram),
        'cuda_total_bytes': int(total_vram),
        'disk_free_bytes': int(disk_free),
        'student_inference_parameters': parameter_count,
        'seed_audit': seed_audit,
        'scene_id_audit': id_audit,
        'test_output': check.stdout.strip(),
        'split_counts': {name: sum(row['count'] for row in build_split_plan(config, name))
                         for name in config['data']['splits']},
        'protocol_deviation': (
            'Before G0 two registered scenes were generated in engineering tests, with only shapes '
            'and IDs inspected; no IQ/label values or predictions were inspected or retained. Tests '
            'now use independent engineering seeds. See docs/task_alignments/D2-A-R6_alignment.md.'),
        'final_blind_generated': False,
    }
    path = STAGE / 'preflight.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return result


def _validate_preflight_head(execution_sha: str, head: str,
                            *, allow_analysis_code_update: bool = False) -> dict[str, str]:
    if execution_sha != head and not allow_analysis_code_update:
        raise RuntimeError('HEAD differs from G0; rerun a new committed preflight')
    return {'prediction_run_git_sha': execution_sha, 'analysis_git_sha': head}


def _frozen_preflight(*, allow_analysis_code_update: bool = False) -> tuple[dict, str]:
    path = STAGE / 'preflight.json'
    if not path.exists():
        raise RuntimeError('run the registered G0 preflight before generating scenes')
    preflight_result = json.loads(path.read_text(encoding='utf-8'))
    head = _git('rev-parse', 'HEAD')
    _validate_preflight_head(preflight_result.get('git_sha', ''), head,
                             allow_analysis_code_update=allow_analysis_code_update)
    if _git('branch', '--show-current') != 'main' or _git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('R6-A execution requires the clean, preflighted main commit')
    remote_lines = _git('ls-remote', 'origin', 'refs/heads/main').split()
    if not remote_lines or remote_lines[0] != head:
        raise RuntimeError('R6-A requires the current main commit to be pushed')
    if not torch.cuda.is_available() or '3050' not in torch.cuda.get_device_name(0):
        raise RuntimeError('R6-A requires RTX 3050 cuda:0; CPU fallback is forbidden')
    return preflight_result, head


def _write_environment(head: str, split: str) -> None:
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    invocation = {
        'git_sha': head,
        'split': split,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'cuda_free_bytes_before_split': int(free_vram),
        'cuda_total_bytes': int(total_vram),
        'disk_free_bytes_before_split': int(shutil.disk_usage(ROOT).free),
        'evaluation_chunk_size': int(read_yaml('experiments/D2-A-R6/R6-A/config.yaml')['data']['chunk_size']),
        'training_batch_changed': False,
    }
    path = STAGE / f'environment_{split}.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        if result.get('git_sha') != head or result.get('split') != split:
            raise ValueError(f'resource record identity mismatch: {path}')
    else:
        result = {'git_sha': head, 'split': split, 'invocations': []}
    result.setdefault('invocations', []).append(invocation)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def run_split(split: str) -> dict:
    _, head = _frozen_preflight()
    config = read_yaml('experiments/D2-A-R6/R6-A/config.yaml')
    physics = read_yaml(config['radar']['source_config'])
    r3_config = read_yaml('experiments/D2-A-R3/config.yaml')
    _write_environment(head, split)
    plan = build_split_plan(config, split)
    output = STAGE / 'predictions' / f'{split}.csv'
    writer = ResumableCSV(output, PREDICTION_FIELDS, head, split)
    if writer.chunks_done > len(plan):
        raise ValueError('prediction checkpoint exceeds registered chunk plan')
    if writer.state['rows'] != writer.chunks_done * len(METHODS) * int(config['data']['chunk_size']) and writer.chunks_done < len(plan):
        # The final chunk in each condition may be shorter; exact row count is checked after completion.
        minimum_rows = sum(min(int(config['data']['chunk_size']), row['count']) for row in plan[:writer.chunks_done]) * len(METHODS)
        if int(writer.state['rows']) != minimum_rows:
            raise ValueError('prediction checkpoint row count does not match completed chunks')
    model = load_frozen_medium(str(ROOT / config['methods']['checkpoint_source']),
                               str(ROOT / config['methods']['checkpoint_record']))
    torch.cuda.reset_peak_memory_stats(0)
    chunks = iter_scene_chunks(config, physics, split, start_chunk=writer.chunks_done)
    for chunk_index in range(writer.chunks_done, len(plan)):
        started = time.perf_counter()
        chunk = next(chunks)
        generation_seconds = time.perf_counter() - started
        timings = {'scene_generation': generation_seconds}
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        rows = evaluate_chunk(chunk, physics, r3_config, model, timings=timings)
        torch.cuda.synchronize(0)
        timings['chunk_total'] = time.perf_counter() - started + generation_seconds
        writer.append(rows, chunk_index, timings=timings,
                      gpu_peak_bytes=int(torch.cuda.max_memory_allocated(0)))
        if (chunk_index + 1) % 100 == 0 or chunk_index + 1 == len(plan):
            print(json.dumps({'split': split, 'chunks_done': chunk_index + 1,
                              'chunks_total': len(plan), 'scenes_done': writer.state['rows'] // len(METHODS)},
                             ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()
    expected_scenes = int(config['data']['splits'][split]['scene_count'])
    if writer.chunks_done != len(plan) or int(writer.state['rows']) != expected_scenes * len(METHODS):
        raise RuntimeError('completed prediction file does not match the registered scene plan')
    return {'split': split, 'scenes': expected_scenes, 'chunks': len(plan),
            'prediction_rows': int(writer.state['rows']),
            'csv': str(output.relative_to(ROOT)),
            'resource_record': str((STAGE / f'environment_{split}.json').relative_to(ROOT)),
            'timings_seconds': writer.state.get('timings_seconds', {}),
            'cuda_peak_allocated_bytes': writer.state.get('cuda_peak_allocated_bytes', 0)}


def recompute_oracle() -> dict:
    """Recompute only offline oracle estimators on the registered H1 scenes."""
    preflight_result, head = _frozen_preflight(allow_analysis_code_update=True)
    config = read_yaml('experiments/D2-A-R6/R6-A/config.yaml')
    physics = read_yaml(config['radar']['source_config'])
    reference_path = STAGE / 'predictions/h1_development.csv'
    reference_progress = reference_path.with_suffix(reference_path.suffix + '.progress.json')
    reference_spec = config['data']['splits']['h1_development']
    plan = build_split_plan(config, 'h1_development')
    if not reference_path.exists() or not reference_progress.exists():
        raise RuntimeError('complete H1 development predictions are required')
    expected_rows = int(reference_spec['scene_count']) * len(METHODS)
    reference_state = json.loads(reference_progress.read_text(encoding='utf-8'))
    if (reference_state.get('git_sha') != preflight_result['git_sha']
            or reference_state.get('owner') != 'D2-A-R6-A'
            or reference_state.get('split') != 'h1_development'
            or reference_state.get('fieldnames') != PREDICTION_FIELDS
            or int(reference_state.get('chunks_done', -1)) != len(plan)
            or int(reference_state.get('rows', -1)) != expected_rows
            or int(reference_state.get('byte_offset', -1)) != reference_path.stat().st_size):
        raise ValueError('H1 prediction checkpoint is incomplete or belongs to another run')
    reference = _load_split(reference_path, 'h1_development', int(reference_spec['scene_count']))['B0']
    output = STAGE / 'predictions/h1_oracle_nearest_bin.csv'
    writer = ResumableCSV(output, ORACLE_FIELDS, head, 'h1_oracle_nearest_bin')
    if writer.chunks_done > len(plan):
        raise ValueError('oracle checkpoint exceeds registered H1 chunk plan')
    chunks = iter_scene_chunks(config, physics, 'h1_development', start_chunk=writer.chunks_done)
    scene_offset = sum(row['count'] for row in plan[:writer.chunks_done])
    for chunk_index in range(writer.chunks_done, len(plan)):
        started = time.perf_counter()
        chunk = next(chunks)
        generated_seconds = time.perf_counter() - started
        count = len(chunk['scene_ids'])
        end = scene_offset + count
        expected_ids = reference['scene_id'][scene_offset:end]
        if not np.array_equal(np.asarray(chunk['scene_ids'], dtype=str), expected_ids):
            raise ValueError(f'regenerated H1 scene IDs differ at chunk {chunk_index}')
        for field, generated in (
                ('seed', np.full(count, int(chunk['spec']['seed']), dtype=np.int64)),
                ('condition_index', np.full(count, int(chunk['spec']['condition_index']), dtype=np.int64)),
                ('truth_range_cell', np.asarray(chunk['truth']['range_cell'], dtype=np.float64)),
                ('truth_doppler_hz', np.asarray(chunk['truth']['doppler_hz'], dtype=np.float64))):
            if not np.array_equal(generated, reference[field][scene_offset:end]):
                raise ValueError(f'regenerated H1 {field} differs at chunk {chunk_index}')
        bins = truth_range_to_bin(chunk['truth']['range_cell'], chunk['iq'].shape[-1])
        started = time.perf_counter()
        estimates = candidate_frequency_estimates(chunk['iq'], bins, physics)
        estimator_seconds = time.perf_counter() - started
        rows = []
        for index, scene_id in enumerate(chunk['scene_ids']):
            row = {'scene_id': str(scene_id), 'split': 'h1_development',
                   'seed': int(chunk['spec']['seed']),
                   'condition_index': int(chunk['spec']['condition_index']),
                   'truth_range_cell': float(chunk['truth']['range_cell'][index]),
                   'oracle_range_bin': int(bins[index]),
                   'truth_doppler_hz': float(chunk['truth']['doppler_hz'][index])}
            row.update({name: float(estimates[name][index]) for name in FREQUENCY_NAMES})
            rows.append(row)
        writer.append(rows, chunk_index,
                      timings={'scene_generation': generated_seconds,
                               'oracle_estimators': estimator_seconds})
        scene_offset = end
        if (chunk_index + 1) % 100 == 0 or chunk_index + 1 == len(plan):
            print(json.dumps({'split': 'h1_oracle_nearest_bin',
                              'chunks_done': chunk_index + 1, 'chunks_total': len(plan),
                              'scenes_done': writer.state['rows']}, ensure_ascii=False), flush=True)
    expected_scenes = int(reference_spec['scene_count'])
    if writer.chunks_done != len(plan) or int(writer.state['rows']) != expected_scenes:
        raise RuntimeError('completed oracle output does not match registered H1 scene plan')
    device = torch.cuda.get_device_name(0)
    record = {'analysis_git_sha': head, 'prediction_run_git_sha': reference_state['git_sha'],
              'split': 'h1_development', 'scene_count': expected_scenes,
              'mapping': 'nearest_available_range_center_half_up', 'gpu_available': True,
              'gpu': device, 'gpu_used': False, 'cpu_frequency_estimators': True,
              'timings_seconds': writer.state.get('timings_seconds', {})}
    record_path = STAGE / 'environment_oracle_recompute.json'
    payload = json.dumps(record, indent=2, ensure_ascii=False) + '\n'
    if record_path.exists() and record_path.read_text(encoding='utf-8') != payload:
        raise FileExistsError(f'refusing to overwrite oracle recompute record: {record_path}')
    if not record_path.exists():
        record_path.write_text(payload, encoding='utf-8')
    return {'scene_count': expected_scenes, 'chunks': len(plan),
            'csv': str(output.relative_to(ROOT)), 'analysis_git_sha': head,
            'timings_seconds': writer.state.get('timings_seconds', {})}


def _load_split(path: Path, split: str, expected_count: int) -> dict[str, dict[str, np.ndarray]]:
    int_fields = ('seed', 'condition_index', 'range_bin')
    float_fields = ('score', 'range_cell', 'doppler_hz', 'truth_range_cell', 'truth_doppler_hz') + tuple(
        field for name in FREQUENCY_NAMES for field in (name, 'oracle_' + name))
    fields = int_fields + float_fields
    values = {method: {field: [] for field in fields} for method in METHODS}
    ids = {method: [] for method in METHODS}
    seen = {method: set() for method in METHODS}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            method = row['method']
            if method not in values or row['split'] != split:
                raise ValueError('prediction CSV contains an unknown method or split')
            scene_id = row['scene_id']
            if scene_id in seen[method]:
                raise ValueError(f'duplicate prediction scene for {method}: {scene_id}')
            seen[method].add(scene_id)
            ids[method].append(scene_id)
            for field in int_fields:
                values[method][field].append(int(row[field]))
            for field in float_fields:
                values[method][field].append(float(row[field]) if row[field] else np.nan)
    baseline_ids = ids[METHODS[0]]
    if len(baseline_ids) != expected_count:
        raise ValueError(f'{split} has {len(baseline_ids)} scenes; expected {expected_count}')
    for method in METHODS[1:]:
        if ids[method] != baseline_ids:
            raise ValueError(f'{method} did not preserve the same scene order as {METHODS[0]}')
    arrays = {}
    for method in METHODS:
        arrays[method] = {field: np.asarray(items, dtype=np.int64 if field in int_fields else np.float64)
                          for field, items in values[method].items()}
        arrays[method]['scene_id'] = np.asarray(ids[method], dtype=str)
    return arrays


def _scalar_doppler_summary(predicted: np.ndarray, truth: np.ndarray,
                            hit: np.ndarray, candidate_correct: np.ndarray,
                            prf_hz: float) -> dict:
    error = np.asarray(wrapped_error(predicted, truth, prf_hz), dtype=np.float64)
    masks = {'all_h1': np.ones(len(error), dtype=bool), 'detection_hit': hit,
             'correct_candidate': candidate_correct,
             'detected_correct_candidate': hit & candidate_correct}
    result = {}
    for name, mask in masks.items():
        selected = error[mask]
        result[name] = {'n': int(mask.sum()),
                        'doppler_mae_hz': float(np.mean(np.abs(selected))) if selected.size else None,
                        'doppler_rmse_hz': float(np.sqrt(np.mean(selected ** 2))) if selected.size else None,
                        'doppler_bias_hz': float(np.mean(selected)) if selected.size else None}
    return result


def _load_oracle_sidecar(reference: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    path = STAGE / 'predictions/h1_oracle_nearest_bin.csv'
    progress_path = path.with_suffix(path.suffix + '.progress.json')
    if not path.exists() or not progress_path.exists():
        raise RuntimeError('nearest-bin H1 oracle sidecar is required before summarization')
    progress = json.loads(progress_path.read_text(encoding='utf-8'))
    if (progress.get('owner') != 'D2-A-R6-A'
            or not progress.get('git_sha')
            or progress.get('split') != 'h1_oracle_nearest_bin'
            or progress.get('fieldnames') != ORACLE_FIELDS
            or int(progress.get('chunks_done', -1)) != len(build_split_plan(
                read_yaml('experiments/D2-A-R6/R6-A/config.yaml'), 'h1_development'))
            or int(progress.get('rows', -1)) != len(reference['scene_id'])
            or int(progress.get('byte_offset', -1)) != path.stat().st_size):
        raise ValueError('nearest-bin oracle sidecar checkpoint is incomplete or invalid')
    rows = []
    seen = set()
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ORACLE_FIELDS:
            raise ValueError('nearest-bin oracle sidecar columns do not match')
        for row in reader:
            if row['scene_id'] in seen:
                raise ValueError(f'duplicate oracle scene ID: {row["scene_id"]}')
            seen.add(row['scene_id'])
            if row['split'] != 'h1_development':
                raise ValueError('oracle sidecar contains an unexpected split')
            rows.append(row)
    if len(rows) != len(reference['scene_id']):
        raise ValueError('oracle sidecar scene count does not match H1 reference')
    ids = np.asarray([row['scene_id'] for row in rows], dtype=str)
    seeds = np.asarray([int(row['seed']) for row in rows], dtype=np.int64)
    conditions = np.asarray([int(row['condition_index']) for row in rows], dtype=np.int64)
    truth_range = np.asarray([float(row['truth_range_cell']) for row in rows], dtype=np.float64)
    truth_doppler = np.asarray([float(row['truth_doppler_hz']) for row in rows], dtype=np.float64)
    bins = np.asarray([int(row['oracle_range_bin']) for row in rows], dtype=np.int64)
    if (not np.array_equal(ids, reference['scene_id'])
            or not np.array_equal(seeds, reference['seed'])
            or not np.array_equal(conditions, reference['condition_index'])
            or not np.array_equal(truth_range, reference['truth_range_cell'])
            or not np.array_equal(truth_doppler, reference['truth_doppler_hz'])):
        raise ValueError('oracle sidecar does not align exactly with registered H1 scenes and truth')
    config = read_yaml('experiments/D2-A-R6/R6-A/config.yaml')
    expected_bins = truth_range_to_bin(truth_range, int(config['radar']['range_bin_count']))
    if not np.array_equal(bins, expected_bins):
        raise ValueError('oracle sidecar range bins do not follow the registered nearest-center mapping')
    estimates = {name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
                 for name in FREQUENCY_NAMES}
    if any(not np.isfinite(values).all() for values in estimates.values()):
        raise ValueError('oracle sidecar contains non-finite frequency estimates')
    return estimates, progress


def summarize() -> dict:
    preflight_result, head = _frozen_preflight(allow_analysis_code_update=True)
    config = read_yaml('experiments/D2-A-R6/R6-A/config.yaml')
    split_data = {}
    all_ids = set()
    counts = {}
    resource = {}
    for split, split_spec in config['data']['splits'].items():
        result_path = STAGE / 'predictions' / f'{split}.csv'
        progress_path = result_path.with_suffix(result_path.suffix + '.progress.json')
        if not result_path.exists() or not progress_path.exists():
            raise RuntimeError(f'{split} predictions are incomplete')
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        expected_rows = int(split_spec['scene_count']) * len(METHODS)
        expected_chunks = len(build_split_plan(config, split))
        if (progress.get('owner') != 'D2-A-R6-A'
                or progress.get('git_sha') != preflight_result['git_sha']
                or progress.get('split') != split or progress.get('fieldnames') != PREDICTION_FIELDS
                or int(progress.get('chunks_done', -1)) != expected_chunks
                or int(progress.get('rows', -1)) != expected_rows
                or int(progress.get('byte_offset', -1)) != result_path.stat().st_size):
            raise ValueError(f'{split} prediction checkpoint identity or row count mismatch')
        arrays = _load_split(result_path, split, int(split_spec['scene_count']))
        ids = set(arrays[METHODS[0]]['scene_id'].tolist())
        if all_ids & ids:
            raise ValueError('scene IDs overlap across R6-A splits')
        all_ids |= ids
        split_data[split] = arrays
        counts[split] = len(ids)
        resource[split] = {
            'timings_seconds': progress.get('timings_seconds', {}),
            'seconds_per_scene': {name: float(value) / counts[split]
                                  for name, value in progress.get('timings_seconds', {}).items()},
            'cuda_peak_allocated_bytes': int(progress.get('cuda_peak_allocated_bytes', 0)),
            'prediction_rows': int(progress['rows']),
        }
    calibration = split_data['h0_calibration']
    verification = split_data['h0_verification']
    h1 = split_data['h1_development']
    oracle_estimates, oracle_progress = _load_oracle_sidecar(h1[METHODS[0]])
    oracle_record = json.loads((STAGE / 'environment_oracle_recompute.json').read_text(encoding='utf-8'))
    if (oracle_record.get('analysis_git_sha') != oracle_progress['git_sha']
            or oracle_record.get('prediction_run_git_sha') != preflight_result['git_sha']
            or oracle_record.get('scene_count') != counts['h1_development']
            or oracle_record.get('mapping') != 'nearest_available_range_center_half_up'):
        raise ValueError('oracle recompute environment record does not match sidecar provenance')
    targets = [float(value) for value in config['methods']['detector_thresholds']['target_pfa_cpi']]
    evaluation = config['evaluation']
    replicates = int(evaluation['bootstrap_replicates'])
    bootstrap_seed = int(evaluation['bootstrap_seed'])
    prf = float(config['radar']['prf_hz'])
    range_gate = float(evaluation['range_gate_cells'])
    doppler_gate = float(evaluation['doppler_gate_hz'])
    metrics = {
        'experiment': 'D2-A-R6-A', 'protocol_version': config['experiment']['protocol_version'],
        'git_sha': head, 'analysis_git_sha': head,
        'prediction_run_git_sha': preflight_result['git_sha'],
        'oracle_recompute_git_sha': oracle_progress['git_sha'],
        'oracle_gate_mapping': {
            'source': 'nearest_available_integer_range_center',
            'rounding': 'half_up', 'out_of_axis': 'clip_to_nearest_edge_bin',
            'legacy_prediction_csv_oracle_columns': 'floor_based_pre_correction_audit_only',
        },
        'g0_status': preflight_result['status'],
        'protocol_order_deviation': preflight_result['protocol_deviation'],
        'checkpoint_source': config['methods']['checkpoint_source'],
        'student_training_parameters': int(config['methods']['checkpoint_training_parameter_count']),
        'student_inference_parameters': int(preflight_result['student_inference_parameters']),
        'bootstrap': {'unit': evaluation['bootstrap_unit'], 'replicates': replicates,
                      'seed': bootstrap_seed},
        'development_data_is_final_blind': False, 'scene_counts': counts,
        'methods': {}, 'paired_native_joint_differences_at_pfa_0.05': {},
        'resources': resource,
    }
    primary_joint = {}
    for method_index, method in enumerate(METHODS):
        cal = calibration[method]
        ver = verification[method]
        test = h1[method]
        summary = summarize_detection_method(
            cal['score'], cal['condition_index'], ver['score'], ver['condition_index'],
            test['score'], test['condition_index'], targets,
            bootstrap_replicates=replicates,
            bootstrap_seed=bootstrap_seed + method_index * 1000)
        method_result = {
            'roc_auc_global_verification_vs_h1': summary['roc_auc_global'],
            'roc_auc_by_condition': summary['roc_auc_by_condition'],
            'workpoints': {},
        }
        for target in targets:
            target_key = f'{target:g}'
            detection = summary['workpoints'][target_key]
            threshold = float(detection['calibration']['threshold'])
            hit = test['score'] >= threshold
            range_error = test['range_cell'] - test['truth_range_cell']
            candidate_correct = np.abs(range_error) <= range_gate
            estimators = {'detector_native': test['doppler_hz']}
            if method == 'Medium':
                estimators = {'medium_cnn_direct': test['doppler_hz']}
            for name in FREQUENCY_NAMES:
                estimates = test[name]
                if not np.isfinite(estimates).all():
                    raise ValueError(f'{method} has missing actual-candidate estimates for {name}')
                estimators['actual_candidate_' + name] = estimates
            estimator_metrics = {}
            for name, prediction in estimators.items():
                estimator_metrics[name] = summarize_parameter_estimator(
                    test['range_cell'], test['truth_range_cell'], prediction,
                    test['truth_doppler_hz'], hit, prf_hz=prf,
                    range_gate_cells=range_gate, doppler_gate_hz=doppler_gate)
            oracle_metrics = {}
            for name in FREQUENCY_NAMES:
                prediction = oracle_estimates[name]
                if not np.isfinite(prediction).all():
                    raise ValueError(f'{method} has missing oracle-candidate estimates for {name}')
                oracle_metrics[name] = _scalar_doppler_summary(
                    prediction, test['truth_doppler_hz'], hit, candidate_correct, prf)
            detection['h1']['estimators'] = estimator_metrics
            detection['h1']['oracle_candidate_frequency_estimators_offline_only'] = oracle_metrics
            detection['h1']['joint_success_native_bootstrap_ci95'] = None
            if np.isclose(target, 0.05):
                native_name = 'medium_cnn_direct' if method == 'Medium' else 'detector_native'
                native = estimator_metrics[native_name]
                native_joint = hit & candidate_correct & (
                    np.abs(wrapped_error(test['doppler_hz'], test['truth_doppler_hz'], prf)) <= doppler_gate)
                native['joint_success_bootstrap_ci95'] = _bootstrap_mean_interval(
                    native_joint.astype(float), replicates,
                    bootstrap_seed + 100000 + method_index)
                detection['h1']['joint_success_native_bootstrap_ci95'] = native['joint_success_bootstrap_ci95']
                primary_joint[method] = native_joint.astype(float)
            by_condition = {}
            for condition in np.unique(test['condition_index']):
                mask = test['condition_index'] == condition
                condition_estimators = {}
                condition_hit = hit[mask]
                for name, prediction in estimators.items():
                    condition_estimators[name] = summarize_parameter_estimator(
                        test['range_cell'][mask], test['truth_range_cell'][mask], prediction[mask],
                        test['truth_doppler_hz'][mask], condition_hit, prf_hz=prf,
                        range_gate_cells=range_gate, doppler_gate_hz=doppler_gate)
                by_condition[str(int(condition))] = {
                    'n': int(mask.sum()), 'pd': float(np.mean(condition_hit)),
                    'estimators': condition_estimators,
                }
            detection['h1']['parameter_metrics_by_condition'] = by_condition
            method_result['workpoints'][target_key] = detection
        metrics['methods'][method] = method_result
    if 'B3' in primary_joint and 'Medium' in primary_joint:
        comparisons = {}
        for method, values in primary_joint.items():
            if method in ('B3', 'Medium'):
                continue
            comparisons[method] = {}
            for baseline in ('B3', 'Medium'):
                difference = values - primary_joint[baseline]
                comparisons[method][baseline] = {
                    'paired_joint_success_difference': float(np.mean(difference)),
                    'paired_scene_bootstrap_ci95': _bootstrap_mean_interval(
                        difference, replicates,
                        bootstrap_seed + 200000 + METHODS.index(method) * 10 + METHODS.index(baseline)),
                    'paired_scene_count': len(difference),
                }
        metrics['paired_native_joint_differences_at_pfa_0.05'] = comparisons
    _write_manifests(split_data)
    return metrics


def _write_csv_once(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    import io
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise ValueError(f'refusing to overwrite different manifest content: {path}')
        return
    with path.open('x', encoding='utf-8', newline='') as handle:
        handle.write(content)


def _write_manifests(split_data: dict) -> None:
    scene_rows, seed_counts = [], {}
    for split, methods in split_data.items():
        data = methods[METHODS[0]]
        for index, scene_id in enumerate(data['scene_id']):
            scene_rows.append({'scene_id': str(scene_id), 'split': split,
                               'seed': int(data['seed'][index]),
                               'condition_index': int(data['condition_index'][index])})
            key = (int(data['condition_index'][index]), int(data['seed'][index]))
            seed_counts[(split, *key)] = seed_counts.get((split, *key), 0) + 1
    seed_rows = [{'split': split, 'condition_index': condition, 'seed': seed,
                  'scene_count': count}
                 for (split, condition, seed), count in seed_counts.items()]
    _write_csv_once(STAGE / 'scene_manifest.csv', ['scene_id', 'split', 'seed', 'condition_index'], scene_rows)
    _write_csv_once(STAGE / 'seed_manifest.csv', ['split', 'condition_index', 'seed', 'scene_count'], seed_rows)


def _save_figure_once(fig, path: Path) -> None:
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=160)
    content = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'refusing to overwrite different figure bytes: {path}')
        return
    path.write_bytes(content)


def _joint_success_by_condition(method_metrics: dict, method: str,
                                condition_ids: list[str]) -> list[float]:
    native = 'medium_cnn_direct' if method == 'Medium' else 'detector_native'
    per_condition = method_metrics['workpoints']['0.05']['h1']['parameter_metrics_by_condition']
    return [float(per_condition[condition]['estimators'][native]['joint_success'])
            for condition in condition_ids]


def _condition_plot_labels(conditions: list[dict]) -> list[str]:
    return [f"{index}: {item['course']} {item['scenario_id']}\nSNR {float(item['snr_db']):g} dB"
            for index, item in enumerate(conditions)]


def _write_figures(metrics: dict) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config = read_yaml('experiments/D2-A-R6/R6-A/config.yaml')
    arrays = {split: _load_split(STAGE / 'predictions' / f'{split}.csv', split,
                                 int(config['data']['splits'][split]['scene_count']))
              for split in config['data']['splits']}
    h0, h1 = arrays['h0_verification'], arrays['h1_development']
    outdir = STAGE / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        negative, positive = h0[method]['score'], h1[method]['score']
        combined = np.concatenate((negative, positive))
        labels = np.concatenate((np.zeros(len(negative), dtype=np.int8), np.ones(len(positive), dtype=np.int8)))
        order = np.argsort(-combined, kind='mergesort')
        sorted_scores, sorted_labels = combined[order], labels[order]
        tp = np.cumsum(sorted_labels)
        last = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
        tpr = tp[last] / len(positive)
        fpr = (last + 1 - tp[last]) / len(negative)
        ax.plot(np.r_[0., fpr], np.r_[0., tpr], label=method)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax.set(xlabel='False alarm probability', ylabel='Detection probability', title='R6-A ROC: independent H0 verification and H1 development')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / 'roc_curves.png'
    _save_figure_once(fig, path)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    methods_metrics = metrics['methods']
    condition_ids = sorted(methods_metrics['B3']['workpoints']['0.05']['verification_by_condition'], key=int)
    labels = _condition_plot_labels(config['radar']['conditions'])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(condition_ids))
    for method in METHODS:
        vals = [methods_metrics[method]['workpoints']['0.05']['verification_by_condition'][condition]['pfa']
                for condition in condition_ids]
        ax.plot(x, vals, marker='o', label=method)
    ax.axhline(.05, color='gray', linestyle='--', linewidth=1)
    ax.set(xticks=x, xticklabels=labels, ylabel='Pfa per CPI', title='Independent H0 verification by condition at global Pfa 0.05 threshold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = outdir / 'pfa_by_condition.png'
    _save_figure_once(fig, path)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for method in METHODS:
        vals = _joint_success_by_condition(methods_metrics[method], method, condition_ids)
        ax.plot(x, vals, marker='o', label=method)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel='Joint success',
           title='H1 joint success by condition at globally calibrated Pfa 0.05')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = outdir / 'joint_success_by_condition.png'
    _save_figure_once(fig, path)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))

    names = METHODS
    actual_rmse, oracle_rmse = [], []
    for method in names:
        h1_metrics = methods_metrics[method]['workpoints']['0.05']['h1']
        native_name = 'medium_cnn_direct' if method == 'Medium' else 'detector_native'
        actual_rmse.append(h1_metrics['estimators'][native_name]['all_h1']['doppler_rmse_hz'])
        oracle_rmse.append(h1_metrics['oracle_candidate_frequency_estimators_offline_only']['local_profile_ml']['all_h1']['doppler_rmse_hz'])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = .38
    ax.bar(x - width / 2, actual_rmse, width, label='Detector native candidate')
    ax.bar(x + width / 2, oracle_rmse, width, label='Oracle range candidate, profile ML')
    ax.set(xticks=x, xticklabels=names, ylabel='Doppler RMSE (Hz)', title='All-H1 Doppler RMSE: actual and offline oracle range candidates')
    ax.legend()
    fig.tight_layout()
    path = outdir / 'doppler_rmse_actual_oracle.png'
    _save_figure_once(fig, path)
    plt.close(fig)
    written.append(str(path.relative_to(ROOT)))
    return written


def _write_analysis(metrics: dict, figures: list[str]) -> Path:
    lines = [
        '# D2-A-R6-A same-scene detector and estimator baseline', '',
        f"**Git SHA:** `{metrics['git_sha']}`  ",
        f"**Prediction run SHA:** `{metrics['prediction_run_git_sha']}`  ",
        f"**Nearest-bin oracle recompute SHA:** `{metrics['oracle_recompute_git_sha']}`  ",
        f"**G0:** {metrics['g0_status']}  ",
        f"**Final blind:** {metrics['development_data_is_final_blind']}  ",
        f"**Protocol-order deviation:** {metrics['protocol_order_deviation']}", '',
        '## Detection summary at global Pfa 0.05', '',
        '| Method | AUC | Verification Pfa | Pfa one-sided 95% upper | Pd | Joint success | Native Doppler RMSE (Hz) |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method, result in metrics['methods'].items():
        wp = result['workpoints']['0.05']
        estimator = wp['h1']['estimators']['medium_cnn_direct' if method == 'Medium' else 'detector_native']
        lines.append('| {m} | {auc:.4f} | {pfa:.4f} | {upper:.4f} | {pd:.4f} | {joint:.4f} | {rmse:.2f} |'.format(
            m=method, auc=result['roc_auc_global_verification_vs_h1'],
            pfa=wp['verification']['pfa'], upper=wp['verification']['pfa_ci95_exact_upper'],
            pd=wp['h1']['pd'], joint=estimator['joint_success'],
            rmse=estimator['all_h1']['doppler_rmse_hz']))
    lines += ['', '## Figures', ''] + [f'- `{figure}`' for figure in figures]
    lines += ['', '## Interpretation limits', '',
              '- R6-A uses development scenes and does not establish final blind performance.',
              '- R3 T1/T2 remain development candidates without fixed-threshold qualification.',
              '- Oracle-candidate estimates are offline diagnostics and cannot be used as deployable performance.',
              '- Oracle distance uses the nearest available integer range-bin center; old floor-based `oracle_*` columns in the original prediction CSV are retained for audit only, while canonical metrics use the recomputed sidecar.',
              '- Detector and frequency-estimator errors are reported separately; joint success includes detection, range, and Doppler gates.',
              '- Timing records describe this offline evaluation boundary, not full system latency.']
    path = STAGE / 'analysis.md'
    content = '\n'.join(lines) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise FileExistsError(f'refusing to overwrite existing analysis: {path}')
    else:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(content)
    return path


def run() -> None:
    config = read_yaml('experiments/D2-A-R6/R6-A/config.yaml')
    for split in config['data']['splits']:
        print(json.dumps(run_split(split), ensure_ascii=False), flush=True)
    metrics = summarize()
    figures = _write_figures(metrics)
    metrics['figures'] = figures
    metrics_path = STAGE / 'metrics.json'
    payload = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if metrics_path.exists():
        if metrics_path.read_text(encoding='utf-8') != payload:
            raise FileExistsError(f'refusing to overwrite existing metrics: {metrics_path}')
    else:
        with metrics_path.open('x', encoding='utf-8') as handle:
            handle.write(payload)
    analysis = _write_analysis(metrics, figures)
    print(json.dumps({'metrics': str(metrics_path.relative_to(ROOT)),
                      'analysis': str(analysis.relative_to(ROOT)), 'figures': figures},
                     ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'run', 'recompute_oracle', 'summarize'))
    parser.add_argument('--split', choices=('h0_calibration', 'h0_verification', 'h1_development'))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'preflight':
        print(json.dumps(preflight(), indent=2, ensure_ascii=False))
    elif args.action == 'run':
        if args.split:
            print(json.dumps(run_split(args.split), indent=2, ensure_ascii=False))
        else:
            run()
    elif args.action == 'recompute_oracle':
        print(json.dumps(recompute_oracle(), indent=2, ensure_ascii=False))
    else:
        metrics = summarize()
        figures = _write_figures(metrics)
        metrics['figures'] = figures
        payload = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        metrics_path = STAGE / 'metrics.json'
        if metrics_path.exists() and metrics_path.read_text(encoding='utf-8') != payload:
            raise FileExistsError(f'refusing to overwrite existing metrics: {metrics_path}')
        if not metrics_path.exists():
            with metrics_path.open('x', encoding='utf-8') as handle:
                handle.write(payload)
        _write_analysis(metrics, figures)
        print(json.dumps({'metrics': str(metrics_path.relative_to(ROOT)), 'figures': figures}, ensure_ascii=False))


if __name__ == '__main__':
    main()
