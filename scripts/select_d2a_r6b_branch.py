#!/usr/bin/env python3
"""Select the single preregistered R6-B branch from permitted R6-A splits."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from radardistill.evaluation.d2a_e001 import empirical_auc
from scripts.run_d2a_r6a import _load_split, _git, read_yaml


def bootstrap_auc_interval(h0_scores: np.ndarray, h1_scores: np.ndarray,
                           replicates: int, seed: int) -> tuple[float, list[float]]:
    """Two-sample scene bootstrap for a condition ROC AUC."""
    h0 = np.asarray(h0_scores, dtype=np.float64)
    h1 = np.asarray(h1_scores, dtype=np.float64)
    if (h0.ndim != 1 or h1.ndim != 1 or not h0.size or not h1.size
            or not np.isfinite(h0).all() or not np.isfinite(h1).all()
            or replicates <= 0):
        raise ValueError('AUC bootstrap requires finite nonempty scene scores and positive replicates')
    point = float(empirical_auc(h0, h1))
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        h0_sample = h0[rng.integers(0, len(h0), size=len(h0))]
        h1_sample = h1[rng.integers(0, len(h1), size=len(h1))]
        values[index] = empirical_auc(h0_sample, h1_sample)
    return point, [float(value) for value in np.quantile(values, [.025, .975])]


def select_branch(*, c3_auc: float, c3_ci95: list[float], hard_auc: float,
                  hard_ci95: list[float], calibration_pfa_range: float,
                  scale_spearman_abs: float | None = None,
                  scale_spearman_ci95: list[float] | None = None,
                  rules: dict) -> tuple[str, list[str]]:
    """Apply the frozen R6-B S/R/U rule without consulting verification data."""
    branch = rules['branch_selection_rule']
    scale_interval_excludes_zero = (
        scale_spearman_ci95 is not None
        and (scale_spearman_ci95[0] > 0 or scale_spearman_ci95[1] < 0)
    )
    scale_ready = (
        scale_spearman_abs is not None
        and scale_spearman_abs >= float(branch['score_scale_spearman_abs_minimum'])
        and (scale_interval_excludes_zero if branch['spearman_95_ci_excludes_zero'] else True)
    )
    s_conditions = [
        c3_auc >= float(branch['c3_pooled_auc_minimum']),
        c3_ci95[0] > float(branch['c3_pooled_auc_95_lower_minimum']),
        hard_auc >= float(branch['hard_cell_auc_minimum']),
        hard_ci95[0] > float(branch['hard_cell_auc_95_lower_minimum']),
        calibration_pfa_range >= float(branch['c3_calibration_cell_pfa_range_minimum']),
        scale_ready,
    ]
    if all(s_conditions):
        return 'S', ['all preregistered scale-shift branch conditions passed']
    reasons = []
    if c3_ci95[1] < float(branch['c3_pooled_auc_minimum']):
        reasons.append('C3 pooled AUC 95% upper bound is below 0.75')
    if hard_ci95[1] < float(branch['hard_cell_auc_minimum']):
        reasons.append('hard-cell AUC 95% upper bound is below 0.70')
    if reasons:
        return 'R', reasons
    return 'U', ['neither frozen S nor R trigger is met']


def run() -> dict:
    r6a_dir = ROOT / 'experiments/D2-A-R6/R6-A'
    r6b_dir = ROOT / 'experiments/D2-A-R6/R6-B'
    config = read_yaml('experiments/D2-A-R6/R6-B/config.yaml')
    r6a_config = read_yaml('experiments/D2-A-R6/R6-A/config.yaml')
    metrics = json.loads((r6a_dir / 'metrics.json').read_text(encoding='utf-8'))
    calibration = _load_split(r6a_dir / 'predictions/h0_calibration.csv',
                              'h0_calibration', 12000)['Medium']
    h1 = _load_split(r6a_dir / 'predictions/h1_development.csv',
                     'h1_development', 6144)['Medium']
    c3_h0 = calibration['condition_index'] >= 6
    c3_h1 = h1['condition_index'] >= 6
    hard_h0 = calibration['condition_index'] == 10
    hard_h1 = h1['condition_index'] == 10
    replicates = int(config['evaluation']['bootstrap_replicates'])
    seed = int(r6a_config['evaluation']['bootstrap_seed'])
    c3_auc, c3_ci = bootstrap_auc_interval(
        calibration['score'][c3_h0], h1['score'][c3_h1], replicates, seed)
    hard_auc, hard_ci = bootstrap_auc_interval(
        calibration['score'][hard_h0], h1['score'][hard_h1], replicates, seed + 1)
    threshold = float(metrics['methods']['Medium']['workpoints']['0.05']['calibration']['threshold'])
    c3_cell_pfas = {}
    for condition in range(6, 12):
        mask = calibration['condition_index'] == condition
        c3_cell_pfas[str(condition)] = float(np.mean(calibration['score'][mask] >= threshold))
    pfa_range = max(c3_cell_pfas.values()) - min(c3_cell_pfas.values())
    branch, reasons = select_branch(
        c3_auc=c3_auc, c3_ci95=c3_ci, hard_auc=hard_auc, hard_ci95=hard_ci,
        calibration_pfa_range=pfa_range,
        rules=config['experiment']['branches'])
    head = _git('rev-parse', 'HEAD')
    result = {
        'experiment': 'D2-A-R6-B', 'protocol_version': config['experiment']['protocol_version'],
        'selection_git_sha': head, 'r6a_prediction_run_git_sha': metrics['prediction_run_git_sha'],
        'r6a_metrics_git_sha': metrics['analysis_git_sha'],
        'input_splits': ['R6-A H0 calibration', 'R6-A H1 development'],
        'r6a_h0_verification_read': False,
        'final_blind_read_or_generated': False,
        'bootstrap': {'unit': 'independent base_scene_id resampling within H0 and H1',
                      'replicates': replicates, 'seed_c3': seed, 'seed_hard_cell': seed + 1},
        'method': 'frozen Medium width96',
        'c3_pooled': {'condition_indices': [6, 7, 8, 9, 10, 11],
                      'h0_calibration_n': int(c3_h0.sum()), 'h1_development_n': int(c3_h1.sum()),
                      'auc': c3_auc, 'ci95': c3_ci},
        'hard_cell': {'condition_index': 10,
                      'definition': 'C3 partial_overlap / SNR=0 dB / CNR=20 dB',
                      'h0_calibration_n': int(hard_h0.sum()), 'h1_development_n': int(hard_h1.sum()),
                      'auc': hard_auc, 'ci95': hard_ci},
        'c3_h0_calibration_pfa_by_condition_at_global_0.05_threshold': c3_cell_pfas,
        'c3_h0_calibration_pfa_range': float(pfa_range),
        'score_scale_spearman': {
            'status': 'not computed: C3 calibration-Pfa range already fails the preregistered S-branch conjunction',
            'value': None, 'ci95': None,
        },
        'selected_branch': branch, 'selection_reasons': reasons,
        'action': ('Run only the preregistered matched truth-only vs training-H0 hard-negative-resampled '
                   'Medium comparison.' if branch == 'R' else 'Follow the selected branch in protocol v1.0.'),
    }
    path = r6b_dir / 'branch_selection.json'
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != payload:
        raise FileExistsError(f'refusing to overwrite branch selection: {path}')
    if not path.exists():
        path.write_text(payload, encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, ensure_ascii=False))
