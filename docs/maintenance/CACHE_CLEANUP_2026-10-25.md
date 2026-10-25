# Generated cache cleanup — 2026-10-25

Removed only completed, regenerable training/calibration/development caches after their experiments had been evaluated. No tracked source/result files were deleted. Each `manifest.json` was copied byte-for-byte into the corresponding `cache_manifests/` directory and parsed again before cache deletion.

| Removed cache | Allocated bytes before cleanup | Files | Archived manifests |
|---|---:|---:|---:|
| `experiments/D1-E001/data/cache` | 4774612992 | 160 | 8 |
| `experiments/D1-E002/data/cache` | 4090687488 | 64 | 8 |
| `experiments/D1-E002/data/calibration_cache` | 4993482752 | 9 | 1 |
| `experiments/D2-A-E002/data/cache` | 16798277632 | 97 | 7 |
| **Total removed** | **30657060864** | **330** | **24** |

Retained: D1-E002 independent blind input cache; all experiment protocols/configurations/manifests, metrics, per-scene results, figures, audit records, training histories and best checkpoints; the untracked local GMTI source/acceptance package `GMTI_APX15_CoreX_Int16_20261014` (verified unchanged). D1 train/calibration and E002 generated IQ caches are reproducible from their frozen configuration, seeds and archived manifests.

- Project allocated size before cleanup: 40027017216 bytes.
- Project allocated size after cleanup: 9370071040 bytes.
- Preserved D1-E002 blind cache: 4090683392 bytes.
- Preserved GMTI local package: 4532666368 bytes.
- Capacity target: whole project below 20 GiB.

Archived manifest files:
- `experiments/D1-E001/cache_manifests/pilot_train.json` (copied from `experiments/D1-E001/data/cache/pilot_train/manifest.json`)
- `experiments/D1-E001/cache_manifests/pilot_validation.json` (copied from `experiments/D1-E001/data/cache/pilot_validation/manifest.json`)
- `experiments/D1-E001/cache_manifests/seed_1_train.json` (copied from `experiments/D1-E001/data/cache/seed_1_train/manifest.json`)
- `experiments/D1-E001/cache_manifests/seed_1_validation.json` (copied from `experiments/D1-E001/data/cache/seed_1_validation/manifest.json`)
- `experiments/D1-E001/cache_manifests/seed_2_train.json` (copied from `experiments/D1-E001/data/cache/seed_2_train/manifest.json`)
- `experiments/D1-E001/cache_manifests/seed_2_validation.json` (copied from `experiments/D1-E001/data/cache/seed_2_validation/manifest.json`)
- `experiments/D1-E001/cache_manifests/seed_3_train.json` (copied from `experiments/D1-E001/data/cache/seed_3_train/manifest.json`)
- `experiments/D1-E001/cache_manifests/seed_3_validation.json` (copied from `experiments/D1-E001/data/cache/seed_3_validation/manifest.json`)
- `experiments/D1-E002/cache_manifests/pilot_seed_1__train.json` (copied from `experiments/D1-E002/data/cache/pilot_seed_1/train/manifest.json`)
- `experiments/D1-E002/cache_manifests/pilot_seed_1__validation.json` (copied from `experiments/D1-E002/data/cache/pilot_seed_1/validation/manifest.json`)
- `experiments/D1-E002/cache_manifests/seed_1__train.json` (copied from `experiments/D1-E002/data/cache/seed_1/train/manifest.json`)
- `experiments/D1-E002/cache_manifests/seed_1__validation.json` (copied from `experiments/D1-E002/data/cache/seed_1/validation/manifest.json`)
- `experiments/D1-E002/cache_manifests/seed_2__train.json` (copied from `experiments/D1-E002/data/cache/seed_2/train/manifest.json`)
- `experiments/D1-E002/cache_manifests/seed_2__validation.json` (copied from `experiments/D1-E002/data/cache/seed_2/validation/manifest.json`)
- `experiments/D1-E002/cache_manifests/seed_3__train.json` (copied from `experiments/D1-E002/data/cache/seed_3/train/manifest.json`)
- `experiments/D1-E002/cache_manifests/seed_3__validation.json` (copied from `experiments/D1-E002/data/cache/seed_3/validation/manifest.json`)
- `experiments/D1-E002/cache_manifests/calibration.json` (copied from `experiments/D1-E002/data/calibration_cache/manifest.json`)
- `experiments/D2-A-E002/cache_manifests/development.json` (copied from `experiments/D2-A-E002/data/cache/development/manifest.json`)
- `experiments/D2-A-E002/cache_manifests/h0_calibration.json` (copied from `experiments/D2-A-E002/data/cache/h0_calibration/manifest.json`)
- `experiments/D2-A-E002/cache_manifests/h0_verification.json` (copied from `experiments/D2-A-E002/data/cache/h0_verification/manifest.json`)
- `experiments/D2-A-E002/cache_manifests/identifiability_boundary.json` (copied from `experiments/D2-A-E002/data/cache/identifiability_boundary/manifest.json`)
- `experiments/D2-A-E002/cache_manifests/teacher_audit.json` (copied from `experiments/D2-A-E002/data/cache/teacher_audit/manifest.json`)
- `experiments/D2-A-E002/cache_manifests/train.json` (copied from `experiments/D2-A-E002/data/cache/train/manifest.json`)
- `experiments/D2-A-E002/cache_manifests/validation.json` (copied from `experiments/D2-A-E002/data/cache/validation/manifest.json`)
