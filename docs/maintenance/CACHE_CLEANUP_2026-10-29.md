# Generated cache cleanup — 2026-10-29

Removed only the completed, rebuildable IQ arrays shared by D2-A-R5-A/R5-B so the approved R6-C cache can be generated without exhausting the project filesystem. The exact removed files were the H0/H1 arrays for `train`, `validation`, `development`, and `exact_overlap_boundary`, plus the H0 arrays for `h0_calibration`, `h0_verification`, and `c3_h0_verification`.

| Removed files | Bytes | GiB |
|---:|---:|---:|
| 11 | 14,186,866,048 | 13.212549 |

The arrays were local generated caches documented as rebuildable in `experiments/D2-A-R5/R5-A/analysis.md` and described by the retained R5 configuration and seed manifests. No tracked result, protocol, configuration, manifest, non-IQ label/metadata array, checkpoint, or prediction was deleted. The separate R5-C-teaching cache was not touched.

This cleanup preserves the reported R5 results and their audit metadata; regenerating the deleted IQ arrays remains possible from the recorded configurations and seeds.
