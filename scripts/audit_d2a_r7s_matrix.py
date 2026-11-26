#!/usr/bin/env python3
"""Independent integrity and provenance audit for the R7 unified matrix.

This script deliberately does not import the matrix builder. It validates the
published CSV against its declared source files and R7-D checkpoint inventory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NUMERIC_COLUMNS = {
    "snr_db", "cnr_db", "parameter_count", "trainable_parameter_count",
    "training_updates", "n_scenes", "n_h0_calibration",
    "n_h0_verification", "n_h1", "training_seed_count", "target_pfa_cpi",
    "pfa_cpi", "false_alarm_cpi_count", "n_valid_h0_cpi", "pfa_ci95_lower",
    "pfa_ci95_upper", "valid_cuts_per_cpi", "valid_h0_cut_count", "false_cut_count",
    "mean_false_cut_exceedances_per_cpi", "pfa_cell", "pfa_cell_ci95_upper", "pd",
    "pd_ci95_lower", "pd_ci95_upper", "auc", "auc_ci95_lower", "auc_ci95_upper",
    "range_rmse_cells", "range_rmse_detected_cells", "doppler_rmse_hz",
    "doppler_rmse_detected_hz", "joint_success", "teacher_localization_rate",
    "teacher_response_lift", "coherence_before_abs_median", "coherence_after_abs_median",
    "phase_residual_p95_rad", "aligned_residual_power_ratio", "clutter_suppression_db",
    "inference_wall_seconds", "gpu_peak_allocated_mib",
}
RATE_COLUMNS = {
    "target_pfa_cpi", "pfa_cpi", "pfa_ci95_lower", "pfa_ci95_upper",
    "pfa_cell", "pfa_cell_ci95_upper", "pd", "pd_ci95_lower", "pd_ci95_upper",
    "auc", "auc_ci95_lower", "auc_ci95_upper", "joint_success",
    "teacher_localization_rate", "coherence_before_abs_median", "coherence_after_abs_median",
}
NONNEGATIVE_COLUMNS = {
    "parameter_count", "trainable_parameter_count", "training_updates",
    "training_scene_exposure", "n_scenes", "n_h0_calibration", "n_h0_verification",
    "n_h1", "training_seed_count", "false_alarm_cpi_count", "n_valid_h0_cpi",
    "valid_cuts_per_cpi", "valid_h0_cut_count", "false_cut_count",
    "mean_false_cut_exceedances_per_cpi", "range_rmse_cells", "range_rmse_detected_cells",
    "doppler_rmse_hz", "doppler_rmse_detected_hz",
    "phase_residual_p95_rad", "aligned_residual_power_ratio",
    "inference_wall_seconds", "gpu_peak_allocated_mib",
}
CONFIDENCE_PAIRS = [
    ("pfa_cpi", "pfa_ci95_lower", "pfa_ci95_upper"),
    ("pd", "pd_ci95_lower", "pd_ci95_upper"),
    ("auc", "auc_ci95_lower", "auc_ci95_upper"),
]
IDENTITY_KEY = (
    "study", "experiment", "record_grain", "dataset", "split", "metric_group",
    "condition_key", "method_id", "seed", "checkpoint_role", "source_metrics", "source_pointer",
)


def _status_value(raw: str) -> bool:
    value = raw.strip().upper()
    return value.startswith((
        "NOT_", "NOT-", "UNVERIFIED", "INSUFFICIENT_", "MISSING_", "DEFERRED",
        "UNKNOWN", "UNAVAILABLE", "NO_", "NOT RUN", "NOT APPLICABLE",
    )) or value in {"N/A", "NA", "NONE", "—", "-"}


def _ci_endpoints_reversed(lower: float | None, upper: float | None) -> bool:
    """Percentile bootstrap intervals need not contain the observed point estimate."""
    return lower is not None and upper is not None and lower > upper + 1e-12


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON Pointer, including array indices and escapes."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON Pointer must be empty or start with '/': {pointer!r}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if token == "-" or not token.isdigit():
                raise ValueError(f"invalid array index {token!r} in {pointer!r}")
            index = int(token)
            current = current[index]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(f"cannot traverse {type(current).__name__} at {token!r}")
    return current


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _float(value: str) -> float | None:
    if not value or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        raise ValueError("non-finite numeric value")
    return number


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _source_paths(row: dict[str, str], root: Path) -> list[tuple[str, Path]]:
    paths = []
    for field in ("source_metrics", "source_protocol", "source_analysis"):
        value = (row.get(field) or "").strip()
        if value:
            paths.append((field, root / value))
    return paths


def _validate_pointer(row: dict[str, str], root: Path, json_cache: dict[Path, Any]) -> tuple[str, str, int]:
    pointer = (row.get("source_pointer") or "").strip()
    if not pointer:
        return "MISSING_POINTER", "source_pointer is empty", 0

    sources = _source_paths(row, root)
    if pointer.startswith("/"):
        metric_path = root / (row.get("source_metrics") or "")
        if not metric_path.is_file():
            return "SOURCE_FILE_MISSING", f"JSON source missing: {metric_path}", 0
        try:
            if metric_path not in json_cache:
                json_cache[metric_path] = json.loads(metric_path.read_text(encoding="utf-8"))
            resolve_json_pointer(json_cache[metric_path], pointer)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            return "UNRESOLVED_JSON_POINTER", str(exc), 0
        return "RESOLVED_JSON_POINTER", "", 1

    if pointer.startswith("csv row "):
        metric_path = root / (row.get("source_metrics") or "")
        if not metric_path.is_file():
            return "SOURCE_FILE_MISSING", f"CSV source missing: {metric_path}", 0
        match = re.fullmatch(r"csv row method=(.*), snr_db=([^,]+)", pointer)
        if not match:
            return "INVALID_CSV_POINTER", pointer, 0
        method, snr = match.groups()
        try:
            _, source_rows = _read_csv(metric_path)
            matches = [r for r in source_rows if r.get("method") == method and r.get("snr_db") == snr]
        except (OSError, UnicodeError, csv.Error) as exc:
            return "UNREADABLE_CSV_SOURCE", str(exc), 0
        if len(matches) == 1:
            return "RESOLVED_CSV_ROW", "", 1
        if len(matches) > 1:
            return "AMBIGUOUS_CSV_ROW", f"{len(matches)} rows match method={method}, snr_db={snr}", len(matches)
        return "UNRESOLVED_CSV_ROW", f"no row matches method={method}, snr_db={snr}", 0

    if pointer.startswith("markdown table row at line "):
        match = re.fullmatch(r"markdown table row at line (\d+):\s*(.*)", pointer)
        if not match:
            return "INVALID_MARKDOWN_POINTER", pointer, 0
        target_line, target = int(match.group(1)), match.group(2)
        markdown = [(field, path) for field, path in sources if path.suffix.lower() in {".md", ".markdown"}]
        for _, path in markdown:
            if path.is_file():
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                if 1 <= target_line <= len(lines) and (not target or target in lines[target_line - 1]):
                    return "RESOLVED_MARKDOWN_ROW", f"{path}:{target_line}", 1
        return "UNRESOLVED_MARKDOWN_ROW", f"line {target_line} does not contain {target!r}", 0

    if pointer.startswith("markdown table row:"):
        target = pointer.split(":", 1)[1].strip()
        markdown = [(field, path) for field, path in sources if path.suffix.lower() in {".md", ".markdown"}]
        matches: list[tuple[Path, int]] = []
        for _, path in markdown:
            if path.is_file():
                for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if target in line:
                        matches.append((path, line_no))
        if len(matches) == 1:
            return "RESOLVED_MARKDOWN_ROW", f"{matches[0][0]}:{matches[0][1]}", 1
        if len(matches) > 1:
            locations = ";".join(f"{p}:{n}" for p, n in matches[:8])
            return "AMBIGUOUS_MARKDOWN_ROW", f"{len(matches)} matches: {locations}", len(matches)
        return "UNRESOLVED_MARKDOWN_ROW", f"no source row contains {target!r}", 0

    return "UNSUPPORTED_POINTER_FORMAT", pointer, 0


def audit_matrix(root: Path, matrix_path: Path, out_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    matrix_path = matrix_path if matrix_path.is_absolute() else root / matrix_path
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    headers, rows = _read_csv(matrix_path)
    findings: list[dict[str, Any]] = []

    def issue(severity: str, code: str, row_index: int | str, field: str, detail: str) -> None:
        findings.append({"severity": severity, "code": code, "row_index": row_index, "field": field, "detail": detail})

    for required in ("study", "experiment", "dataset", "split", "record_grain", "metric_group", "method_id", "source_pointer", "source_metrics"):
        if required not in headers:
            issue("CRITICAL", "REQUIRED_COLUMN_MISSING", "", required, "required matrix field absent")

    exact_rows: Counter[tuple[str, ...]] = Counter(tuple(row.get(h, "") for h in headers) for row in rows)
    exact_dup_rows = sum(n - 1 for n in exact_rows.values() if n > 1)
    if exact_dup_rows:
        issue("HIGH", "EXACT_DUPLICATE_ROWS", "", "row", f"{exact_dup_rows} extra rows are exact duplicates")

    key_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=2):
        key_groups[tuple(row.get(k, "") for k in IDENTITY_KEY)].append(index)
        for column in NUMERIC_COLUMNS.intersection(headers):
            raw = row.get(column, "")
            if not raw:
                continue
            try:
                value = _float(raw)
            except ValueError:
                issue("HIGH", "NONFINITE_NUMERIC", index, column, repr(raw))
                continue
            if value is None:
                if _status_value(raw):
                    continue
                issue("MEDIUM", "NONNUMERIC_IN_NUMERIC_FIELD", index, column, repr(raw))
                continue
            if column in RATE_COLUMNS and not 0.0 <= value <= 1.0:
                issue("HIGH", "RATE_OUT_OF_RANGE", index, column, repr(value))
            if column in NONNEGATIVE_COLUMNS and value < 0.0:
                issue("HIGH", "NEGATIVE_NONNEGATIVE_METRIC", index, column, repr(value))
            if column in {"parameter_count", "trainable_parameter_count", "training_updates", "n_scenes", "n_h0_calibration", "n_h0_verification", "n_h1", "training_seed_count", "false_alarm_cpi_count", "n_valid_h0_cpi", "valid_cuts_per_cpi", "valid_h0_cut_count", "false_cut_count"} and not value.is_integer():
                issue("MEDIUM", "NONINTEGER_COUNT", index, column, repr(value))

        parsed = {c: None for c in RATE_COLUMNS.intersection(headers)}
        for column in parsed:
            try:
                parsed[column] = _float(row.get(column, ""))
            except ValueError:
                pass
        for value_col, lower_col, upper_col in CONFIDENCE_PAIRS:
            low, high = parsed.get(lower_col), parsed.get(upper_col)
            if _ci_endpoints_reversed(low, high):
                issue("HIGH", "CI_ENDPOINTS_REVERSED", index, value_col, f"lower={low} > upper={high}")

        unit = (row.get("false_alarm_unit") or "").upper()
        cell_value = row.get("pfa_cell", "")
        if "CPI" in unit:
            try:
                if _float(cell_value) is not None:
                    issue("CRITICAL", "CPI_REPORTED_AS_CELL_PFA", index, "pfa_cell", f"false_alarm_unit={unit}; value={cell_value}")
            except ValueError:
                pass
        status = (row.get("metric_status") or "").upper()
        if status == "NOT_RUN":
            measured = []
            for column in ("pfa_cpi", "pd", "auc", "range_rmse_cells", "doppler_rmse_hz", "joint_success"):
                try:
                    if _float(row.get(column, "")) is not None:
                        measured.append(column)
                except ValueError:
                    measured.append(column)
            if measured:
                issue("HIGH", "NOT_RUN_HAS_NUMERIC_RESULT", index, "metric_status", ",".join(measured))

    repeated_identity_groups = [(key, indexes) for key, indexes in key_groups.items() if len(indexes) > 1]
    for key, indexes in repeated_identity_groups:
        issue("MEDIUM", "REPEATED_RECORD_IDENTITY", indexes[0], "identity", f"rows={indexes}; key={key}")

    json_cache: dict[Path, Any] = {}
    source_validation: list[dict[str, Any]] = []
    path_stats = Counter()
    for index, row in enumerate(rows, start=2):
        path_results = []
        for field, path in _source_paths(row, root):
            if path.is_file():
                path_stats[f"{field}_exists"] += 1
                path_results.append(f"{field}=EXISTS")
            else:
                path_stats[f"{field}_missing"] += 1
                path_results.append(f"{field}=MISSING:{path}")
                issue("HIGH", "SOURCE_FILE_MISSING", index, field, str(path))
        pointer_status, pointer_detail, pointer_match_count = _validate_pointer(row, root, json_cache)
        path_stats[pointer_status] += 1
        if pointer_status.startswith(("UNRESOLVED", "AMBIGUOUS", "INVALID", "UNSUPPORTED", "SOURCE_FILE_MISSING", "UNREADABLE")):
            severity = "HIGH" if pointer_status.startswith("UNRESOLVED") or pointer_status == "SOURCE_FILE_MISSING" else "MEDIUM"
            issue(severity, pointer_status, index, "source_pointer", pointer_detail)
        source_validation.append({
            "matrix_row": index,
            "study": row.get("study", ""), "experiment": row.get("experiment", ""),
            "method_id": row.get("method_id", ""), "dataset": row.get("dataset", ""),
            "split": row.get("split", ""), "condition_key": row.get("condition_key", ""),
            "source_metrics": row.get("source_metrics", ""), "source_pointer": row.get("source_pointer", ""),
            "pointer_status": pointer_status, "pointer_match_count": pointer_match_count,
            "pointer_detail": pointer_detail, "source_file_checks": ";".join(path_results),
        })

    inventory_counts: Counter[tuple[str, ...]] = Counter()
    inventory_fields = ["study", "experiment", "dataset", "split", "record_grain", "metric_group", "method_role", "method_id", "metric_status", "applicability_status"]
    for row in rows:
        inventory_counts[tuple(row.get(key, "") for key in inventory_fields)] += 1
    inventory_rows = []
    for keys, count in sorted(inventory_counts.items()):
        inventory_rows.append({**dict(zip(inventory_fields, keys)), "record_count": count})

    checkpoint_inventory_path = root / "experiments/D2-A-R7/R7-D/checkpoint_inventory.csv"
    checkpoint_results = []
    if checkpoint_inventory_path.is_file():
        _, checkpoint_rows = _read_csv(checkpoint_inventory_path)
        for inventory_row in checkpoint_rows:
            path = root / (inventory_row.get("checkpoint_path") or "")
            actual_sha = ""
            actual_bytes: int | None = None
            if path.is_file():
                actual_bytes = path.stat().st_size
                import hashlib
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                actual_sha = digest.hexdigest()
            expected_sha = inventory_row.get("checkpoint_sha256", "")
            expected_bytes = inventory_row.get("checkpoint_bytes", "")
            checks = []
            if not path.is_file():
                checks.append("MISSING")
                issue("HIGH", "CHECKPOINT_MISSING", inventory_row.get("method_id", ""), "checkpoint_path", str(path))
            if expected_sha and actual_sha and actual_sha != expected_sha:
                checks.append("SHA256_MISMATCH")
                issue("CRITICAL", "CHECKPOINT_SHA_MISMATCH", inventory_row.get("method_id", ""), "checkpoint_sha256", f"expected={expected_sha}; actual={actual_sha}")
            if expected_bytes and actual_bytes is not None and int(expected_bytes) != actual_bytes:
                checks.append("SIZE_MISMATCH")
                issue("HIGH", "CHECKPOINT_SIZE_MISMATCH", inventory_row.get("method_id", ""), "checkpoint_bytes", f"expected={expected_bytes}; actual={actual_bytes}")
            checkpoint_results.append({
                "method_id": inventory_row.get("method_id", ""), "checkpoint_path": inventory_row.get("checkpoint_path", ""),
                "expected_sha256": expected_sha, "actual_sha256": actual_sha,
                "expected_bytes": expected_bytes, "actual_bytes": actual_bytes,
                "status": "PASS" if not checks and path.is_file() else ";".join(checks) or "MISSING",
            })

    try:
        r7d_metrics = json.loads((root / "experiments/D2-A-R7/R7-D/metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        r7d_metrics = {}
        issue("CRITICAL", "R7D_METRICS_UNREADABLE", "", "metrics.json", str(exc))
    r7d_rows = [row for row in rows if row.get("experiment") == "R7-D"]
    r7d_snr_rows = [row for row in r7d_rows if row.get("record_grain") == "SNR"]
    r7d_diag_rows = [row for row in r7d_rows if row.get("record_grain") == "DIAGNOSTIC"]
    r7d_method_count = len({row.get("method_id", "") for row in r7d_snr_rows})
    r7d_checkpoint_ids = {m.get("method_id") for m in r7d_metrics.get("models", [])}
    r7d_classical_ids = {m.get("method_id") for m in r7d_metrics.get("classical", [])}
    matrix_checkpoint_ids = {row.get("method_id") for row in r7d_snr_rows if row.get("checkpoint_path")}
    matrix_classical_ids = {row.get("method_id") for row in r7d_snr_rows if not row.get("checkpoint_path")}
    r7d_reconciliation = {
        "metrics_checkpoint_count": len(r7d_checkpoint_ids),
        "inventory_checkpoint_rows": len(checkpoint_results),
        "matrix_checkpoint_ids": len(matrix_checkpoint_ids),
        "metrics_classical_count": len(r7d_classical_ids),
        "matrix_classical_ids": len(matrix_classical_ids),
        "matrix_snr_rows": len(r7d_snr_rows), "matrix_diagnostic_rows": len(r7d_diag_rows),
        "matrix_total_rows": len(r7d_rows), "matrix_snr_method_ids": r7d_method_count,
        "missing_checkpoint_ids_in_matrix": sorted(r7d_checkpoint_ids - matrix_checkpoint_ids),
        "unexpected_checkpoint_ids_in_matrix": sorted(matrix_checkpoint_ids - r7d_checkpoint_ids),
        "missing_classical_ids_in_matrix": sorted(r7d_classical_ids - matrix_classical_ids),
        "unexpected_classical_ids_in_matrix": sorted(matrix_classical_ids - r7d_classical_ids),
    }
    for field in ("missing_checkpoint_ids_in_matrix", "unexpected_checkpoint_ids_in_matrix", "missing_classical_ids_in_matrix", "unexpected_classical_ids_in_matrix"):
        if r7d_reconciliation[field]:
            issue("HIGH", "R7D_METHOD_RECONCILIATION", "", field, repr(r7d_reconciliation[field]))

    _write_csv(out_dir / "source_validation.csv", list(source_validation[0]) if source_validation else ["matrix_row"], source_validation)
    _write_csv(out_dir / "record_inventory.csv", inventory_fields + ["record_count"], inventory_rows)
    _write_csv(out_dir / "inconsistencies.csv", ["severity", "code", "row_index", "field", "detail"], findings)
    report = {
        "audit": "D2-A-R7-S/S1 independent matrix audit",
        "method": "Python standard-library read of CSV/JSON/CSV/Markdown sources; no import of the matrix builder",
        "matrix_path": str(matrix_path.relative_to(root)),
        "matrix_file_bytes": matrix_path.stat().st_size,
        "row_count_excluding_header": len(rows), "column_count": len(headers), "columns": headers,
        "exact_duplicate_extra_rows": exact_dup_rows,
        "repeated_identity_group_count": len(repeated_identity_groups),
        "identity_key": list(IDENTITY_KEY),
        "counts": {
            field: dict(Counter(row.get(field, "") for row in rows))
            for field in ("study", "experiment", "dataset", "split", "record_grain", "metric_group", "method_role", "metric_status", "applicability_status")
        },
        "source_path_checks": dict(path_stats),
        "source_pointer_statuses": {k: v for k, v in path_stats.items() if k.startswith(("RESOLVED", "UNRESOLVED", "AMBIGUOUS", "INVALID", "UNSUPPORTED", "SOURCE_FILE", "UNREADABLE"))},
        "r7d_reconciliation": r7d_reconciliation,
        "checkpoint_inventory_count": len(checkpoint_results),
        "checkpoint_hash_match_count": sum(item["status"] == "PASS" for item in checkpoint_results),
        "checkpoint_results": checkpoint_results,
        "numeric_validation": {
            "numeric_columns_checked": sorted(NUMERIC_COLUMNS.intersection(headers)),
            "rate_columns_checked": sorted(RATE_COLUMNS.intersection(headers)),
            "confidence_interval_pairs_checked": [list(pair) for pair in CONFIDENCE_PAIRS],
            "confidence_interval_rule": "check finite ordered endpoints and metric-specific bounds; percentile bootstrap intervals are not required to contain the observed point estimate",
        },
        "finding_count": len(findings),
        "finding_counts_by_severity": dict(Counter(item["severity"] for item in findings)),
        "finding_counts_by_code": dict(Counter(item["code"] for item in findings)),
        "output_files": ["matrix_audit.json", "source_validation.csv", "record_inventory.csv", "inconsistencies.csv"],
    }
    (out_dir / "matrix_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--matrix", type=Path, default=Path("experiments/D2-A-R7/unified_performance_matrix.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/D2-A-R7-S/S1"))
    args = parser.parse_args()
    report = audit_matrix(args.repo_root, args.matrix, args.out_dir)
    print(json.dumps({
        "rows": report["row_count_excluding_header"], "columns": report["column_count"],
        "exact_duplicate_extra_rows": report["exact_duplicate_extra_rows"],
        "findings": report["finding_count"], "finding_counts_by_code": report["finding_counts_by_code"],
        "source_pointer_statuses": report["source_pointer_statuses"],
        "checkpoint_hash_match_count": report["checkpoint_hash_match_count"],
        "r7d_reconciliation": report["r7d_reconciliation"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
