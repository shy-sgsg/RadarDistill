from scripts.audit_d2a_r7s_matrix import _ci_endpoints_reversed, _status_value, resolve_json_pointer


def test_resolve_json_pointer_handles_list_indices_and_escaped_keys():
    document = {"models/and/results": [{"~value": 17}]}

    assert resolve_json_pointer(document, "/models~1and~1results/0/~0value") == 17


def test_resolve_json_pointer_reports_missing_path():
    document = {"models": []}

    try:
        resolve_json_pointer(document, "/models/2/name")
    except (KeyError, IndexError, ValueError):
        return
    raise AssertionError("missing JSON Pointer path must not resolve")


def test_status_tokens_are_not_treated_as_numeric_values():
    assert _status_value("NOT_APPLICABLE_NO_THRESHOLD")
    assert _status_value("UNVERIFIED_NO_DENSE_SCORE_MAP")
    assert not _status_value("0.012")


def test_percentile_bootstrap_interval_may_exclude_point_estimate_but_endpoints_must_be_ordered():
    # The audit validates interval structure, not a property percentile CIs do not guarantee.
    assert not _ci_endpoints_reversed(0.498032, 0.498548)
    assert _ci_endpoints_reversed(0.51, 0.49)
