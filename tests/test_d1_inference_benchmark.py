import pytest
from importlib.util import find_spec


def test_latency_summary_reports_batch_percentiles_and_throughput():
    assert find_spec("radardistill.evaluation.inference_benchmark") is not None
    from radardistill.evaluation.inference_benchmark import summarize_latency_samples

    summary = summarize_latency_samples([0.01, 0.02, 0.03], batch_size=2)

    assert summary["repeat_count"] == 3
    assert summary["batch_size"] == 2
    assert summary["mean_latency_ms"] == pytest.approx(20.0)
    assert summary["p50_latency_ms"] == pytest.approx(20.0)
    assert summary["p95_latency_ms"] == pytest.approx(29.0)
    assert summary["mean_throughput_samples_per_second"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("elapsed_seconds", "batch_size"),
    [([], 1), ([0.01], 0), ([0.01, float("nan")], 1), ([0.01, -0.01], 1)],
)
def test_latency_summary_rejects_invalid_measurements(elapsed_seconds, batch_size):
    assert find_spec("radardistill.evaluation.inference_benchmark") is not None
    from radardistill.evaluation.inference_benchmark import summarize_latency_samples

    with pytest.raises(ValueError):
        summarize_latency_samples(elapsed_seconds, batch_size=batch_size)
