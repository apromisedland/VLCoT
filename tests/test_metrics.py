import pytest

from vlcot.evaluation.metrics import RECOVERY_SUITES, recovery_metrics, summarize_group, verification_f1
from vlcot.types import EpisodeRecord


def test_recovery_levels_cannot_be_averaged_to_obtain_overall():
    records = []
    for suite in RECOVERY_SUITES:
        for level, count in [(1, 1), (2, 3), (3, 1), (4, 1)]:
            for trial in range(count):
                record = EpisodeRecord(f"{suite}/{level}/{trial}", "libero_recover", suite, "task", 0,
                                       recovery_level=level, success=level == 1,
                                       termination="success" if level == 1 else "budget_exhausted")
                records.append(record.to_dict())
    result = recovery_metrics(records)
    assert sum(result["by_level_macro"].values()) / 4 == 0.25
    assert result["overall_suite_macro"] == pytest.approx(1 / 6)


def test_missing_suite_is_not_silently_treated_as_zero():
    record = EpisodeRecord("one", "libero_recover", "libero_spatial", "task", 0,
                           recovery_level=1, success=True, termination="success")
    result = recovery_metrics([record.to_dict()])
    assert result["overall_suite_macro"] is None
    assert result["by_level_macro"]["1"] is None


def test_successful_and_failed_episode_times_both_count():
    success = EpisodeRecord("one", "libero", "libero_10", "task", 0, success=True,
                            termination="success", inference_seconds=2, decision_latencies_ms=[10, 20])
    failure = EpisodeRecord("two", "libero", "libero_10", "task", 0,
                            termination="budget_exhausted", inference_seconds=6, decision_latencies_ms=[40])
    result = summarize_group([success.to_dict(), failure.to_dict()])
    assert result["mean_inference_seconds"] == 4
    assert result["task_success_by_suite"]["default/libero_10"] == 0.5
    assert result["p95_decision_latency_ms"] == pytest.approx(38)


def test_undefined_diagnostic_is_not_reported_as_perfect():
    assert verification_f1([False, False], [False, False])["f1"] is None
    assert verification_f1([True, True, False], [True, False, True])["f1"] == 0.5


def test_infrastructure_error_invalidates_scores():
    record = EpisodeRecord("one", "libero", "libero_10", "task", 0, termination="error")
    result = summarize_group([record.to_dict()])
    assert not result["valid_run"]
    assert result["task_success_by_suite"] is None
