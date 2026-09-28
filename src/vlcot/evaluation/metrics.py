from collections import defaultdict

import numpy as np


RECOVERY_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_100")


def verification_f1(predictions, labels):
    predicted = np.asarray(predictions, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    if predicted.shape != truth.shape or not predicted.size:
        raise ValueError("F1 requires nonempty, paired predictions and independent labels")
    true_positive = int(np.sum(predicted & truth))
    false_positive = int(np.sum(predicted & ~truth))
    false_negative = int(np.sum(~predicted & truth))
    denominator = 2 * true_positive + false_positive + false_negative
    return {"f1": 2 * true_positive / denominator if denominator else None,
            "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
            "examples": int(predicted.size)}


def recovery_metrics(records):
    cells = defaultdict(list)
    for record in records:
        level = record.get("recovery_level")
        if level is None:
            continue
        if record["suite"] not in RECOVERY_SUITES or level not in {1, 2, 3, 4}:
            raise ValueError("Recovery protocol requires Spatial/Object/Goal/LIBERO-100 and levels 1 through 4")
        cells[(record["suite"], level)].append(int(record["success"]))
    if not cells:
        return None
    by_level = {}
    for level in range(1, 5):
        complete = all(cells[(suite, level)] for suite in RECOVERY_SUITES)
        by_level[str(level)] = float(np.mean([np.mean(cells[(suite, level)]) for suite in RECOVERY_SUITES])) if complete else None
    suite_scores = {}
    for suite in RECOVERY_SUITES:
        outcomes = [outcome for level in range(1, 5) for outcome in cells[(suite, level)]]
        suite_scores[suite] = float(np.mean(outcomes)) if outcomes else None
    overall = (float(np.mean(list(suite_scores.values())))
               if all(score is not None for score in suite_scores.values()) else None)
    return {"by_level_macro": by_level, "overall_suite_macro": overall, "suite_rates": suite_scores,
            "counts": {f"{suite}/L{level}": len(cells[(suite, level)]) for suite in RECOVERY_SUITES for level in range(1, 5)},
            "complete_level_coverage": all(value is not None for value in by_level.values())}


def summarize_group(records):
    if not records:
        raise ValueError("Cannot summarize an empty run")
    identities = [record["episode_id"] for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate episode records would bias reported denominators")
    invalid = [record["episode_id"] for record in records if record["termination"] in {"error", "unfinished"}]
    task_cells = defaultdict(list)
    for record in records:
        condition = record["metadata"].get("condition", "default")
        task_cells[(condition, record["suite"], record["task"])].append(int(record["success"]))
    suite_tasks = defaultdict(list)
    for (condition, suite, task), outcomes in task_cells.items():
        suite_tasks[(condition, suite)].append(float(np.mean(outcomes)))
    success = {f"{condition}/{suite}": float(np.mean(values)) for (condition, suite), values in suite_tasks.items()}
    latency = [value for record in records for value in record["decision_latencies_ms"]]
    libero_suites = [f"default/{suite}" for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")]
    overall_libero = float(np.mean([success[suite] for suite in libero_suites])) if all(suite in success for suite in libero_suites) else None
    return {
        "episodes": len(records), "valid_run": not invalid, "invalid_episodes": invalid,
        "task_success_by_suite": success if not invalid else None,
        "libero_four_suite_average": overall_libero if not invalid else None,
        "recovery": recovery_metrics(records) if not invalid else None,
        "mean_inference_seconds": float(np.mean([record["inference_seconds"] for record in records])),
        "p95_decision_latency_ms": float(np.percentile(latency, 95)) if latency else None,
        "mean_regenerated_vectors": float(np.mean([record["regenerated_vectors"] for record in records])),
        "termination_counts": {reason: sum(record["termination"] == reason for record in records)
                               for reason in sorted({record["termination"] for record in records})},
    }


def summarize(records):
    dimensions = ("checkpoint_sha256", "strategy", "model_seed", "protocol_sha256", "cost_profile_sha256",
                  "budget_limit", "precision", "device", "hardware", "warmup_passes")
    groups = defaultdict(list)
    for record in records:
        metadata = record["metadata"]
        key = (record["benchmark"], *(metadata.get(name) for name in dimensions))
        groups[key].append(record)
    return {"units": {"success": "fraction", "inference_time": "seconds", "decision_latency": "milliseconds"},
            "groups": [{"benchmark": key[0], **dict(zip(dimensions, key[1:])), **summarize_group(values)}
                       for key, values in groups.items()]}
