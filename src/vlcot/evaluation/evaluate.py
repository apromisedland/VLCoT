import json
from pathlib import Path

from vlcot.control.budget import BudgetLedger
from vlcot.control.runner import run_episode
from vlcot.data.io import file_hash, read_json, resolve_path, write_json
from vlcot.evaluation.environments import LiberoEnvironment, RoboTwinEnvironment
from vlcot.evaluation.metrics import RECOVERY_SUITES, summarize
from vlcot.types import EpisodeRecord


def load_protocol(manifest_path):
    protocol = read_json(manifest_path)
    if protocol.get("schema_version") != 1 or not protocol.get("episodes"):
        raise ValueError("A nonempty versioned evaluation manifest is required")
    if not protocol.get("provenance", {}).get("source"):
        raise ValueError("Evaluation protocol provenance.source is required")
    benchmark = protocol["benchmark"]
    if benchmark not in {"libero", "libero_recover", "robotwin50", "robotwin10"}:
        raise ValueError("Unsupported benchmark")
    official = None
    if benchmark in {"libero_recover", "robotwin10"}:
        if not protocol["provenance"].get("official_manifest") or not protocol["provenance"].get("revision"):
            raise ValueError("This protocol requires the official instance/task manifest and its revision")
        source_file = resolve_path(manifest_path, protocol["provenance"]["official_manifest"])
        expected_hash = protocol["provenance"].get("sha256")
        if not expected_hash or file_hash(source_file) != expected_hash:
            raise ValueError("Official protocol manifest checksum is missing or does not match")
        official = read_json(source_file)
    identities = set()
    for episode in protocol["episodes"]:
        if episode["id"] in identities:
            raise ValueError("Repeated evaluation episode ID")
        identities.add(episode["id"])
        if int(episode["max_steps"]) < 1:
            raise ValueError("Every evaluation instance requires a positive max_steps")
        for key in ("suite", "task", "seed"):
            if key not in episode:
                raise ValueError(f"Evaluation episode is missing {key}")
        if benchmark == "libero_recover":
            if episode["suite"] not in RECOVERY_SUITES or episode.get("recovery_level") not in {1, 2, 3, 4}:
                raise ValueError("Recovery labels must use the four official suites and levels")
            for key in ("source_trajectory", "failure_scenario", "instruction"):
                if not episode.get(key):
                    raise ValueError(f"Recovery instance requires {key}")
            for key in ("bddl_file", "state_file"):
                episode[key] = str(resolve_path(manifest_path, episode[key]))
                if not Path(episode[key]).is_file():
                    raise FileNotFoundError(episode[key])
        elif benchmark.startswith("robotwin"):
            if not episode.get("instruction") or episode.get("condition") not in {"clean", "randomized"}:
                raise ValueError("RoboTwin episodes require instruction and clean/randomized condition")
    if benchmark.startswith("robotwin"):
        expected_tasks = 10 if benchmark == "robotwin10" else 50
        task_names = {episode["task"] for episode in protocol["episodes"]}
        if len(task_names) != expected_tasks and not protocol.get("subset_run", False):
            raise ValueError(f"A complete {benchmark} protocol requires {expected_tasks} distinct tasks")
        if benchmark == "robotwin10":
            official_tasks = set(official.get("task_names", []))
            if len(official_tasks) != 10 or not task_names.issubset(official_tasks):
                raise ValueError("Evaluation tasks do not match the pinned official ten-task list")
        if not protocol.get("subset_run", False):
            for task in task_names:
                for condition in ("clean", "randomized"):
                    matching = [episode for episode in protocol["episodes"]
                                if episode["task"] == task and episode["condition"] == condition]
                    if not matching or (benchmark == "robotwin50" and len(matching) != 100):
                        raise ValueError("Full RoboTwin protocols require both conditions; robotwin50 requires 100 trials per task/condition")
    if benchmark == "libero" and not protocol.get("subset_run", False):
        for suite in {episode["suite"] for episode in protocol["episodes"]}:
            if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
                raise ValueError("Ordinary LIBERO evaluation uses Spatial/Object/Goal/Long")
            task_indices = {int(episode["task_id"]) for episode in protocol["episodes"] if episode["suite"] == suite}
            if task_indices != set(range(10)):
                raise ValueError("A full LIBERO suite must cover all ten tasks; otherwise mark subset_run")
    if benchmark == "libero_recover":
        official_instances = {entry["id"]: entry for entry in official.get("instances", [])}
        if not official_instances or not identities.issubset(official_instances):
            raise ValueError("Recovery episode IDs do not belong to the pinned official instance manifest")
        if not protocol.get("subset_run", False) and identities != set(official_instances):
            raise ValueError("Full recovery evaluation must include every predetermined official failure instance")
        for episode in protocol["episodes"]:
            expected = official_instances[episode["id"]]
            for key in ("suite", "task", "recovery_level", "source_trajectory", "failure_scenario"):
                if episode[key] != expected[key]:
                    raise ValueError(f"Recovery instance {episode['id']} differs from official metadata: {key}")
            for filename, checksum in (("state_file", "state_sha256"), ("bddl_file", "bddl_sha256")):
                if file_hash(episode[filename]) != expected[checksum]:
                    raise ValueError(f"Recovery {filename} differs from its official snapshot")
    return protocol


def evaluate(checkpoint, manifest_path, benchmark_root, output_directory, budget_limit, cost_profile,
             strategy="local", device="cuda", backbone_path=None):
    import torch
    from vlcot.checkpoint import load_policy_model
    from vlcot.control.policy import VLCoTPolicy

    protocol = load_protocol(manifest_path)
    costs = read_json(cost_profile)["unit_costs"]
    BudgetLedger(budget_limit, costs)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Evaluation requires a fresh output directory; no episodes are silently skipped")
    model, config, normalizer = load_policy_model(checkpoint, device, backbone_path)
    policy = VLCoTPolicy(model, config.verification, normalizer, strategy)
    metadata = {"checkpoint_sha256": file_hash(Path(checkpoint) / "trainable.pt"),
                "protocol_sha256": file_hash(manifest_path), "cost_profile_sha256": file_hash(cost_profile),
                "strategy": policy.strategy, "model_seed": config.training.seed,
                "backbone": config.model.backbone, "backbone_revision": config.model.revision,
                "precision": str(model.dtype), "device": device, "subset_run": protocol.get("subset_run", False),
                "budget_limit": budget_limit, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "hardware": torch.cuda.get_device_name(model.device) if device.startswith("cuda") else "cpu",
                "warmup_passes": 0}
    write_json(output / "run.json", {**metadata, "config": config.to_dict(), "protocol": protocol,
                                     "budget_limit": budget_limit, "unit_costs": costs})
    records = []
    synchronize = (lambda: torch.cuda.synchronize(model.device)) if device.startswith("cuda") else (lambda: None)
    for episode in protocol["episodes"]:
        ledger = BudgetLedger(budget_limit, costs, synchronize=synchronize)
        record = EpisodeRecord(episode["id"], protocol["benchmark"], episode["suite"], episode["task"],
                               int(episode["seed"]), episode.get("recovery_level"), episode.get("source_trajectory"),
                               metadata={**metadata, "condition": episode.get("condition", "default"),
                                         "failure_scenario": episode.get("failure_scenario")})
        try:
            environment_type = RoboTwinEnvironment if protocol["benchmark"].startswith("robotwin") else LiberoEnvironment
            environment = environment_type(episode, protocol.get("environment", {}), benchmark_root)
            record = run_episode(environment, policy, ledger, record, int(episode["max_steps"]))
        except Exception as error:
            record.termination = "error"
            record.events.append({"event": "setup_error", "detail": str(error), "type": type(error).__name__})
        records.append(record.to_dict())
        with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record.to_dict(), allow_nan=False) + "\n")
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    if any(record["termination"] == "error" for record in records):
        raise RuntimeError("Evaluation encountered infrastructure/model errors; all instances were recorded and scores invalidated")
    return summary
