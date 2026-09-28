import shutil
from pathlib import Path

import h5py
import numpy as np

from vlcot.data.io import file_hash, read_json, resolve_path, validate_manifest, write_json


def condition_indices(conditions, names):
    indices = []
    values = []
    for condition in conditions:
        indices.append(names.index(condition["predicate"]))
        value = condition["value"]
        if value not in (0, 1):
            raise ValueError("Predicate conditions must have binary values")
        values.append(value)
    return np.asarray(indices, dtype=int), np.asarray(values, dtype=np.float32)


def build_stages(states, valid, schema, names):
    if states.shape != valid.shape or states.ndim != 2 or states.shape[1] != len(names):
        raise ValueError("Predicate traces must match the manifest vocabulary")
    if np.any(~np.isfinite(states[valid.astype(bool)])):
        raise ValueError("Valid predicate labels must be finite")
    if not np.isin(valid, [0, 1]).all() or not np.isin(states[valid.astype(bool)], [0, 1]).all():
        raise ValueError("Predicate validity and established discrete labels must be binary")
    stable_frames = int(schema.get("stable_frames", 3))
    if stable_frames < 1 or not schema.get("events"):
        raise ValueError("Annotation requires stable_frames >= 1 and explicit semantic event contracts")
    boundaries = []
    for rule_index, event in enumerate(schema["events"]):
        for required_key in ("when", "effects", "requires", "releases"):
            if required_key not in event:
                raise ValueError(f"Event {event.get('name')} is missing {required_key}")
        indices, expected = condition_indices(event["when"], names)
        if not len(indices):
            raise ValueError("Event detection requires at least one predicate")
        holds = np.all(valid[:, indices].astype(bool) & (states[:, indices] == expected), axis=1)
        active = bool(holds[0])
        consecutive = 0
        for frame in range(1, len(states)):
            if not holds[frame]:
                if np.all(valid[frame, indices]):
                    active = False
                consecutive = 0
            else:
                consecutive += 1
                if not active and consecutive >= stable_frames:
                    boundaries.append((frame, rule_index, event))
                    active = True
    boundaries.sort(key=lambda item: (item[0], item[1]))
    grouped = []
    for frame, _, event in boundaries:
        if not grouped or grouped[-1][0] != frame:
            grouped.append((frame, [event]))
        else:
            grouped[-1][1].append(event)
    goals = schema.get("goal_conditions", [])
    if not goals:
        raise ValueError("goal_conditions must identify instruction-specific final subgoals")
    goal_indices, goal_values = condition_indices(goals, names)
    if not np.all(valid[-1, goal_indices]) or not np.all(states[-1, goal_indices] == goal_values):
        raise ValueError("Demonstration does not establish its labeled task goal")
    if not grouped or grouped[-1][0] != len(states) - 1:
        grouped.append((len(states) - 1, [{"effects": goals, "requires": schema.get("terminal_requires", []),
                                         "releases": [], "name": "terminal_goal"}]))
    result = {name: [] for name in ("ends", "expected", "valid", "relevant", "required", "required_values",
                                   "releases", "contract_valid")}
    previous = 0
    for frame, events in grouped:
        relevant = np.zeros(len(names), dtype=np.float32)
        requirements = np.zeros(len(names), dtype=np.float32)
        required_values = np.zeros(len(names), dtype=np.float32)
        releases = np.zeros(len(names), dtype=np.float32)
        for event in events:
            indices, values = condition_indices(event["effects"], names)
            if not np.all(valid[frame, indices]) or not np.all(states[frame, indices] == values):
                raise ValueError("Event effects are not supported by endpoint labels")
            relevant[indices] = 1
            indices, values = condition_indices(event["requires"], names)
            if np.any(requirements[indices].astype(bool) & (required_values[indices] != values)):
                raise ValueError("Simultaneous events have conflicting prerequisites")
            if not np.all(valid[previous, indices]) or not np.all(states[previous, indices] == values):
                raise ValueError("Stage prerequisites are not supported at its boundary")
            requirements[indices] = 1
            required_values[indices] = values
            for name in event["releases"]:
                releases[names.index(name)] = 1
        result["ends"].append(frame)
        result["expected"].append(np.where(valid[frame], states[frame], 0))
        result["valid"].append(valid[frame])
        result["relevant"].append(relevant)
        result["required"].append(requirements)
        result["required_values"].append(required_values)
        result["releases"].append(releases)
        result["contract_valid"].append(np.ones(len(names), dtype=np.float32))
        previous = frame
    return {name: np.asarray(value, dtype=np.int64 if name == "ends" else np.float32) for name, value in result.items()}


def annotate(manifest_path, schema_path, output_directory):
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    schemas = read_json(schema_path)
    if schemas["predicate_names"] != manifest["predicate_names"]:
        raise ValueError("Annotation schema and dataset predicate ordering differ")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Annotation requires a fresh output directory")
    episodes = []
    for index, episode in enumerate(manifest["episodes"]):
        schema = schemas["tasks"].get(episode["task"])
        if schema is None:
            raise ValueError(f"Missing predicate/event schema for task {episode['task']}")
        source = resolve_path(manifest_path, episode["path"])
        destination = output / f"episode_{index:06d}.h5"
        shutil.copyfile(source, destination)
        with h5py.File(destination, "r+") as data:
            if "stages" in data:
                raise ValueError("Input is already annotated; use a prepared trajectory source")
            if "offline/predicates" not in data or "offline/predicate_valid" not in data:
                raise ValueError("Offline predicate traces are required; use replay-labels or import recorded labels")
            states = np.asarray(data["offline/predicates"], dtype=np.float32)
            valid = np.asarray(data["offline/predicate_valid"], dtype=np.float32)
            if len(states) != len(data["actions"]):
                raise ValueError("Predicate traces must align with every prepared observation/action")
            stages = build_stages(states, valid, schema, manifest["predicate_names"])
            for name, value in stages.items():
                data.create_dataset(f"stages/{name}", data=value)
            visible = np.asarray(data["offline/observable"]) if "offline/observable" in data else np.zeros_like(valid)
            visibility_valid = (np.asarray(data["offline/visibility_valid"]) if "offline/visibility_valid" in data
                                else np.zeros_like(valid))
            if visible.shape != valid.shape or visibility_valid.shape != valid.shape:
                raise ValueError("Visibility label shapes differ from state labels")
            if not np.isin(visibility_valid, [0, 1]).all() or not np.isin(visible[visibility_valid.astype(bool)], [0, 1]).all():
                raise ValueError("Known visibility labels and masks must be binary")
            visible = np.where(visibility_valid, visible, 0)
            for name, value in {"state": np.where(valid, states, 0),
                                "state_valid": valid * visible * visibility_valid,
                                "observable": visible, "visibility_valid": visibility_valid}.items():
                data.create_dataset(f"observation_targets/{name}", data=value)
            data.attrs["annotation_sha256"] = file_hash(schema_path)
        episodes.append({**episode, "path": destination.name})
    result = {**manifest, "episodes": episodes, "annotation_sha256": file_hash(schema_path)}
    write_json(output / "manifest.json", result)
    return output / "manifest.json"
