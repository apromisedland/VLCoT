import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(manifest_path, relative):
    return (Path(manifest_path).resolve().parent / relative).resolve()


def assign_splits(episodes, seed=0):
    parents = list(range(len(episodes)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, episode in enumerate(episodes):
        for field in ("source_trajectory", "failure_scenario"):
            value = episode.get(field)
            if value:
                key = (field, str(value))
                if key in owners:
                    parents[root(index)] = root(owners[key])
                owners[key] = index
    groups = {}
    for index in range(len(episodes)):
        groups.setdefault(root(index), []).append(index)
    for indices in groups.values():
        explicit = {episodes[index]["split"] for index in indices if "split" in episodes[index]}
        if len(explicit) > 1:
            raise ValueError("Related source trajectories or failure scenarios cross dataset splits")
        identity = "|".join(sorted(episodes[index]["id"] for index in indices))
        fraction = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:16], 16) / 2**64
        split = next(iter(explicit)) if explicit else ("train" if fraction < 0.9 else "validation" if fraction < 0.95 else "test")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        for index in indices:
            episodes[index]["split"] = split
    return episodes


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or not manifest.get("episodes"):
        raise ValueError("Expected a nonempty schema_version=1 trajectory manifest")
    for key in ("predicate_names", "camera_names"):
        values = manifest.get(key, [])
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty unique ordered list")
    identities = [episode["id"] for episode in manifest["episodes"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate trajectory IDs")
    for episode in manifest["episodes"]:
        for key in ("task", "suite", "benchmark", "source_trajectory", "instruction"):
            if not episode.get(key):
                raise ValueError(f"Episode {episode['id']} requires {key}")
        if episode.get("recovery", False) and not episode.get("failure_scenario"):
            raise ValueError("Recovery data requires a failure_scenario grouping identifier")


def decode_image(values, encoding):
    array = np.asarray(values)
    if encoding == "raw_rgb":
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError("raw_rgb images must be HWC uint8 arrays")
        return array
    if encoding not in {"jpeg", "legacy_robotwin_jpeg"}:
        raise ValueError(f"Unknown image encoding: {encoding}")
    payload = bytes(values) if isinstance(values, (bytes, np.bytes_)) else array.astype(np.uint8).tobytes()
    image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    return image[..., ::-1].copy() if encoding == "legacy_robotwin_jpeg" else image


def prepare_data(source_manifest, output_directory, seed=0):
    from vlcot.evaluation.environments import transform_image

    source_manifest = Path(source_manifest).resolve()
    manifest = read_json(source_manifest)
    validate_manifest(manifest)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Preparation requires a fresh output directory")
    episodes = assign_splits([dict(entry) for entry in manifest["episodes"]], seed)
    prepared = []
    for index, episode in enumerate(episodes):
        source = resolve_path(source_manifest, episode["path"])
        mapping = episode.get("mapping", manifest.get("mapping"))
        if not mapping:
            raise ValueError("An explicit HDF5 field mapping is required")
        destination = output / f"episode_{index:06d}.h5"
        with h5py.File(source, "r") as raw, h5py.File(destination, "w") as target:
            group = raw[episode.get("group", "/")]
            actions = np.asarray(group[mapping["actions"]], dtype=np.float32)
            proprio_parts = [np.asarray(group[key], dtype=np.float32) for key in mapping["proprio"]]
            proprio = np.concatenate([part.reshape(len(part), -1) for part in proprio_parts], axis=-1)
            offset = int(mapping.get("action_offset", 0))
            if offset < 0 or actions.ndim != 2:
                raise ValueError("Action offset must be nonnegative and actions must have shape [time, action_dim]")
            lengths = [len(actions) - offset, len(proprio)]
            lengths.extend(len(group[mapping["images"][name]]) for name in manifest["camera_names"])
            length = min(lengths)
            if length < 2 or max(lengths) - length > max(1, offset):
                raise ValueError("Trajectory arrays have incompatible time axes")
            if not np.isfinite(actions).all() or not np.isfinite(proprio).all():
                raise ValueError("Non-finite actions or proprioception")
            target.create_dataset("actions", data=actions[offset:offset + length])
            target.create_dataset("proprio", data=proprio[:length])
            for name in manifest["camera_names"]:
                images = group[mapping["images"][name]]
                encoding = mapping.get("image_encoding", "raw_rgb")
                transform = mapping.get("image_transform", "none")
                first = transform_image(decode_image(images[0], encoding), transform)
                dataset = target.create_dataset(f"images/{name}", shape=(length, *first.shape), dtype="uint8",
                                                chunks=(1, *first.shape), compression="gzip")
                for frame in range(length):
                    dataset[frame] = transform_image(decode_image(images[frame], encoding), transform)
            if ("observable" in mapping) != ("visibility_valid" in mapping):
                raise ValueError("Supply observable and visibility_valid together")
            for key in ("predicates", "predicate_valid", "observable", "visibility_valid", "sim_states"):
                if key in mapping:
                    values = np.asarray(group[mapping[key]])
                    if len(values) < length:
                        raise ValueError(f"Offline field {key} is shorter than its trajectory")
                    target.create_dataset(f"offline/{key}", data=values[:length])
            target.attrs["instruction"] = episode["instruction"]
            target.attrs["source_sha256"] = file_hash(source)
        metadata = {key: value for key, value in episode.items() if key not in {"mapping", "group", "path"}}
        metadata.update(path=destination.name, source_sha256=file_hash(source), length=length)
        prepared.append(metadata)
    result = {"schema_version": 1, "predicate_names": manifest["predicate_names"],
              "camera_names": manifest["camera_names"], "episodes": prepared, "split_seed": seed,
              "source_manifest_sha256": file_hash(source_manifest)}
    write_json(output / "manifest.json", result)
    return output / "manifest.json"


class ActionNormalizer:
    def __init__(self, statistics):
        self.statistics = statistics
        self.low = np.asarray(statistics["low"], dtype=np.float32)
        self.high = np.asarray(statistics["high"], dtype=np.float32)
        if self.low.shape != self.high.shape or self.low.ndim != 1 or not np.isfinite([self.low, self.high]).all():
            raise ValueError("Invalid action normalization statistics")
        if np.any(self.high < self.low):
            raise ValueError("Normalization upper quantile is below lower quantile")
        self.variable = self.high - self.low > 1e-6

    def normalize(self, actions):
        if actions.shape[-1] != len(self.low):
            raise ValueError("Action dimension differs from normalization statistics")
        span = np.where(self.variable, self.high - self.low, 1)
        values = 2 * (actions - self.low) / span - 1
        return np.where(self.variable, np.clip(values, -1, 1), 0).astype(np.float32)

    def denormalize(self, actions):
        values = (np.clip(actions, -1, 1) + 1) / 2 * (self.high - self.low) + self.low
        return np.where(self.variable, values, self.low).astype(np.float32)


def fit_normalizer(manifest_path, include_recovery=False, reservoir_size=200000, seed=0):
    manifest = read_json(manifest_path)
    generator = np.random.default_rng(seed)
    reservoir = None
    seen = 0
    identities = []
    for episode in manifest["episodes"]:
        if episode["split"] != "train" or (episode.get("recovery", False) and not include_recovery):
            continue
        identities.append(episode["id"])
        with h5py.File(resolve_path(manifest_path, episode["path"]), "r") as data:
            for action in data["actions"]:
                if reservoir is None:
                    reservoir = np.empty((reservoir_size, len(action)), dtype=np.float32)
                if seen < reservoir_size:
                    reservoir[seen] = action
                else:
                    candidate = int(generator.integers(seen + 1))
                    if candidate < reservoir_size:
                        reservoir[candidate] = action
                seen += 1
    if not seen:
        raise ValueError("No eligible training actions are available")
    sample = reservoir[:min(seen, reservoir_size)]
    statistics = {"low": np.quantile(sample, 0.01, axis=0).tolist(), "high": np.quantile(sample, 0.99, axis=0).tolist(),
                  "source_split": "train", "episode_ids": identities, "seed": seed,
                  "observations_seen": seen, "reservoir_size": min(seen, reservoir_size),
                  "manifest_sha256": file_hash(manifest_path)}
    return ActionNormalizer(statistics)
