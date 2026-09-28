import h5py
import numpy as np

from vlcot.config import ModelConfig
from vlcot.data.io import assign_splits, read_json, resolve_path, validate_manifest
from vlcot.types import Observation


class TrajectoryDataset:
    def __init__(self, manifest_path, config: ModelConfig, normalizer, split="train", include_recovery=False, seed=0):
        self.manifest_path = manifest_path
        self.config = config
        self.normalizer = normalizer
        self.seed = seed
        self.epoch = 0
        manifest = read_json(manifest_path)
        validate_manifest(manifest)
        if any("split" not in entry for entry in manifest["episodes"]):
            raise ValueError("Prepared manifests must explicitly identify every trajectory split")
        assign_splits([dict(entry) for entry in manifest["episodes"]])
        if config.predicate_names != manifest["predicate_names"] or config.camera_names != manifest["camera_names"]:
            raise ValueError("Model vocabulary/cameras must exactly match the dataset")
        self.episodes = [entry for entry in manifest["episodes"] if entry["split"] == split
                         and (include_recovery or not entry.get("recovery", False))]
        if not self.episodes:
            raise ValueError(f"No eligible episodes in split {split}")
        benchmarks = {entry["benchmark"] for entry in self.episodes}
        if len(benchmarks) != 1:
            raise ValueError("Fine-tune each benchmark separately")
        self.split = split

    def __len__(self):
        return len(self.episodes)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        episode = self.episodes[index]
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        with h5py.File(resolve_path(self.manifest_path, episode["path"]), "r") as data:
            if "stages/ends" not in data:
                raise ValueError("Dataset must be annotated before training")
            ends = np.asarray(data["stages/ends"])
            length = len(data["actions"])
            if len(ends) < 1 or np.any(np.diff(ends) <= 0) or ends[-1] >= length:
                raise ValueError("Invalid stage boundaries")
            if generator.random() < 0.5:
                stage = int(generator.integers(len(ends)))
                frame = 0 if stage == 0 else int(ends[stage - 1])
            else:
                frame = int(generator.integers(length))
                stage = min(int(np.searchsorted(ends, frame, side="left")), len(ends) - 1)
            stop = min(stage + self.config.max_stages, len(ends))

            def observation(position):
                return Observation({name: np.asarray(data[f"images/{name}"][position]) for name in self.config.camera_names},
                                   np.asarray(data["proprio"][position]), episode["instruction"], int(position))

            stride = self.config.execution_horizon
            positions = list(range(frame, max(-1, frame - stride * self.config.observation_window), -stride))[::-1]
            recent = [observation(position) for position in positions]
            def history_at(boundary):
                summary = np.zeros(2 * len(self.config.predicate_names), dtype=np.float32)
                for prior in range(boundary):
                    released = np.asarray(data["stages/releases"][prior]).astype(bool)
                    summary[len(released):][released] = 0
                    mask = np.asarray(data["stages/relevant"][prior]).astype(bool)
                    summary[:len(mask)][mask] = data["stages/expected"][prior][mask]
                    summary[len(mask):][mask] = 1
                return summary

            history = history_at(stage)
            prefix_windows = []
            for boundary in range(0, stage, self.config.max_stages):
                planning_frame = 0 if boundary == 0 else int(ends[boundary - 1])
                prefix_windows.append({"observation": observation(planning_frame), "history": history_at(boundary),
                                       "stage_count": min(self.config.max_stages, stage - boundary)})
            targets = {name: np.asarray(data[f"stages/{name}"][:stop]) for name in
                       ("expected", "valid", "relevant", "required", "required_values", "releases", "contract_valid")}
            targets["stop"] = np.zeros(stop, dtype=np.float32)
            targets["stop"][-1] = 1
            targets["stop_valid"] = np.ones(stop, dtype=np.float32)
            if stage:
                targets["stop_valid"][:stage] = 0
            actions = np.zeros((self.config.action_horizon, self.config.action_dim), dtype=np.float32)
            action_mask = np.zeros_like(actions)
            available = min(self.config.action_horizon, length - frame)
            actions[:available] = self.normalizer.normalize(np.asarray(data["actions"][frame:frame + available]))
            action_mask[:available] = 1
            observation_targets = {name: np.asarray(data[f"observation_targets/{name}"][frame]) for name in
                                   ("state", "state_valid", "observable", "visibility_valid")}
            if data["proprio"].shape[-1] != self.config.proprio_dim:
                raise ValueError("Proprioception dimensions differ from model configuration")
            return {"episode_id": episode["id"], "initial": observation(0), "recent": recent,
                    "history": history, "prefix_count": stage, "suffix_count": stop - stage,
                    "prefix_windows": prefix_windows,
                    "stage_targets": targets, "actions": actions, "action_mask": action_mask,
                    "observation_targets": observation_targets}


def collate_samples(samples):
    return samples
