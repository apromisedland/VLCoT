from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Observation:
    images: dict[str, np.ndarray]
    proprio: np.ndarray
    instruction: str
    step: int

    def validate(self, cameras: list[str], proprio_dim: int):
        if set(self.images) != set(cameras):
            raise ValueError(f"Expected cameras {cameras}, received {list(self.images)}")
        if self.proprio.shape != (proprio_dim,) or not np.isfinite(self.proprio).all():
            raise ValueError("Invalid proprioception shape or non-finite value")
        for image in self.images.values():
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Observation images must be HWC uint8 RGB")
        if not self.instruction.strip() or self.step < 0:
            raise ValueError("Observation requires an instruction and nonnegative step")


@dataclass(frozen=True)
class StageSpec:
    expected: np.ndarray
    relevant: np.ndarray
    required: np.ndarray
    required_values: np.ndarray
    releases: np.ndarray

    def __post_init__(self):
        arrays = (self.expected, self.relevant, self.required, self.required_values, self.releases)
        if self.expected.ndim != 1 or any(value.shape != self.expected.shape for value in arrays):
            raise ValueError("Stage predicate arrays must have identical one-dimensional shapes")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("Non-finite stage semantics")
        if np.any((self.expected < 0) | (self.expected > 1)):
            raise ValueError("Expected states must be probabilities")


@dataclass
class StagePlan:
    latents: Any
    stages: list[StageSpec]
    active: int = 0

    def __post_init__(self):
        if len(self.latents) != len(self.stages) or not 0 <= self.active <= len(self.stages):
            raise ValueError("Plan latent count, stage count, or active index is inconsistent")


@dataclass(frozen=True)
class StateEstimate:
    probabilities: np.ndarray
    observable: np.ndarray
    step: int

    def __post_init__(self):
        if self.probabilities.ndim != 1 or self.observable.shape != self.probabilities.shape:
            raise ValueError("State probability and observability shapes differ")
        for values in (self.probabilities, self.observable):
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("State probabilities must be finite and in [0, 1]")

    @property
    def confidence(self):
        certainty = np.maximum(self.probabilities, 1 - self.probabilities)
        return np.minimum(certainty, self.observable)


@dataclass
class HistoryEntry:
    stage: int
    expected: np.ndarray
    relevant: np.ndarray
    confirmed_step: int
    releases: np.ndarray | None = None


@dataclass
class VerifiedHistory:
    entries: list[HistoryEntry] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)

    def confirm(self, stage: int, specification: StageSpec, step: int):
        self.invalidate_from(stage)
        self.entries.append(HistoryEntry(stage, specification.expected.copy(), specification.relevant.copy(), step,
                                         specification.releases.copy()))

    def invalidate_from(self, stage: int):
        self.entries = [entry for entry in self.entries if entry.stage < stage]

    def summary(self, predicate_count: int) -> np.ndarray:
        values = np.zeros(predicate_count, dtype=np.float32)
        valid = np.zeros(predicate_count, dtype=np.float32)
        for entry in self.entries:
            if entry.releases is not None:
                valid[entry.releases.astype(bool)] = 0
            mask = entry.relevant.astype(bool)
            values[mask] = entry.expected[mask]
            valid[mask] = 1
        return np.concatenate((values, valid))


@dataclass
class EpisodeRecord:
    episode_id: str
    benchmark: str
    suite: str
    task: str
    seed: int
    recovery_level: int | None = None
    source_trajectory: str | None = None
    success: bool = False
    termination: str = "unfinished"
    environment_steps: int = 0
    inference_seconds: float = 0.0
    decision_latencies_ms: list[float] = field(default_factory=list)
    regenerated_vectors: int = 0
    costs: dict[str, dict[str, float]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
