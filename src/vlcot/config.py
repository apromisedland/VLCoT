from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    backbone: str = "openvla/openvla-7b"
    revision: str = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
    local_files_only: bool = True
    action_dim: int = 7
    proprio_dim: int = 8
    predicate_names: list[str] = field(default_factory=list)
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "wrist"])
    vectors_per_stage: int = 4
    max_stages: int = 8
    action_horizon: int = 8
    execution_horizon: int = 4
    observation_window: int = 4
    head_dim: int = 512
    attention_heads: int = 8
    lora_rank: int = 32
    lora_alpha: int = 64
    max_context_tokens: int = 3072
    variant: str = "full"
    initial_frame: bool = False

    def validate(self):
        if self.variant not in {"base", "auxiliary", "no_alignment", "no_repair", "full"}:
            raise ValueError(f"Unknown model variant: {self.variant}")
        if not self.predicate_names or len(set(self.predicate_names)) != len(self.predicate_names):
            raise ValueError("predicate_names must be a nonempty ordered list of unique names")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be unique and nonempty")
        for name in ("action_dim", "proprio_dim", "vectors_per_stage", "max_stages", "action_horizon",
                     "execution_horizon", "observation_window", "head_dim", "attention_heads",
                     "lora_rank", "max_context_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.execution_horizon > self.action_horizon or self.head_dim % self.attention_heads:
            raise ValueError("Invalid execution horizon or attention head dimensions")


@dataclass
class VerificationConfig:
    confidence: float = 0.8
    discrepancy: float = 0.15
    confirmations: int = 3
    contradictions: int = 3
    stagnation_observations: int = 6
    progress_epsilon: float = 0.01

    def validate(self):
        if not 0.5 < self.confidence < 1 or not 0 < self.discrepancy < 1:
            raise ValueError("Invalid verification confidence/discrepancy")
        if min(self.confirmations, self.contradictions, self.stagnation_observations) < 1:
            raise ValueError("Verification persistence counts must be positive")
        if not 0 < self.progress_epsilon < 1:
            raise ValueError("progress_epsilon must lie between zero and one")


@dataclass
class TrainConfig:
    max_steps: int | None = None
    learning_rate: float = 1e-4
    effective_batch_size: int = 64
    micro_batch_size: int = 1
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True
    stage_weight: float = 1.0
    observation_weight: float = 1.0
    auxiliary_weight: float = 1.0
    seed: int = 0
    save_every: int = 1000
    validate_every: int = 1000
    validation_batches: int = 32
    warmup_steps: int = 100
    workers: int = 0
    max_grad_norm: float = 1.0
    recovery_data: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


def checked_dataclass(cls, values: dict[str, Any]):
    unknown = set(values) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(values: dict[str, Any]) -> Config:
    unknown = set(values) - {"model", "verification", "training"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    config = Config(
        checked_dataclass(ModelConfig, values.get("model", {})),
        checked_dataclass(VerificationConfig, values.get("verification", {})),
        checked_dataclass(TrainConfig, values.get("training", {})),
    )
    config.model.validate()
    config.verification.validate()
    return config


def load_config(path: str | Path) -> Config:
    return config_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
