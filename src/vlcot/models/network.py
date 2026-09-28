from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from vlcot.config import ModelConfig
from vlcot.control.budget import BudgetExhausted, BudgetLedger
from vlcot.types import Observation, StageSpec


@dataclass
class EncodedObservation:
    tokens: torch.Tensor
    step: int


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, values):
        return values + self.layers(values)


class StageDecoder(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, predicates: int):
        super().__init__()
        self.features = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, head_dim), nn.GELU())
        self.heads = nn.ModuleDict({
            name: nn.Linear(head_dim, predicates)
            for name in ("expected", "relevant", "required", "required_values", "releases")
        })
        self.stop = nn.Linear(head_dim, 1)

    def forward(self, latents: torch.Tensor):
        features = self.features(latents.mean(dim=-2))
        result = {name: head(features) for name, head in self.heads.items()}
        result["stop"] = self.stop(features).squeeze(-1)
        return result


class ObservationEstimator(nn.Module):
    def __init__(self, hidden_size: int, config: ModelConfig):
        super().__init__()
        width = config.head_dim
        self.projection = nn.Linear(hidden_size, width)
        self.temporal = nn.Embedding(config.observation_window, width)
        self.queries = nn.Parameter(torch.randn(1, len(config.predicate_names), width) * 0.02)
        self.attention = nn.MultiheadAttention(width, config.attention_heads, batch_first=True)
        self.features = ResidualBlock(width)
        self.state = nn.Linear(width, 1)
        self.visibility = nn.Linear(width, 1)

    def forward(self, observations: list[EncodedObservation]):
        memory = []
        for index, observation in enumerate(observations):
            values = self.projection(observation.tokens)
            memory.append(values + self.temporal.weight[index].view(1, 1, -1))
        context = torch.cat(memory, dim=1)
        features, _ = self.attention(self.queries, context, context, need_weights=False)
        features = self.features(features + self.queries)
        return {"state": self.state(features).squeeze(0).squeeze(-1),
                "observable": self.visibility(features).squeeze(0).squeeze(-1)}


class ActionHead(nn.Module):
    def __init__(self, hidden_size: int, config: ModelConfig):
        super().__init__()
        width = config.head_dim
        self.projection = nn.Linear(hidden_size, width)
        self.queries = nn.Parameter(torch.randn(1, config.action_horizon, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, config.attention_heads, batch_first=True)
        self.regression = nn.Sequential(ResidualBlock(width), ResidualBlock(width), nn.Linear(width, config.action_dim))

    def forward(self, context: torch.Tensor, latents: torch.Tensor):
        memory = torch.cat((context, latents.reshape(1, -1, context.shape[-1])), dim=1)
        memory = self.projection(memory)
        features, _ = self.attention(self.queries, memory, memory, need_weights=False)
        return torch.tanh(self.regression(features + self.queries)).squeeze(0)


class VLCoTModel(nn.Module):
    def __init__(self, backbone: nn.Module, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = backbone
        width = backbone.hidden_size
        predicates = len(config.predicate_names)
        self.proprio_projection = nn.Linear(config.proprio_dim, width)
        self.history_projection = nn.Linear(2 * predicates, width)
        self.view_embeddings = nn.Embedding(len(config.camera_names), width)
        self.latent_start = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.latent_feedback = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.slot_embeddings = nn.Embedding(config.vectors_per_stage, width)
        self.stage_decoder = StageDecoder(width, config.head_dim, predicates)
        self.observation_estimator = ObservationEstimator(width, config)
        self.action_head = ActionHead(width, config)
        self.auxiliary_state = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, predicates))

    @property
    def device(self):
        return self.latent_start.device

    @property
    def dtype(self):
        return self.latent_start.dtype

    def tensor(self, values):
        return torch.as_tensor(values, device=self.device, dtype=self.dtype)

    def encode(self, observation: Observation) -> EncodedObservation:
        observation.validate(self.config.camera_names, self.config.proprio_dim)
        vision = []
        for index, name in enumerate(self.config.camera_names):
            tokens = self.backbone.visual_embeddings(observation.images[name])
            vision.append(tokens + self.view_embeddings.weight[index].view(1, 1, -1))
        proprio = self.proprio_projection(self.tensor(observation.proprio)).view(1, 1, -1)
        tokens = torch.cat((self.backbone.bos_embedding(), *vision,
                            self.backbone.text_embeddings(observation.instruction), proprio), dim=1)
        if tokens.shape[1] > self.config.max_context_tokens:
            raise ValueError("Observation exceeds the configured context length")
        return EncodedObservation(tokens, observation.step)

    def context(self, encoded: EncodedObservation, history, initial: EncodedObservation | None = None):
        summary = self.history_projection(self.tensor(history)).view(1, 1, -1)
        pieces = [encoded.tokens, summary]
        if self.config.initial_frame:
            if initial is None:
                raise ValueError("Initial-frame augmentation requires the episode's actual initial observation")
            if initial.step != encoded.step:
                pieces.append(initial.tokens)
        return torch.cat(pieces, dim=1)

    def empty_latents(self):
        return self.latent_start.new_empty((0, self.config.vectors_per_stage, self.backbone.hidden_size))

    def generate_stages(self, context, prefix=None, stage_limit=None, exact=False,
                        ledger: BudgetLedger | None = None, operation: str = "initial"):
        prefix = self.empty_latents() if prefix is None else prefix
        limit = self.config.max_stages if stage_limit is None else stage_limit
        if limit < 1:
            return self.empty_latents(), {}
        fixed = None
        feedback = [self.latent_start]
        segments = []
        decoded_segments = []
        for stage_index in range(limit):
            vector_count = self.config.vectors_per_stage
            if ledger is not None and not ledger.permits(operation, vector_count):
                if not segments:
                    raise BudgetExhausted(f"Cannot generate the next {operation} stage")
                break
            charge = ledger.charge(operation, vector_count) if ledger is not None else nullcontext()
            with charge:
                if fixed is None:
                    prefix_tokens = self.latent_feedback(prefix.reshape(1, -1, self.backbone.hidden_size))
                    fixed = torch.cat((context, prefix_tokens), dim=1)
                vectors = []
                for slot in range(vector_count):
                    current = feedback[-1] + self.slot_embeddings.weight[slot].view(1, 1, -1)
                    inputs = torch.cat((fixed, *feedback[:-1], current), dim=1)
                    if inputs.shape[1] > self.config.max_context_tokens:
                        raise ValueError("Latent context exceeds max_context_tokens; no history was silently truncated")
                    latent = self.backbone.hidden(inputs)[:, -1:, :]
                    vectors.append(latent.squeeze(0).squeeze(0))
                    feedback[-1] = current
                    feedback.append(self.latent_feedback(latent))
                segment = torch.stack(vectors)
                segments.append(segment)
                semantic_input = segment.detach() if self.config.variant == "no_alignment" else segment
                decoded = self.stage_decoder(semantic_input)
                decoded_segments.append(decoded)
            if not exact and decoded["stop"].sigmoid().item() >= 0.5:
                break
        return torch.stack(segments), {
            name: torch.stack([segment[name] for segment in decoded_segments]) for name in decoded_segments[0]
        }

    def stage_specs(self, decoded: dict[str, torch.Tensor]) -> list[StageSpec]:
        probabilities = {name: values.detach().float().sigmoid().cpu().numpy() for name, values in decoded.items()}
        return [StageSpec(
            probabilities["expected"][index],
            probabilities["relevant"][index] >= 0.5,
            probabilities["required"][index] >= 0.5,
            probabilities["required_values"][index],
            probabilities["releases"][index] >= 0.5,
        ) for index in range(len(probabilities["expected"]))]

    def estimate(self, observations):
        if not observations or len(observations) > self.config.observation_window:
            raise ValueError("Invalid observation window length")
        return self.observation_estimator(observations)

    def actions(self, context, latents):
        return self.action_head(context, latents)

    def contextual_features(self, context):
        if context.shape[1] > self.config.max_context_tokens:
            raise ValueError("Context exceeds max_context_tokens")
        return self.backbone.hidden(context)
