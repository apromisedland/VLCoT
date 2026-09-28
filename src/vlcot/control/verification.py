from dataclasses import dataclass
from enum import Enum

import numpy as np

from vlcot.config import VerificationConfig
from vlcot.types import StagePlan, StateEstimate, VerifiedHistory


class Decision(str, Enum):
    CONTINUE = "continue"
    UNCONFIRMED = "unconfirmed"
    COMPLETE = "complete"
    REPAIR = "repair"


@dataclass(frozen=True)
class VerificationResult:
    decision: Decision
    repair_from: int | None = None
    discrepancy: float | None = None
    reason: str = ""


def required_history(plan: StagePlan, history: VerifiedHistory) -> dict[int, np.ndarray]:
    count = plan.stages[0].expected.size
    owners = np.full(count, -1, dtype=int)
    values = np.zeros(count)
    for entry in history.entries:
        owners[plan.stages[entry.stage].releases.astype(bool)] = -1
        mask = entry.relevant.astype(bool)
        owners[mask] = entry.stage
        values[mask] = entry.expected[mask] >= 0.5
    required = {}
    for stage in plan.stages[plan.active:]:
        matching = stage.required.astype(bool) & ((stage.required_values >= 0.5) == values) & (owners >= 0)
        for owner in np.unique(owners[matching]):
            required.setdefault(int(owner), np.zeros(count, dtype=bool))[:] |= matching & (owners == owner)
        overwritten = stage.relevant.astype(bool) | stage.releases.astype(bool)
        owners[overwritten] = -1
    return required


class StageVerifier:
    def __init__(self, config: VerificationConfig):
        config.validate()
        self.config = config
        self.reset()

    def reset(self):
        self.stage = None
        self.last_step = -1
        self.completions = 0
        self.contradictions = {}
        self.best_error = float("inf")
        self.stagnation = 0

    def check(self, plan: StagePlan, history: VerifiedHistory, observed: StateEstimate) -> VerificationResult:
        if plan.active >= len(plan.stages):
            raise ValueError("Extend the planning window before verification")
        if self.stage != plan.active:
            self.reset()
            self.stage = plan.active
        if observed.step <= self.last_step:
            raise ValueError("A verification observation must be newer than the preceding observation")
        self.last_step = observed.step
        specification = plan.stages[plan.active]
        reliable = observed.confidence >= self.config.confidence
        required = required_history(plan, history)
        entries = {entry.stage: entry for entry in history.entries}
        affected = []
        unknown_history = False
        for owner, mask in required.items():
            contradiction = mask & reliable & (np.abs(entries[owner].expected - observed.probabilities) > self.config.discrepancy)
            unknown_history |= bool(np.any(mask & ~reliable))
            self.contradictions[owner] = self.contradictions.get(owner, 0) + 1 if contradiction.any() else 0
            if self.contradictions[owner] >= self.config.contradictions:
                affected.append(owner)
        for owner in set(self.contradictions) - set(required):
            self.contradictions.pop(owner)
        if affected:
            self.completions = 0
            return VerificationResult(Decision.REPAIR, min(affected), reason="required_history_contradicted")
        mask = specification.relevant.astype(bool)
        if not mask.any() or np.any(mask & ~reliable) or unknown_history:
            self.completions = 0
            self.stagnation = 0
            self.best_error = float("inf")
            return VerificationResult(Decision.UNCONFIRMED, reason="insufficient_observation_evidence")
        weights = observed.confidence[mask]
        differences = np.abs(specification.expected[mask] - observed.probabilities[mask])
        error = float(np.sum(weights * differences) / np.sum(weights))
        supported = error <= self.config.discrepancy and np.all(differences <= self.config.discrepancy)
        pending_contradiction = any(self.contradictions.values())
        if supported and not pending_contradiction:
            self.completions += 1
            self.stagnation = 0
            if self.completions >= self.config.confirmations:
                return VerificationResult(Decision.COMPLETE, discrepancy=error, reason="consecutive_support")
            return VerificationResult(Decision.CONTINUE, discrepancy=error, reason="awaiting_consecutive_support")
        self.completions = 0
        if error < self.best_error - self.config.progress_epsilon:
            self.best_error = error
            self.stagnation = 0
        else:
            self.stagnation += 1
        if self.stagnation >= self.config.stagnation_observations:
            return VerificationResult(Decision.REPAIR, plan.active, error, "persistent_stagnation")
        return VerificationResult(Decision.CONTINUE, discrepancy=error, reason="in_progress")
