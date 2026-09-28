from collections import deque

import numpy as np
import torch

from vlcot.control.budget import BudgetExhausted
from vlcot.control.verification import Decision, StageVerifier
from vlcot.types import StagePlan, StateEstimate, VerifiedHistory


class VLCoTPolicy:
    def __init__(self, model, verification, normalizer, strategy="local"):
        if strategy not in {"disabled", "full", "current", "local"}:
            raise ValueError("Unknown repair strategy")
        self.model = model
        self.config = model.config
        self.normalizer = normalizer
        self.strategy = "disabled" if self.config.variant == "no_repair" else strategy
        self.verifier = StageVerifier(verification)
        self.reset(None)

    def reset(self, ledger):
        self.ledger = ledger
        self.history = VerifiedHistory()
        self.stage_plan = None
        self.encoded = deque(maxlen=self.config.observation_window)
        self.initial = None
        self.regenerated_vectors = 0
        self.events = []
        self.verifier.reset()

    def _observe(self, observation):
        if self.ledger is None:
            raise ValueError("reset requires an episode budget before inference")
        if self.encoded and observation.step == self.encoded[-1].step:
            return self.encoded[-1]
        if self.encoded and observation.step < self.encoded[-1].step:
            raise ValueError("Observation time moved backwards without reset")
        with self.ledger.charge("encode", len(self.config.camera_names)):
            encoded = self.model.encode(observation)
        self.encoded.append(encoded)
        if self.initial is None:
            self.initial = encoded
        return encoded

    def _context(self, encoded):
        summary = self.history.summary(len(self.config.predicate_names))
        return self.model.context(encoded, summary, self.initial)

    def _generate(self, encoded, prefix, operation):
        with self.ledger.charge(operation):
            context = self._context(encoded)
        reserve = self.ledger.unit_costs["action"] + self.ledger.unit_costs["verify"]
        cost_per_stage = self.ledger.unit_costs[operation] * self.config.vectors_per_stage
        available = self.ledger.limit - self.ledger.spent - reserve
        limit = min(self.config.max_stages, int((available + 1e-9) // cost_per_stage))
        if limit < 1:
            raise BudgetExhausted("No budget for a stage plus its observation/action calls")
        return self.model.generate_stages(context, prefix=prefix, stage_limit=limit,
                                          ledger=self.ledger, operation=operation)

    @torch.inference_mode()
    def plan(self, observation):
        encoded = self._observe(observation)
        latents, decoded = self._generate(encoded, self.model.empty_latents(), "initial")
        self.stage_plan = StagePlan(latents, self.model.stage_specs(decoded))
        self.events.append({"step": observation.step, "event": "initial_plan", "stages": len(latents)})
        return self.stage_plan

    @torch.inference_mode()
    def estimate_state(self, observation):
        self._observe(observation)
        with self.ledger.charge("verify"):
            estimate = self.model.estimate(list(self.encoded))
            probabilities = estimate["state"].sigmoid().float().cpu().numpy()
            observable = estimate["observable"].sigmoid().float().cpu().numpy()
        return StateEstimate(probabilities, observable, observation.step)

    @torch.inference_mode()
    def repair(self, observation, start):
        if self.stage_plan is None or not 0 <= start <= self.stage_plan.active:
            raise ValueError("Repair must start at or before the active stage")
        if self.strategy == "disabled":
            return self.stage_plan
        self.history.invalidate_from(start)
        if self.strategy == "current":
            start = self.stage_plan.active
        retained = self.stage_plan.latents[:start]
        conditioning = self.model.empty_latents() if self.strategy == "full" else retained
        suffix, decoded = self._generate(self._observe(observation), conditioning, "repair")
        self.stage_plan = StagePlan(torch.cat((retained, suffix)),
                                    self.stage_plan.stages[:start] + self.model.stage_specs(decoded), start)
        self.regenerated_vectors += len(suffix) * self.config.vectors_per_stage
        self.verifier.reset()
        self.events.append({"step": observation.step, "event": "repair", "from_stage": start,
                            "generated_stages": len(suffix), "strategy": self.strategy})
        return self.stage_plan

    @torch.inference_mode()
    def _extend(self, observation):
        old = self.stage_plan
        suffix, decoded = self._generate(self._observe(observation), old.latents, "rolling")
        self.stage_plan = StagePlan(torch.cat((old.latents, suffix)),
                                    old.stages + self.model.stage_specs(decoded), old.active)
        self.events.append({"step": observation.step, "event": "rolling_extension", "stages": len(suffix)})

    @torch.inference_mode()
    def decode_actions(self, observation):
        encoded = self._observe(observation)
        latents = (self.model.empty_latents() if self.stage_plan is None else
                   self.stage_plan.latents[self.stage_plan.active:])
        with self.ledger.charge("action"):
            context = self.model.contextual_features(self._context(encoded))
            normalized = self.model.actions(context, latents).float().cpu().numpy()
        actions = self.normalizer.denormalize(normalized)
        if not np.isfinite(actions).all():
            raise FloatingPointError("The policy produced non-finite actions")
        return actions

    @torch.inference_mode()
    def next_chunk(self, observation):
        self._observe(observation)
        if self.config.variant in {"base", "auxiliary"}:
            return self.decode_actions(observation)
        if self.stage_plan is None:
            self.plan(observation)
        if self.stage_plan.active == len(self.stage_plan.stages):
            self._extend(observation)
        estimate = self.estimate_state(observation)
        result = self.verifier.check(self.stage_plan, self.history, estimate)
        self.events.append({"step": observation.step, "event": "verification", "decision": result.decision.value,
                            "reason": result.reason, "discrepancy": result.discrepancy})
        if result.decision == Decision.COMPLETE:
            active = self.stage_plan.active
            self.history.confirm(active, self.stage_plan.stages[active], observation.step)
            self.stage_plan.active += 1
            if self.stage_plan.active == len(self.stage_plan.stages):
                self._extend(observation)
        elif result.decision == Decision.REPAIR:
            if self.strategy != "disabled":
                self.repair(observation, result.repair_from)
            else:
                self.history.invalidate_from(result.repair_from)
        return self.decode_actions(observation)
