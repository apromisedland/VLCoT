import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vlcot.control.budget import BudgetExhausted
from vlcot.types import EpisodeRecord, Observation


@dataclass
class EnvironmentStep:
    observation: Observation
    terminated: bool
    success: bool


class Environment(Protocol):
    def reset(self) -> EnvironmentStep: ...
    def step(self, action: np.ndarray) -> EnvironmentStep: ...
    def close(self) -> None: ...


def run_episode(environment: Environment, policy, ledger, record: EpisodeRecord, max_steps: int,
                clock=time.perf_counter):
    if max_steps < 1:
        raise ValueError("An explicit positive environment step limit is required")
    policy.reset(ledger)
    try:
        state = environment.reset()
        while record.environment_steps < max_steps:
            if state.terminated or state.success:
                record.success = bool(state.success)
                record.termination = "success" if state.success else "environment_terminated"
                break
            started = clock()
            actions = policy.next_chunk(state.observation)
            ledger.synchronize()
            record.decision_latencies_ms.append((clock() - started) * 1000)
            steps = min(policy.config.execution_horizon, max_steps - record.environment_steps)
            for action in actions[:steps]:
                state = environment.step(action)
                record.environment_steps += 1
                if state.terminated or state.success:
                    break
        if record.termination == "unfinished":
            record.success = bool(state.success)
            record.termination = "success" if state.success else "step_limit"
    except BudgetExhausted as error:
        record.termination = "budget_exhausted"
        record.events.append({"event": "budget_exhausted", "detail": str(error)})
    except Exception as error:
        record.termination = "error"
        record.events.append({"event": "error", "type": type(error).__name__, "detail": str(error)})
    finally:
        record.inference_seconds = ledger.seconds
        record.costs = ledger.records
        record.regenerated_vectors = policy.regenerated_vectors
        record.events.extend(policy.events)
        record.metadata["budget_limit"] = ledger.limit
        record.metadata["budget_spent"] = ledger.spent
        try:
            environment.close()
        except Exception as error:
            record.events.append({"event": "environment_close_error", "detail": str(error)})
    return record
