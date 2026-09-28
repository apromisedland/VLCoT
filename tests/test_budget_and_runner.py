from types import SimpleNamespace

import numpy as np
import pytest

from vlcot.control.budget import OPERATIONS, BudgetExhausted, BudgetLedger
from vlcot.control.runner import EnvironmentStep, run_episode
from vlcot.types import EpisodeRecord, Observation


class Clock:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        self.time += 0.25
        return self.time


def ledger(limit=10):
    return BudgetLedger(limit, {operation: 1.0 for operation in OPERATIONS}, clock=Clock())


def test_call_is_refused_before_computation():
    accounting = ledger(1)
    with accounting.charge("encode"):
        pass
    entered = False
    with pytest.raises(BudgetExhausted):
        with accounting.charge("action"):
            entered = True
    assert not entered
    assert accounting.spent == 1


def test_failed_model_call_is_still_charged():
    accounting = ledger()
    with pytest.raises(RuntimeError):
        with accounting.charge("repair", 4):
            raise RuntimeError("model failure")
    assert accounting.spent == 4
    assert accounting.seconds == 0.25
    assert accounting.records["repair"]["calls"] == 1


def test_nested_time_accounting_is_rejected():
    accounting = ledger()
    with accounting.charge("action"):
        with pytest.raises(RuntimeError):
            with accounting.charge("encode"):
                pass


class ScriptedEnvironment:
    def __init__(self):
        self.steps = 0
        self.closed = False

    def state(self):
        observation = Observation({"camera": np.zeros((8, 8, 3), np.uint8)}, np.zeros(1), "place object", self.steps)
        return EnvironmentStep(observation, self.steps >= 2, self.steps >= 2)

    def reset(self):
        self.steps = 0
        return self.state()

    def step(self, action):
        self.steps += 1
        return self.state()

    def close(self):
        self.closed = True


class ScriptedPolicy:
    config = SimpleNamespace(execution_horizon=4)

    def reset(self, accounting):
        self.ledger = accounting
        self.events = []
        self.regenerated_vectors = 0

    def next_chunk(self, observation):
        with self.ledger.charge("action", 2):
            return np.zeros((8, 1))


def test_runner_stops_chunk_at_environment_success():
    environment = ScriptedEnvironment()
    record = EpisodeRecord("one", "libero", "libero_10", "task", 0)
    result = run_episode(environment, ScriptedPolicy(), ledger(), record, 10, clock=Clock())
    assert result.success
    assert result.environment_steps == 2
    assert len(result.decision_latencies_ms) == 1
    assert environment.closed


def test_exhaustion_is_a_recorded_failure_without_actions():
    environment = ScriptedEnvironment()
    record = EpisodeRecord("one", "libero_recover", "libero_100", "task", 0, recovery_level=4)
    result = run_episode(environment, ScriptedPolicy(), ledger(1), record, 10, clock=Clock())
    assert not result.success
    assert result.termination == "budget_exhausted"
    assert result.environment_steps == 0
