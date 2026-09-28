import numpy as np

from vlcot.config import VerificationConfig
from vlcot.control.verification import Decision, StageVerifier
from vlcot.types import StagePlan, StageSpec, StateEstimate, VerifiedHistory


def stage(expected, relevant=None, required=None, required_values=None, releases=None):
    expected = np.asarray(expected, dtype=float)
    count = len(expected)
    return StageSpec(expected, np.ones(count, bool) if relevant is None else np.asarray(relevant, bool),
                     np.zeros(count, bool) if required is None else np.asarray(required, bool),
                     np.zeros(count) if required_values is None else np.asarray(required_values, float),
                     np.zeros(count, bool) if releases is None else np.asarray(releases, bool))


def plan(stages, active=0):
    return StagePlan(np.zeros((len(stages), 4, 8)), stages, active)


def estimate(values, step, visible=None):
    return StateEstimate(np.asarray(values, float), np.ones(len(values)) if visible is None else np.asarray(visible), step)


def test_missing_evidence_never_certifies_zero_discrepancy():
    verifier = StageVerifier(VerificationConfig())
    current = plan([stage([0])])
    for step_index in range(8):
        result = verifier.check(current, VerifiedHistory(), estimate([0], step_index, [0]))
        assert result.decision == Decision.UNCONFIRMED


def test_interruption_resets_consecutive_completion():
    verifier = StageVerifier(VerificationConfig())
    current = plan([stage([1])])
    decisions = []
    for step_index, visible in enumerate([1, 1, 0, 1, 1, 1]):
        decisions.append(verifier.check(current, VerifiedHistory(), estimate([1], step_index, [visible])).decision)
    assert decisions[-1] == Decision.COMPLETE
    assert Decision.COMPLETE not in decisions[:-1]


def test_intermediate_mismatch_is_not_immediate_failure():
    verifier = StageVerifier(VerificationConfig())
    current = plan([stage([1])])
    for step_index, value in enumerate([0.0, 0.02, 0.04, 0.06, 0.08]):
        result = verifier.check(current, VerifiedHistory(), estimate([value], step_index))
        assert result.decision == Decision.CONTINUE


def test_stagnation_eventually_repairs_active_stage():
    verifier = StageVerifier(VerificationConfig(stagnation_observations=3))
    current = plan([stage([1])])
    for step_index in range(4):
        result = verifier.check(current, VerifiedHistory(), estimate([0], step_index))
    assert result.decision == Decision.REPAIR
    assert result.repair_from == 0


def test_invalid_drawer_prerequisite_repairs_earliest_stage():
    opening = stage([1, 0], relevant=[1, 0])
    grasping = stage([1, 1], relevant=[0, 1], required=[1, 0], required_values=[1, 0])
    current = plan([opening, grasping], active=1)
    history = VerifiedHistory()
    history.confirm(0, opening, 0)
    verifier = StageVerifier(VerificationConfig())
    for step_index in range(1, 4):
        result = verifier.check(current, history, estimate([0, 0], step_index))
    assert result.decision == Decision.REPAIR
    assert result.repair_from == 0


def test_legitimate_release_does_not_invalidate_grasp():
    grasping = stage([1, 0], relevant=[1, 0])
    placement = stage([0, 1], relevant=[0, 1], releases=[1, 0])
    final = stage([0, 1], relevant=[0, 1], required=[0, 1], required_values=[0, 1])
    current = plan([grasping, placement, final], active=2)
    history = VerifiedHistory()
    history.confirm(0, grasping, 0)
    history.confirm(1, placement, 1)
    verifier = StageVerifier(VerificationConfig())
    for step_index in range(2, 5):
        result = verifier.check(current, history, estimate([0, 1], step_index))
    assert result.decision == Decision.COMPLETE
    assert history.summary(2)[2] == 0


def test_dependency_survives_multiple_planning_windows():
    opening = stage([1, 0], relevant=[1, 0])
    middle = [stage([1, 0], relevant=[0, 1]) for _ in range(9)]
    current_stage = stage([1, 1], relevant=[0, 1], required=[1, 0], required_values=[1, 0])
    current = plan([opening, *middle, current_stage], active=10)
    history = VerifiedHistory()
    for index, specification in enumerate(current.stages[:10]):
        history.confirm(index, specification, index)
    verifier = StageVerifier(VerificationConfig())
    for step_index in range(11, 14):
        result = verifier.check(current, history, estimate([0, 0], step_index))
    assert result.repair_from == 0
