import h5py
import numpy as np
import pytest

from vlcot.data.annotation import build_stages
from vlcot.data.io import assign_splits, fit_normalizer, write_json


def test_source_and_failure_groups_prevent_transitive_leakage():
    episodes = [
        {"id": "first", "source_trajectory": "source-a", "failure_scenario": "failure-a", "split": "train"},
        {"id": "bridge", "source_trajectory": "source-a", "failure_scenario": "failure-b"},
        {"id": "last", "source_trajectory": "source-b", "failure_scenario": "failure-b", "split": "test"},
    ]
    with pytest.raises(ValueError, match="cross dataset splits"):
        assign_splits(episodes)


def test_unobserved_state_changes_do_not_create_stages():
    states = np.array([[0], [1], [1], [1], [1]], dtype=float)
    validity = np.array([[1], [0], [0], [0], [1]], dtype=float)
    schema = {"stable_frames": 3, "goal_conditions": [{"predicate": "goal", "value": 1}],
              "events": [{"when": [{"predicate": "goal", "value": 1}],
                          "effects": [{"predicate": "goal", "value": 1}], "requires": [], "releases": []}]}
    stages = build_stages(states, validity, schema, ["goal"])
    assert stages["ends"].tolist() == [4]


def test_training_normalization_excludes_held_out_and_recovery_data(tmp_path):
    episodes = []
    for identity, split, recovery, values in [
        ("train", "train", False, [[0, 5], [2, 5]]),
        ("test", "test", False, [[1000, 1000]]),
        ("recover", "train", True, [[-1000, -1000]]),
    ]:
        path = tmp_path / f"{identity}.h5"
        with h5py.File(path, "w") as data:
            data.create_dataset("actions", data=np.asarray(values, dtype=float))
        episodes.append({"id": identity, "path": path.name, "split": split, "recovery": recovery})
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"episodes": episodes})
    normalizer = fit_normalizer(manifest)
    assert normalizer.statistics["episode_ids"] == ["train"]
    assert normalizer.high[0] < 2.01
    assert normalizer.low[0] >= 0
    assert normalizer.denormalize(np.array([[0, 0]]))[0, 1] == 5
