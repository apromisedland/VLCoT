import pytest

from vlcot.data.io import file_hash, write_json
from vlcot.evaluation.evaluate import load_protocol


def test_arbitrary_ten_task_substitution_is_rejected(tmp_path):
    official = tmp_path / "official.json"
    write_json(official, {"task_names": [f"task-{index}" for index in range(10)]})
    manifest = tmp_path / "evaluation.json"
    write_json(manifest, {
        "schema_version": 1, "benchmark": "robotwin10", "subset_run": True,
        "provenance": {"source": "official-source", "revision": "version", "official_manifest": official.name,
                       "sha256": file_hash(official)},
        "episodes": [{"id": "one", "task": "unrelated-task", "suite": "robotwin10", "seed": 0,
                      "max_steps": 100, "instruction": "place", "condition": "clean"}],
    })
    with pytest.raises(ValueError, match="ten-task list"):
        load_protocol(manifest)


def test_missing_official_recovery_manifest_is_rejected(tmp_path):
    manifest = tmp_path / "evaluation.json"
    write_json(manifest, {"schema_version": 1, "benchmark": "libero_recover", "provenance": {"source": "source"},
                          "episodes": [{"id": "one"}]})
    with pytest.raises(ValueError, match="official instance/task manifest"):
        load_protocol(manifest)
