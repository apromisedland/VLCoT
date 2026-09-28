import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from vlcot.data.io import file_hash, read_json, resolve_path, write_json
from vlcot.evaluation.environments import LIBERO_REVISION, check_revision


def replay_libero_labels(manifest_path, schema_path, benchmark_root, output_directory):
    root = Path(benchmark_root).resolve()
    check_revision(root, LIBERO_REVISION)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from libero.libero.envs import SegmentationRenderEnv
    from libero.libero.envs.predicates import eval_predicate_fn

    manifest = read_json(manifest_path)
    schemas = read_json(schema_path)
    if manifest["predicate_names"] != schemas["predicate_names"]:
        raise ValueError("Replay schema predicate order differs from the trajectory vocabulary")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Replay labeling requires a fresh output directory")
    episodes = []
    for episode_index, episode in enumerate(manifest["episodes"]):
        schema = schemas["tasks"][episode["task"]]
        bddl = resolve_path(schema_path, schema["bddl_file"])
        camera_mapping = schema.get("cameras", {"agentview": "agentview", "wrist": "robot0_eye_in_hand"})
        if set(camera_mapping) != set(manifest["camera_names"]):
            raise ValueError("Replay cameras must match policy cameras")
        environment = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_names=list(camera_mapping.values()),
                                            camera_heights=256, camera_widths=256)
        destination = output / f"episode_{episode_index:06d}.h5"
        shutil.copyfile(resolve_path(manifest_path, episode["path"]), destination)
        try:
            environment.seed(int(episode.get("seed", 0)))
            environment.reset()
            with h5py.File(destination, "r+") as data:
                if "offline/sim_states" not in data:
                    raise ValueError("LIBERO replay requires stored simulator states for every prepared frame")
                length = len(data["actions"])
                shape = (length, len(manifest["predicate_names"]))
                states, valid, observable, visibility_valid = [np.zeros(shape, dtype=np.float32) for _ in range(4)]
                for frame in range(length):
                    raw = environment.set_init_state(np.asarray(data["offline/sim_states"][frame]))
                    for predicate_index, name in enumerate(manifest["predicate_names"]):
                        specification = schema["predicates"].get(name)
                        if specification is None:
                            continue
                        kind = specification["kind"]
                        if kind == "bddl":
                            arguments = [environment.env.object_states_dict[name] for name in specification["objects"]]
                            states[frame, predicate_index] = eval_predicate_fn(specification["relation"], *arguments)
                        elif kind == "grasp":
                            robot = environment.env.robots[int(specification.get("robot", 0))]
                            object_model = environment.env.objects_dict[specification["object"]]
                            states[frame, predicate_index] = environment.env._check_grasp(robot.gripper, object_model.contact_geoms)
                        elif kind == "goal":
                            states[frame, predicate_index] = environment.check_success()
                        else:
                            raise ValueError(f"Unknown offline predicate kind: {kind}")
                        valid[frame, predicate_index] = 1
                        visibility_objects = specification.get("visibility_objects")
                        if not visibility_objects:
                            continue
                        if specification.get("visibility_rule") != "all_objects_visible_proxy":
                            raise ValueError("Segmentation visibility requires explicit opt-in to the documented proxy")
                        minimum_pixels = int(specification.get("minimum_visible_pixels", 20))
                        visible_objects = []
                        for object_name in visibility_objects:
                            if object_name not in environment.instance_to_id:
                                raise ValueError(f"No segmentation instance for {object_name}; supply independently labeled visibility")
                            instance = environment.instance_to_id[object_name]
                            visible_objects.append(any(
                                np.count_nonzero(raw[f"{camera}_segmentation_instance"] == instance) >= minimum_pixels
                                for camera in camera_mapping.values()
                            ))
                        observable[frame, predicate_index] = all(visible_objects)
                        visibility_valid[frame, predicate_index] = 1
                for name, values in {"predicates": states, "predicate_valid": valid, "observable": observable,
                                     "visibility_valid": visibility_valid}.items():
                    key = f"offline/{name}"
                    if key in data:
                        if name in {"observable", "visibility_valid"}:
                            supplied = np.asarray(data[key])
                            if supplied.shape != shape:
                                raise ValueError("Supplied visibility trace shape mismatch")
                            values = supplied
                        del data[key]
                    data.create_dataset(key, data=values)
                data.attrs["label_schema_sha256"] = file_hash(schema_path)
        finally:
            environment.close()
        episodes.append({**episode, "path": destination.name})
    write_json(output / "manifest.json", {**manifest, "episodes": episodes, "replay_revision": LIBERO_REVISION})
    return output / "manifest.json"
