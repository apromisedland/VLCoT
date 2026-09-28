import importlib
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

from vlcot.control.runner import EnvironmentStep
from vlcot.types import Observation


LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
ROBOTWIN_REVISION = "13c3c47ff4312dd62484bcd51be034af55c062d1"


def check_revision(root, expected):
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if result.stdout.strip() != expected:
        raise ValueError(f"Benchmark checkout must be pinned to {expected}")


def transform_image(image, transform):
    if transform == "none":
        return np.asarray(image, dtype=np.uint8).copy()
    if transform == "rotate180":
        return np.asarray(image, dtype=np.uint8)[::-1, ::-1].copy()
    if transform == "flip_vertical":
        return np.asarray(image, dtype=np.uint8)[::-1].copy()
    raise ValueError(f"Unknown image transform: {transform}")


class LiberoEnvironment:
    def __init__(self, episode, settings, root):
        self.episode = episode
        self.settings = settings
        self.root = Path(root).resolve()
        check_revision(self.root, LIBERO_REVISION)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self.environment = None
        self.counter = 0

    def observation(self, raw):
        from robosuite.utils.transform_utils import quat2axisangle

        mappings = self.settings.get("cameras", {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"})
        images = {name: transform_image(raw[source], self.settings.get("image_transform", "none"))
                  for name, source in mappings.items()}
        proprio = np.concatenate((raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]),
                                  raw["robot0_gripper_qpos"])).astype(np.float32)
        return Observation(images, proprio, self.instruction, self.counter)

    def reset(self):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.counter = 0
        recovery = self.episode.get("recovery_level") is not None
        if recovery:
            bddl = self.episode["bddl_file"]
            self.instruction = self.episode["instruction"]
            state = np.load(self.episode["state_file"], allow_pickle=False)
        else:
            suite_name = self.episode["suite"]
            suite = benchmark.get_benchmark_dict()[suite_name]()
            task = suite.get_task(int(self.episode["task_id"]))
            if self.episode["task"] != task.name:
                raise ValueError("Manifest task name does not match the pinned LIBERO task")
            self.instruction = task.language
            bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
            initial_states = suite.get_task_init_states(int(self.episode["task_id"]))
            state = np.asarray(initial_states[int(self.episode["initial_state_index"])])
        resolution = int(self.settings.get("image_size", 256))
        self.environment = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
        self.environment.seed(int(self.episode["seed"]))
        self.environment.reset()
        raw = self.environment.set_init_state(state)
        settle = int(self.settings.get("settle_steps", 0 if recovery else 10))
        if recovery and settle:
            raise ValueError("Recovery snapshots must not be modified by implicit settling steps")
        for _ in range(settle):
            raw, _, _, _ = self.environment.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=float))
        success = bool(self.environment.check_success())
        return EnvironmentStep(self.observation(raw), success, success)

    def step(self, action):
        if np.shape(action) != (7,):
            raise ValueError("LIBERO expects seven-dimensional native delta-pose/gripper actions")
        raw, _, done, _ = self.environment.step(np.asarray(action, dtype=float))
        self.counter += 1
        success = bool(self.environment.check_success())
        return EnvironmentStep(self.observation(raw), bool(done), success)

    def close(self):
        if self.environment is not None:
            self.environment.close()
            self.environment = None


@contextmanager
def working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


class RoboTwinEnvironment:
    def __init__(self, episode, settings, root):
        self.episode = episode
        self.settings = settings
        self.root = Path(root).resolve()
        check_revision(self.root, ROBOTWIN_REVISION)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self.environment = None
        self.counter = 0

    def _configuration(self):
        def read(relative):
            return yaml.safe_load((self.root / relative).read_text(encoding="utf-8"))

        condition = self.episode["condition"]
        if condition not in {"clean", "randomized"}:
            raise ValueError("RoboTwin condition must be clean or randomized")
        task_config = "demo_clean" if condition == "clean" else "demo_randomized"
        config = read(f"task_config/{task_config}.yml")
        embodiments = read("task_config/_embodiment_config.yml")
        cameras = read("task_config/_camera_config.yml")
        embodiment = config["embodiment"]
        if len(embodiment) == 1:
            config["left_robot_file"] = config["right_robot_file"] = embodiments[embodiment[0]]["file_path"]
            config["dual_arm_embodied"] = True
        elif len(embodiment) == 3:
            config["left_robot_file"] = embodiments[embodiment[0]]["file_path"]
            config["right_robot_file"] = embodiments[embodiment[1]]["file_path"]
            config["embodiment_dis"] = embodiment[2]
            config["dual_arm_embodied"] = False
        else:
            raise ValueError("Unsupported RoboTwin embodiment configuration")
        for side in ("left", "right"):
            robot_file = Path(config[f"{side}_robot_file"])
            config[f"{side}_embodiment_config"] = read(robot_file / "config.yml")
        head = cameras[config["camera"]["head_camera_type"]]
        config.update(head_camera_h=head["h"], head_camera_w=head["w"], task_name=self.episode["task"],
                      task_config=task_config, policy_name="vlcot", collect_data=False, eval_video_log=False,
                      render_freq=0, is_test=True, eval_mode=True)
        config["data_type"]["rgb"] = True
        config["data_type"]["qpos"] = True
        return config

    def observation(self, raw):
        mapping = self.settings.get("cameras", {"head": "head_camera", "left_wrist": "left_camera",
                                                 "right_wrist": "right_camera"})
        images = {name: transform_image(raw["observation"][source]["rgb"], self.settings.get("image_transform", "none"))
                  for name, source in mapping.items()}
        proprio = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)
        return Observation(images, proprio, self.episode["instruction"], self.counter)

    def reset(self):
        with working_directory(self.root):
            module = importlib.import_module(f"envs.{self.episode['task']}")
            self.environment = getattr(module, self.episode["task"])()
            config = self._configuration()
            self.environment.setup_demo(now_ep_num=int(self.episode.get("trial", 0)),
                                        seed=int(self.episode["seed"]), **config)
            self.environment.set_instruction(self.episode["instruction"])
            self.counter = 0
            raw = self.environment.get_obs()
            self.action_dim = len(raw["joint_action"]["vector"])
            success = bool(self.environment.check_success())
            return EnvironmentStep(self.observation(raw), success, success)

    def step(self, action):
        if np.shape(action) != (self.action_dim,):
            raise ValueError("RoboTwin qpos action dimensionality differs from the selected embodiment")
        with working_directory(self.root):
            self.environment.take_action(np.asarray(action), action_type="qpos")
            self.counter += 1
            raw = self.environment.get_obs()
            success = bool(self.environment.check_success())
            terminated = success or self.environment.take_action_cnt >= self.environment.step_lim
            return EnvironmentStep(self.observation(raw), terminated, success)

    def close(self):
        if self.environment is not None:
            with working_directory(self.root):
                self.environment.close_env()
            self.environment = None
