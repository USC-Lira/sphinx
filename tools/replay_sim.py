import os
from typing import Optional

import numpy as np
import torch
import pyrallis

import common_utils
from envs.robomimic_env import RobomimicEnv, RobomimicEnvConfig
from dataset_utils.rotation_wrappers import EulerQuatRotationWrapper
from interactive_scripts.dataset_recorder import ActMode


def replay_processed_episodes(
	episodes: list[list[dict]],
	dataset_path: str,
	save_dir: str,
	max_demos: int = 2,
	action_wrapper: Optional[EulerQuatRotationWrapper] = None,
	camera_view: str = "agentview_image",
) -> list[str]:
	"""Replay processed Hydra episodes in robomimic and save demo videos.

	Args:
		episodes: Processed episodes from HydraDataset._load_and_process_episodes.
		dataset_path: Path containing env_cfg.yaml.
		save_dir: Base directory to save demos (videos saved under save_dir/demos).
		max_demos: Number of episodes to replay.
		action_wrapper_cfg: Action wrapper config to convert actions for eval.
		record_camera: Camera key to record in videos.
	"""
	if len(episodes) == 0:
		return []

	env_cfg_path = os.path.join(dataset_path, "env_cfg.yaml")
	env_cfg = pyrallis.load(RobomimicEnvConfig, open(env_cfg_path))  # type: ignore

	env = RobomimicEnv(env_cfg)
	demos_dir = os.path.join(save_dir, "demos")
	recorder = common_utils.Recorder(demos_dir)

	saved_paths: list[str] = []
	num_demos = min(max_demos, len(episodes))

	for demo_idx in range(num_demos):
		env.reset()

		if camera_view is not None:
			recorder.add_numpy(env.observe(), [camera_view])

		episode = episodes[demo_idx]

		prev_waypoint_action = None

		for step in episode:
			target_mode = step["target_mode"]
			if isinstance(target_mode, torch.Tensor):
				target_mode = int(target_mode.item())

			if target_mode == ActMode.Waypoint.value:
				waypoint_action = step["waypoint_action"]
				if prev_waypoint_action is not None and np.allclose(waypoint_action, prev_waypoint_action, atol=1e-6):
					# Current timestep is part of the same waypoint action that was applied in the previous timestep
					continue
				prev_waypoint_action = waypoint_action
				
				if isinstance(waypoint_action, torch.Tensor):
					waypoint_action = waypoint_action.detach().cpu().numpy()

				if action_wrapper is not None:
					waypoint_action = action_wrapper.process_for_eval(
						waypoint_action, is_delta=False
					)

				ee_pos, ee_euler, gripper_open = np.split(waypoint_action, [3, 6])
				gripper_open = 0 if float(gripper_open[0]) < 0.5 else 1
				env.move_to(ee_pos, ee_euler, gripper_open, recorder=recorder)

				if camera_view is not None:
					obs = env.observe()
					obs[camera_view][:5, :, :] = (255, 0, 0)
					recorder.add_numpy(obs, [camera_view])

			elif target_mode == ActMode.Dense.value:
				dense_action = step["dense_action"]
				prev_waypoint_action = None  # Reset prev_waypoint_action since we're no longer in a waypoint mode

				if isinstance(dense_action, torch.Tensor):
					dense_action = dense_action.detach().cpu().numpy()

				if action_wrapper is not None:
					dense_action = action_wrapper.process_for_eval(
						dense_action, is_delta=True
					)

				ee_pos, ee_euler, gripper_open = np.split(dense_action, [3, 6])

				if camera_view is not None:
					obs = env.observe()
					obs[camera_view][:5, :, :] = (135, 206, 235)
					recorder.add_numpy(obs, [camera_view])

				env.apply_action(ee_pos, ee_euler, float(gripper_open[0]), is_delta=True)

			if env.terminal:
				break

		if camera_view is not None:
			recorder.add_numpy(env.observe(), [camera_view])

		demo_name = f"demo{demo_idx:05d}"
		saved_paths.append(recorder.save(demo_name))

	del env
	del recorder

	return saved_paths
