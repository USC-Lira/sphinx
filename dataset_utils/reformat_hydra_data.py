#!/usr/bin/env python3
# Reformats the original hydra dataset to be compatible with Sphinx codebase.

import argparse
import os
import numpy as np
import gc
import multiprocessing as mp

from interactive_scripts.dataset_recorder import ActMode


def reformat_episode(data, start, end):
    # ==========================================================
    # PASS 1: extract waypoint list ONLY
    # ==========================================================
    waypoints = []

    for t in range(start, end + 1):
        # Timestep i is a waypoint if: 
            # previous region is sparse mode and i is clicked (meaning it ends the sparse segment) OR
            # i is sparse mode AND i is the last timestep
        if (data["click_state"][t].item() == 1 and t-1 >= start and data["mode"][t-1, 0].item() == 0) or (data["mode"][t, 0].item() == 0 and t == end):
            ee_pos = data["robot0_eef_pos"][t]
            ee_euler = data["robot0_eef_eul"][t]
            gripper = data["robot0_gripper_qpos"][t, 0]

            wp_action = np.concatenate([
                ee_pos,
                ee_euler,
                np.array([gripper], dtype=np.float32),
            ]).astype(np.float32)

            waypoints.append({
                "action": wp_action,
                "click_state": None,
            })

    # ==========================================================
    # PASS 2: build final demo (insert + label)
    # ==========================================================
    demo = []
    waypoint_idx = -1
    rollout_timestep = 0

    prev_mode = None
    prev_click = 0

    for t in range(start, end + 1):
        hydra_mode = data["mode"][t, 0].item()
        click_state = data["click_state"][t].item()
        obs = {
            "ee_pos": data["robot0_eef_pos"][t].astype(np.float32),
            "ee_euler": data["robot0_eef_eul"][t].astype(np.float32),
            "gripper_open": np.array(
                [data["robot0_gripper_qpos"][t, 0]],
                dtype=np.float32,
            ),
            "agentview_image": data["image"][t].astype(np.uint8),
            "object": data["object"][t].astype(np.float32),
            "target/position": data["target/position"][t].astype(np.float32),
            "target/orientation": data["target/orientation"][t].astype(np.float32),
            "target/orientation_eul": data["target/orientation_eul"][t].astype(np.float32),
        }
        if "robot0_eye_in_hand_image" in data:
            obs["robot0_eye_in_hand_image"] = data["robot0_eye_in_hand_image"][t].astype(np.uint8)
            
        # proprio is a flattened concatenation of ee_pos, ee_euler, gripper_open
        proprio = np.concatenate([obs["ee_pos"], obs["ee_euler"], obs["gripper_open"]], axis=0)
        obs["proprio"] = proprio

        # map timestep mode
        if hydra_mode == 0:
            step_mode = ActMode.Interpolate
        elif hydra_mode == 1:
            step_mode = ActMode.Dense
        else:
            continue

        # --------------------------------------------------
        # decide waypoint insertion
        # --------------------------------------------------
        insert_wp = False
        if step_mode == ActMode.Interpolate: # waypoint actions are always before interpolate timesteps
            # insert a waypoint action right after dense modes or right after previous waypoint has been reached
            if (
                prev_mode is None or
                prev_mode == ActMode.Dense or
                prev_click == 1
            ):
                insert_wp = True

        if insert_wp:
            waypoint_idx += 1
            print(f"Extracting waypoint {waypoint_idx} from {len(waypoints)} waypoints")

            if waypoint_idx == len(waypoints):
                print(f"Waypoint index out of range: {waypoint_idx} vs {len(waypoints)}")
                return None
            
            wp = waypoints[waypoint_idx]

            demo.append({
                "obs": obs, # record the current timestep's observations as the initial observation before the waypoint action
                "action": wp["action"],
                "mode": ActMode.Waypoint,
                "waypoint_idx": waypoint_idx,
                "rollout_timestep": rollout_timestep,
                "click_state": wp["click_state"],
            })

            rollout_timestep += 1

        # --------------------------------------------------
        # add timestep itself
        # --------------------------------------------------

        demo.append({
            "obs": obs,
            "action": data["action"][t].astype(np.float32),
            "mode": step_mode,
            "waypoint_idx": waypoint_idx if step_mode == ActMode.Interpolate else -1,
            "rollout_timestep": rollout_timestep,
            "click_state": data["click_state"][t].astype(np.float32),
        })

        rollout_timestep += 1
        prev_mode = step_mode
        prev_click = click_state

    assert waypoint_idx == len(waypoints) - 1, f"Waypoint insertion mismatch ({waypoint_idx} vs {len(waypoints)})"

    return demo

def process_episode_wrapper(args_tuple):
    in_file, out_dir, demo_idx, start, end = args_tuple
    data = np.load(in_file, allow_pickle=True, mmap_mode="r")
    demo_steps = reformat_episode(data, start, end)
    if not demo_steps: 
        return
    out_path = os.path.join(out_dir, f"demo{demo_idx:05d}.npz")
    np.savez_compressed(out_path, demo_steps)
    print(f"Saved {out_path} ({len(demo_steps)} steps)")
    del demo_steps, data
    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.in_file, allow_pickle=True, mmap_mode="r")
    done = data["done"].astype(bool)
    done_idxs = np.where(done)[0]
    del data

    print(f"Found {len(done_idxs)} episodes")

    # Build task list
    tasks = []
    start = 0
    for demo_idx, end in enumerate(done_idxs):
        tasks.append((args.in_file, args.out_dir, demo_idx, start, int(end)))
        start = int(end) + 1

    # Process in parallel - adjust number based on your CPU cores
    with mp.Pool(processes=8) as pool:
        pool.map(process_episode_wrapper, tasks)

    print("Done.")


if __name__ == "__main__":
    main()