#!/usr/bin/env python3
"""
Add robot0_eye_in_hand_image from HDF5 to a long human_square_30k.npz file.

Usage:
python add_robot_image_long_npz.py \
    --npz_file /path/to/human_square_30k.npz \
    --hdf5_file /path/to/square_dataset.hdf5 \
    --out_file /path/to/human_square_30k_updated.npz
"""

import argparse
import h5py
import numpy as np
import os

def add_robot_image_long_npz(npz_file, hdf5_file, out_file):
    # Load long NPZ and copy to a mutable dict
    npz_data = np.load(npz_file, allow_pickle=True)
    data = {k: npz_data[k] for k in npz_data}
    npz_data.close()
    print(f"Loaded {npz_file} with keys: {list(data.keys())}")

    done_flags = data["done"].astype(bool)
    done_indices = np.where(done_flags)[0]
    print(f"Found {len(done_indices)} demos in long npz")

    # Load HDF5 dataset
    h5_root = h5py.File(hdf5_file, "r")
    h5_data = h5_root["data"]

    start_idx = 0
    for demo_num, end_idx in enumerate(done_indices):
        demo_length = end_idx - start_idx + 1

        h5_demo_key = f"demo_{demo_num}"
        if h5_demo_key not in h5_data:
            print(f"Warning: {h5_demo_key} not found in HDF5, skipping demo")
            start_idx = end_idx + 1
            continue

        h5_demo = h5_data[h5_demo_key]
        obs_group = h5_demo["obs"]

        if "robot0_eye_in_hand_image" not in obs_group:
            print(f"No robot image for {h5_demo_key}, skipping")
            start_idx = end_idx + 1
            continue

        robot_images = obs_group["robot0_eye_in_hand_image"][:]
        if robot_images.shape[0] != demo_length:
            raise ValueError(
                f"Step mismatch for {h5_demo_key}: npz={demo_length} vs hdf5={robot_images.shape[0]}"
            )

        # Inject robot images into NPZ data dict
        if "robot0_eye_in_hand_image" not in data:
            # Initialize full array with zeros
            data["robot0_eye_in_hand_image"] = np.zeros(
                (data["done"].shape[0],) + robot_images.shape[1:], dtype=np.uint8
            )

        data["robot0_eye_in_hand_image"][start_idx:end_idx + 1] = robot_images.astype(np.uint8)
        print(f"Added robot images for {h5_demo_key}, steps {start_idx} → {end_idx}")

        start_idx = end_idx + 1

    h5_root.close()

    # Save updated NPZ
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(out_file, **data)
    print(f"Saved updated npz to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file", required=True, help="Path to long human_square_30k.npz")
    parser.add_argument("--hdf5_file", required=True, help="Path to HDF5 dataset file")
    parser.add_argument("--out_file", required=True, help="Path to save updated npz file")
    args = parser.parse_args()

    add_robot_image_long_npz(args.npz_file, args.hdf5_file, args.out_file)
