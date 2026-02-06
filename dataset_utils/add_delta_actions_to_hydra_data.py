#!/usr/bin/env python3
"""
Replace non-Waypoint-step actions in per-demo NPZ files with delta actions from HDF5.

Usage:
python update_dense_actions_from_hdf5.py \
    --npz_dir /path/to/npz_demos \
    --hdf5_file /path/to/dataset.hdf5 \
    --out_dir /path/to/output_npz_demos
"""

import argparse
import h5py
import numpy as np
import os
import re
from glob import glob

from interactive_scripts.dataset_recorder import ActMode


def update_demo_npz(npz_path, h5_data, demo_num, out_dir):
    demo_name = os.path.basename(npz_path)
    print(f"\nProcessing {demo_name}")

    npz = np.load(npz_path, allow_pickle=True)

    if "arr_0" not in npz:
        raise KeyError(f"{demo_name} missing 'arr_0' (expected list of steps)")

    steps = list(npz["arr_0"])  # list of per-step dicts
    npz.close()

    h5_demo_key = f"demo_{demo_num}"
    if h5_demo_key not in h5_data:
        raise KeyError(f"{h5_demo_key} not found in HDF5")

    h5_demo = h5_data[h5_demo_key]

    if "actions" not in h5_demo:
        raise KeyError(f"{h5_demo_key} missing 'actions'")

    hdf5_actions = h5_demo["actions"][:]

    # Indices of steps that consume delta actions
    non_waypoint_indices = [
        i for i, step in enumerate(steps)
        if step["mode"] != ActMode.Waypoint
    ]

    print(f"  Non-Waypoint steps: {len(non_waypoint_indices)}")
    print(f"  HDF5 steps:        {hdf5_actions.shape[0]}")

    if len(non_waypoint_indices) != hdf5_actions.shape[0]:
        raise ValueError(
            f"Step mismatch in {demo_name}: "
            f"non_waypoint_steps={len(non_waypoint_indices)} "
            f"vs hdf5={hdf5_actions.shape[0]}"
        )

    # Overwrite ONLY non-Waypoint actions
    for i, step_idx in enumerate(non_waypoint_indices):
        steps[step_idx]["action"] = hdf5_actions[i]

    # Save updated demo with identical structure
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, demo_name)
    np.savez_compressed(out_path, np.array(steps, dtype=object))

    print(f"  ✔ Saved → {out_path}")


def main(npz_dir, hdf5_file, out_dir):
    npz_files = sorted(glob(os.path.join(npz_dir, "demo*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No demo NPZ files found in {npz_dir}")

    print(f"Found {len(npz_files)} demo NPZ files")

    with h5py.File(hdf5_file, "r") as h5_root:
        h5_data = h5_root["data"]

        for demo_num, npz_path in enumerate(npz_files):
            demo_num = int(re.search(r'demo(\d+)', npz_path).group(1))
            update_demo_npz(
                npz_path=npz_path,
                h5_data=h5_data,
                demo_num=demo_num,
                out_dir=out_dir,
            )

    print("\nAll demos processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", required=True, help="Directory of demoXXXXX.npz files")
    parser.add_argument("--hdf5_file", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--out_dir", required=True, help="Output directory for updated NPZs")
    args = parser.parse_args()

    main(args.npz_dir, args.hdf5_file, args.out_dir)
