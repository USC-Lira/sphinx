#!/usr/bin/env python3
"""
Overlay fields from a Robomimic HDF5 dataset into a long-format NPZ file,
then clean the output to retain only whitelisted keys.

Usage:
    python add_robot_image_long_npz.py \
        --npz_file  /path/to/human_square_30k.npz \
        --hdf5_file /path/to/square_dataset.hdf5 \
        --out_file  /path/to/human_square_30k_updated.npz
"""

import argparse
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fields to copy from HDF5 → NPZ.
# Format: (hdf5_group, field_name, npz_key, dtype_override)
#   hdf5_group:     "obs"  → demo/obs/<field_name>
#                   "demo" → demo/<field_name>
#   dtype_override: None   → use native HDF5 dtype
FIELDS_TO_COPY = [
    ("obs",  "robot0_eye_in_hand_image", "robot0_eye_in_hand_image", np.uint8),
    ("obs",  "agentview_image",          "image",          np.uint8),
    ("obs",  "robot0_eef_pos",           "robot0_eef_pos",           None),
    ("obs",  "robot0_eef_euler",         "robot0_eef_eul",         None),  # falls back to quat→euler
    ("obs",  "robot0_gripper_qpos",      "robot0_gripper_qpos",      None),
    ("demo", "actions",                  "action",                  None),
]

# Only these keys will be written to the output NPZ.
KEYS_TO_KEEP = {
    "done",
    "click_state",
    "mode",
    "robot0_eye_in_hand_image",
    "image",
    "robot0_eef_pos",
    "robot0_eef_eul",
    "robot0_gripper_qpos",
    "action",
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_npz(path: str) -> dict:
    npz = np.load(path, allow_pickle=True)
    data = {k: npz[k] for k in npz}
    npz.close()
    print(f"[npz]   Loaded '{path}'")
    print(f"        keys: {sorted(data.keys())}")
    return data


def save_npz(data: dict, out_file: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    np.savez_compressed(out_file, **data)
    print(f"[npz]   Saved → '{out_file}'")


def get_done_indices(data: dict) -> np.ndarray:
    indices = np.where(data["done"].astype(bool))[0]
    print(f"[npz]   Found {len(indices)} demo(s)")
    return indices


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(data: dict) -> dict:
    """Drop every key not in KEYS_TO_KEEP."""
    cleaned = {k: v for k, v in data.items() if k in KEYS_TO_KEEP}
    dropped = sorted(set(data.keys()) - KEYS_TO_KEEP)
    if dropped:
        print(f"[clean] Dropped : {dropped}")
    print(f"[clean] Keeping : {sorted(cleaned.keys())}")
    return cleaned


# ---------------------------------------------------------------------------
# HDF5 → NPZ copy logic
# ---------------------------------------------------------------------------

def resolve_h5_group(h5_demo, group_name: str):
    """Return the HDF5 group for 'obs' fields or the demo root for others."""
    if group_name == "obs":
        return h5_demo.get("obs")
    return h5_demo


def quat_to_euler(quat_arr: np.ndarray) -> np.ndarray:
    """Convert (N, 4) xyzw quaternions to (N, 3) XYZ euler angles (radians)."""
    return Rotation.from_quat(quat_arr).as_euler("xyz")


def load_field(group, field_name: str, demo_key: str):
    """
    Load a field from an HDF5 group.
    Special case: if field_name is 'robot0_eef_euler' and it's missing,
    fall back to 'robot0_eef_quat' and convert to euler in-place.
    Returns the array or None if unavailable.
    """
    if field_name in group:
        return group[field_name][:]

    if field_name == "robot0_eef_euler":
        if "robot0_eef_quat" in group:
            print(f"  [conv]  '{demo_key}': 'robot0_eef_euler' not found, "
                  f"converting from 'robot0_eef_quat'")
            quat = group["robot0_eef_quat"][:]
            return quat_to_euler(quat)
        else:
            print(f"  [skip]  '{demo_key}': neither 'robot0_eef_euler' nor "
                  f"'robot0_eef_quat' found")
            return None

    print(f"  [skip]  '{demo_key}/obs/{field_name}': not in HDF5")
    return None


def ensure_output_array(data: dict, npz_key: str, total_steps: int,
                        field_shape: tuple, dtype) -> None:
    """Lazily allocate a zero-filled array in `data` on first encounter."""
    if npz_key not in data:
        shape = (total_steps,) + field_shape
        data[npz_key] = np.zeros(shape, dtype=dtype)
        print(f"  [init]  '{npz_key}'  shape={shape}  dtype={dtype}")


def copy_demo_fields(data: dict, h5_demo, start_idx: int, end_idx: int,
                     demo_length: int, demo_key: str) -> None:
    """Copy every field in FIELDS_TO_COPY from one HDF5 demo into `data`."""
    total_steps = data["done"].shape[0]

    for group_name, field_name, npz_key, dtype_override in FIELDS_TO_COPY:
        group = resolve_h5_group(h5_demo, group_name)

        if group is None:
            print(f"  [skip]  '{demo_key}': group '{group_name}' not found")
            continue

        arr = load_field(group, field_name, demo_key)
        if arr is None:
            continue

        if arr.shape[0] != demo_length:
            raise ValueError(
                f"Step mismatch '{demo_key}/{field_name}': "
                f"npz={demo_length}  hdf5={arr.shape[0]}"
            )

        dtype = dtype_override if dtype_override is not None else arr.dtype
        ensure_output_array(data, npz_key, total_steps, arr.shape[1:], dtype)
        data[npz_key][start_idx : end_idx + 1] = arr.astype(dtype)
        print(f"  [ok]    '{npz_key}'  steps {start_idx}→{end_idx}")


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def overlay_robomimic_data(npz_file: str, hdf5_file: str, out_file: str) -> None:
    data = load_npz(npz_file)
    done_indices = get_done_indices(data)

    with h5py.File(hdf5_file, "r") as h5_root:
        h5_data = h5_root["data"]
        start_idx = 0

        for demo_num, end_idx in enumerate(done_indices):
            demo_length = end_idx - start_idx + 1
            demo_key = f"demo_{demo_num}"

            if demo_key not in h5_data:
                print(f"[warn]  '{demo_key}' not in HDF5, skipping")
                start_idx = end_idx + 1
                continue

            print(f"\n[demo]  {demo_key}  steps {start_idx}→{end_idx}  (len={demo_length})")
            copy_demo_fields(data, h5_data[demo_key], start_idx, end_idx,
                             demo_length, demo_key)
            start_idx = end_idx + 1

    data = clean_data(data)
    save_npz(data, out_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay HDF5 Robomimic fields into a long-format NPZ file."
    )
    parser.add_argument("--npz_file",  required=True, help="Input long NPZ file")
    parser.add_argument("--hdf5_file", required=True, help="Source HDF5 dataset")
    parser.add_argument("--out_file",  required=True, help="Output NPZ file path")
    args = parser.parse_args()

    overlay_robomimic_data(args.npz_file, args.hdf5_file, args.out_file)