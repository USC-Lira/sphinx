import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch


def plot_timesteps_modes_waypoints(
    eef_positions,
    modes,
    out_path,
    title=None,
):
    """
    Parameters
    ----------
    eef_positions : (N, 3) np.ndarray
        End-effector positions over time
    modes : list[str]
        Mode name per timestep (e.g. 'Interpolate', 'Waypoint', 'Dense')
    out_path : str
        Where to save the PNG
    title : str, optional
        Figure title
    """

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    eef_positions = np.asarray(eef_positions)

    # ---- Color mapping ----
    color_map = {
        "Interpolate": "pink",
        "Waypoint": "red",
        "Dense": "skyblue",
    }

    modes = np.asarray(modes)

    # ---- Count waypoints ----
    waypoint_mask = modes == "Waypoint"
    num_waypoints = int(np.sum(waypoint_mask))

    # ---- Plot non-waypoints first ----
    non_wp_mask = ~waypoint_mask
    if np.any(non_wp_mask):
        pos = eef_positions[non_wp_mask]
        cols = [color_map.get(m, "gray") for m in modes[non_wp_mask]]
        ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=cols,
            s=40,
            alpha=0.7,
            zorder=1,
            edgecolors='none',
        )

        # ---- Arrows between consecutive points ----
        for i in range(len(pos) - 1):
            p0 = pos[i]
            v = pos[i + 1] - p0
            ax.quiver(
                p0[0], p0[1], p0[2],
                v[0], v[1], v[2],
                color="black",
                linewidth=0.5,
                arrow_length_ratio=0.08,
                alpha=0.6,
                zorder=0,
            )

    # ---- Plot waypoints last (on top) ----
    if np.any(waypoint_mask):
        pos = eef_positions[waypoint_mask]
        ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],  # slight offset in Z for waypoints to be on top
            c=color_map["Waypoint"],
            s=80,
            alpha=1.0,
            zorder=2,
        )


    # ---- Legend ----
    legend_elements = [
        Patch(color="pink", label="Interpolate"),
        Patch(color="red", label=f"Waypoints ({num_waypoints})"),
        Patch(color="blue", label="Dense"),
        Patch(color="gray", label="Other"),
    ]
    ax.legend(handles=legend_elements, title="Mode")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

def load_demo_npz(npz_path):
    """
    Load a Hydra demo NPZ file and extract ee positions and modes.

    NOTE:
    - For Waypoint mode, position comes from action[:3]
    - For all other modes, position comes from obs["ee_pos"]
    """
    data = np.load(npz_path, allow_pickle=True)

    # Usually stored under a single unnamed key (e.g. arr_0)
    arr = data[list(data.keys())[0]]

    ee_positions = []
    modes = []
    timesteps = []

    for elem in arr:
        obs = elem["obs"]
        action = elem["action"]

        mode_enum = elem["mode"]
        mode_name = getattr(mode_enum, "name", str(mode_enum))

        timestep = elem.get("rollout_timestep", None)

        # ---- IMPORTANT FIX ----
        if mode_name == "Waypoint":
            ee_pos = np.asarray(action[:3], dtype=float)
        else:
            ee_pos = np.asarray(obs["ee_pos"], dtype=float)

        ee_positions.append(ee_pos)
        modes.append(mode_name)
        timesteps.append(timestep)

    return (
        np.asarray(ee_positions),
        modes,
        np.asarray(timesteps),
    )


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    demo_paths = sorted(
        glob.glob(os.path.join(in_dir, "demo*.npz"))
    )

    if len(demo_paths) == 0:
        print(f"No demo*.npz files found in {in_dir}")
        return

    print(f"Found {len(demo_paths)} demos")

    for demo_path in demo_paths:
        demo_name = os.path.splitext(os.path.basename(demo_path))[0]
        out_path = os.path.join(out_dir, f"{demo_name}.png")

        print(f"Processing {demo_name}")

        ee_positions, modes, timesteps = load_demo_npz(demo_path)

        plot_timesteps_modes_waypoints(
            eef_positions=ee_positions,
            modes=modes,
            out_path=out_path,
            title=demo_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="Directory with demoXXXXX.npz files")
    parser.add_argument("--out_dir", required=True, help="Directory to save PNG plots")

    args = parser.parse_args()
    main(args.in_dir, args.out_dir)
