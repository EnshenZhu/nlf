"""
Evaluate the deviation (as a percentage) of the first beta value (betas[0])
of a single person's SMPL-X profile, stored under:

    nlf_smplx_params/<person_id>/<cam_xx>/<pose>.npz

Two deviations are computed:

1. Across cameras (for a fixed pose): for each pose file name (e.g. 0025.npz)
   that is shared by all cameras, collect betas[0] from every camera and
   compute the relative deviation (std / mean * 100) across cameras.

2. Across poses (for a fixed camera): for each camera, collect betas[0]
   across all of its pose files (0025.npz .. 1500.npz) and compute the
   relative deviation (std / mean * 100) across poses.

Run directly: `python deviation_eval_single_person.py`
"""

import numpy as np
from pathlib import Path

PERSON_ID = "100831"
ROOT = Path(__file__).parent / "nlf_smplx_params" / PERSON_ID


def relative_deviation_pct(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    cum_error = sum(abs(value - mean) for value in values) / len(values)
    # std = values.std()
    if mean == 0:
        return float("nan")
    return abs(cum_error / mean) * 100.0


def load_beta0(npz_path: Path) -> float:
    with np.load(npz_path) as d:
        return float(d["betas"][0])


def main():
    cam_dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    if not cam_dirs:
        raise SystemExit(f"No camera folders found under {ROOT}")

    # pose_name -> {cam_name: beta0}
    pose_to_cam_beta0 = {}
    # cam_name -> {pose_name: beta0}
    cam_to_pose_beta0 = {}

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        cam_to_pose_beta0[cam_name] = {}
        for npz_path in sorted(cam_dir.glob("*.npz")):
            pose_name = npz_path.stem
            beta0 = load_beta0(npz_path)
            cam_to_pose_beta0[cam_name][pose_name] = beta0
            pose_to_cam_beta0.setdefault(pose_name, {})[cam_name] = beta0

    # ---- Across cameras (same pose, different camera) ----
    print("=" * 70)
    print("Deviation of betas[0] ACROSS CAMERAS (same pose, different camera)")
    print("=" * 70)

    num_cams = len(cam_dirs)
    common_poses = sorted(
        p for p, cams in pose_to_cam_beta0.items() if len(cams) == num_cams
    )
    if len(common_poses) < len(pose_to_cam_beta0):
        missing = len(pose_to_cam_beta0) - len(common_poses)
        print(f"(skipping {missing} pose(s) not present in every camera)")

    cross_cam_devs = []
    for pose_name in common_poses:
        cam_beta0_map = pose_to_cam_beta0[pose_name]
        values = [cam_beta0_map[c.name] for c in cam_dirs]
        dev_pct = relative_deviation_pct(values)
        cross_cam_devs.append(dev_pct)
        print(f"  {pose_name}.npz: mean={np.mean(values): .6f}  "
              f"std={np.std(values): .6f}  deviation={dev_pct:6.3f}%")

    if cross_cam_devs:
        print(f"\n  Average across-camera deviation: "
              f"{np.mean(cross_cam_devs):.3f}%")

    # ---- Across poses (same camera, different pose) ----
    print()
    print("=" * 70)
    print("Deviation of betas[0] ACROSS POSES (same camera, different pose)")
    print("=" * 70)

    cross_pose_devs = []
    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        pose_beta0_map = cam_to_pose_beta0[cam_name]
        values = list(pose_beta0_map.values())
        dev_pct = relative_deviation_pct(values)
        cross_pose_devs.append(dev_pct)
        first_pose = min(pose_beta0_map)
        last_pose = max(pose_beta0_map)
        print(f"  {cam_name} ({first_pose}.npz -> {last_pose}.npz, "
              f"n={len(values)}): mean={np.mean(values): .6f}  "
              f"std={np.std(values): .6f}  deviation={dev_pct:6.3f}%")

    if cross_pose_devs:
        print(f"\n  Average across-pose deviation: "
              f"{np.mean(cross_pose_devs):.3f}%")

    # ---- Overall summary ----
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Person ID: {PERSON_ID}")
    print(f"  Cameras: {num_cams}, Poses per camera: "
          f"{len(next(iter(cam_to_pose_beta0.values())))}")
    if cross_cam_devs:
        print(f"  Mean across-camera deviation: {np.mean(cross_cam_devs):.3f}%")
    if cross_pose_devs:
        print(f"  Mean across-pose deviation:   {np.mean(cross_pose_devs):.3f}%")


if __name__ == "__main__":
    main()
