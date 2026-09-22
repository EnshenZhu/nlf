"""
Evaluate the deviation (as a percentage) of every beta value (betas[0..N-1])
of a single person's SMPL-X profile, stored under:

    nlf_smplx_params/<person_id>/<cam_xx>/<pose>.npz

For each beta index, two deviations are computed:

1. Across cameras (for a fixed pose): for each pose file name (e.g. 0025.npz)
   that is shared by all cameras, collect betas[i] from every camera and
   compute the relative deviation (cum_error / mean * 100) across cameras,
   then average that over all poses.

2. Across poses (for a fixed camera): for each camera, collect betas[i]
   across all of its pose files (0025.npz .. 1500.npz) and compute the
   relative deviation (cum_error / mean * 100) across poses, then average
   that over all cameras.

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
    if mean == 0:
        return float("nan")
    return abs(cum_error / mean) * 100.0


def load_betas(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as d:
        return np.asarray(d["betas"], dtype=np.float64)


def main():
    cam_dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    if not cam_dirs:
        raise SystemExit(f"No camera folders found under {ROOT}")

    # pose_name -> {cam_name: betas}
    pose_to_cam_betas = {}
    # cam_name -> {pose_name: betas}
    cam_to_pose_betas = {}

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        cam_to_pose_betas[cam_name] = {}
        for npz_path in sorted(cam_dir.glob("*.npz")):
            pose_name = npz_path.stem
            betas = load_betas(npz_path)
            cam_to_pose_betas[cam_name][pose_name] = betas
            pose_to_cam_betas.setdefault(pose_name, {})[cam_name] = betas

    num_betas = len(next(iter(next(iter(cam_to_pose_betas.values())).values())))
    num_cams = len(cam_dirs)

    common_poses = sorted(
        p for p, cams in pose_to_cam_betas.items() if len(cams) == num_cams
    )
    if len(common_poses) < len(pose_to_cam_betas):
        missing = len(pose_to_cam_betas) - len(common_poses)
        print(f"(skipping {missing} pose(s) not present in every camera)")

    per_beta_cross_cam_avg = []
    per_beta_cross_pose_avg = []

    for beta_id in range(num_betas):
        # ---- Across cameras (same pose, different camera) ----
        cross_cam_devs = []
        for pose_name in common_poses:
            cam_beta_map = pose_to_cam_betas[pose_name]
            values = [cam_beta_map[c.name][beta_id] for c in cam_dirs]
            cross_cam_devs.append(relative_deviation_pct(values))

        # ---- Across poses (same camera, different pose) ----
        cross_pose_devs = []
        for cam_dir in cam_dirs:
            pose_beta_map = cam_to_pose_betas[cam_dir.name]
            values = [betas[beta_id] for betas in pose_beta_map.values()]
            cross_pose_devs.append(relative_deviation_pct(values))

        per_beta_cross_cam_avg.append(np.mean(cross_cam_devs) if cross_cam_devs else float("nan"))
        per_beta_cross_pose_avg.append(np.mean(cross_pose_devs) if cross_pose_devs else float("nan"))

    # ---- Summary table ----
    print("=" * 70)
    print("NLF Deviation Summary (per beta index)")
    print("=" * 70)
    print(f"  Person ID: {PERSON_ID}")
    print(f"  Cameras: {num_cams}, Common poses: {len(common_poses)}")
    print("-" * 70)
    print(f"  {'Beta':>4}  {'Avg across-camera dev %':>24}  {'Avg across-pose dev %':>22}")
    print("-" * 70)
    for beta_id in range(num_betas):
        print(f"  {beta_id:>4}  {per_beta_cross_cam_avg[beta_id]:>24.3f}  "
              f"{per_beta_cross_pose_avg[beta_id]:>22.3f}")
    print("-" * 70)
    print(f"  {'mean':>4}  {np.mean(per_beta_cross_cam_avg):>24.3f}  "
          f"{np.mean(per_beta_cross_pose_avg):>22.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
