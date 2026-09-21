"""Run the NLF multi-person model on the 10demo dataset and export SMPL-X parameters.

For a given person (e.g. 100831), every camera folder under
``<data-root>/<person>/images/cam_XX`` holds 60 frames (0025.jpg ... 1500.jpg,
step 25) of one continuous motion filmed from a fixed camera. This script feeds
each camera's frames to NLF in chronological order and does simple
nearest-box tracking across frames so the same person is followed throughout
the sequence (rather than picking an unrelated detection in a frame that
happens to contain more than one candidate box).

Just run it with no arguments (e.g. click "Run" in your IDE, or
``python run_nlf_smplx.py``) -- all paths below default relative to this
file's location (the repo root), and the NLF checkpoint is downloaded
automatically into ``models/`` on first run. Everything can still be
overridden with CLI flags if you want, see ``--help``.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision  # noqa: F401  (required for the traced model to load correctly)
from torchvision.io import ImageReadMode, read_image

REPO_ROOT = Path(__file__).resolve().parent

# --- Defaults, used when the corresponding CLI flag is omitted ---------------
DEFAULT_DATA_ROOT = REPO_ROOT / '10demo' / 'main'
DEFAULT_PERSON = '100831'
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'nlf_smplx_params'  # independent of the 10demo/ folder
DEFAULT_MODEL_PATH = REPO_ROOT / 'models' / 'nlf_l_multi_0.3.2.torchscript'
DEFAULT_MODEL_URL = (
    'https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript'
)
DEFAULT_BATCH_SIZE = 16
DEFAULT_DETECTOR_THRESHOLD = 0.3
DEFAULT_NUM_AUG = 5
DEFAULT_BETA_REGULARIZER = 10.0

FRAME_RE = re.compile(r'^(\d{4})\.jpg$')

# Slices into the 55-joint SMPL-X axis-angle pose vector (165 = 55 * 3) as
# returned by NLF's `detect_smpl_batched(..., model_name='smplx')`.
N_GLOBAL = 3
N_BODY = 21 * 3
N_JAW = 3
N_LEYE = 3
N_REYE = 3
N_LHAND = 15 * 3
N_RHAND = 15 * 3


def split_smplx_pose(pose: np.ndarray) -> dict:
    i = 0
    parts = {}
    for name, n in [
        ('global_orient', N_GLOBAL),
        ('body_pose', N_BODY),
        ('jaw_pose', N_JAW),
        ('leye_pose', N_LEYE),
        ('reye_pose', N_REYE),
        ('left_hand_pose', N_LHAND),
        ('right_hand_pose', N_RHAND),
    ]:
        parts[name] = pose[i:i + n]
        i += n
    assert i == pose.shape[0], f'expected 165-dim SMPL-X pose, got {pose.shape[0]}'
    return parts


def ensure_model(model_path: Path, model_url: str) -> Path:
    """Download the NLF checkpoint into place if it isn't there yet."""
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Model checkpoint not found at {model_path}.')
    print(f'Downloading it from {model_url} (this is a one-time ~500MB download) ...')

    tmp_path = model_path.with_suffix(model_path.suffix + '.part')

    def report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100 * done / total_size
        sys.stdout.write(f'\r  {done / 1e6:8.1f} / {total_size / 1e6:.1f} MB ({pct:5.1f}%)')
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(model_url, tmp_path, reporthook=report)
        print()
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.rename(model_path)
    print(f'Saved model to {model_path}')
    return model_path


def list_camera_dirs(person_images_dir: Path, cameras_arg: str | None) -> list[Path]:
    all_cams = sorted(
        p for p in person_images_dir.iterdir() if p.is_dir() and p.name.startswith('cam_')
    )
    if not cameras_arg:
        return all_cams
    wanted = set(c.strip() for c in cameras_arg.split(','))
    selected = [p for p in all_cams if p.name in wanted]
    missing = wanted - {p.name for p in selected}
    if missing:
        raise SystemExit(f'Requested camera(s) not found: {sorted(missing)}')
    return selected


def list_frame_paths(camera_dir: Path) -> list[Path]:
    frames = []
    for p in camera_dir.iterdir():
        m = FRAME_RE.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda t: t[0])
    return [p for _, p in frames]


def box_center(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box[:4]
    return np.array([x + w / 2, y + h / 2])


def pick_tracked_index(boxes: np.ndarray, prev_box: np.ndarray | None) -> int:
    """Pick which detected person to keep, favoring continuity with the previous frame."""
    if prev_box is None:
        return int(np.argmax(boxes[:, 4]))  # highest detector score
    prev_c = box_center(prev_box)
    centers = np.stack([box_center(b) for b in boxes])
    dists = np.linalg.norm(centers - prev_c, axis=1)
    return int(np.argmin(dists))


def load_image_batch(paths: list[Path], device: torch.device) -> torch.Tensor:
    images = [read_image(str(p), mode=ImageReadMode.RGB) for p in paths]
    shapes = {tuple(im.shape) for im in images}
    if len(shapes) != 1:
        raise RuntimeError(
            f'Frames in the same camera must share resolution, got shapes {shapes} for {paths}'
        )
    return torch.stack(images).to(device)


def process_camera(
    model,
    frame_paths: list[Path],
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    detector_threshold: float,
    num_aug: int,
    beta_regularizer: float,
    overwrite: bool,
) -> tuple[int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_box = None
    n_saved = 0
    missing = []

    for chunk_start in range(0, len(frame_paths), batch_size):
        chunk_paths = frame_paths[chunk_start:chunk_start + batch_size]

        # Skip frames whose output already exists, unless overwriting -- but we still
        # need to run them through the model if a later frame in the same chunk needs
        # tracking continuity, so only short-circuit whole chunks.
        if not overwrite and all(
            (out_dir / f'{p.stem}.npz').exists() for p in chunk_paths
        ):
            for p in chunk_paths:
                data = np.load(out_dir / f'{p.stem}.npz')
                prev_box = data['box']
            n_saved += len(chunk_paths)
            continue

        frame_batch = load_image_batch(chunk_paths, device)
        with torch.inference_mode():
            pred = model.detect_smpl_batched(
                frame_batch,
                model_name='smplx',
                detector_threshold=detector_threshold,
                num_aug=num_aug,
                beta_regularizer=beta_regularizer,
            )

        for path, boxes_t, pose_t, betas_t, trans_t in zip(
            chunk_paths, pred['boxes'], pred['pose'], pred['betas'], pred['trans']
        ):
            boxes = boxes_t.detach().cpu().numpy()
            if boxes.shape[0] == 0:
                missing.append(path.name)
                continue

            idx = pick_tracked_index(boxes, prev_box)
            prev_box = boxes[idx]

            pose = pose_t[idx].detach().cpu().numpy().astype(np.float32)
            betas = betas_t[idx].detach().cpu().numpy().astype(np.float32)
            trans = trans_t[idx].detach().cpu().numpy().astype(np.float32)
            parts = split_smplx_pose(pose)

            np.savez(
                out_dir / f'{path.stem}.npz',
                pose=pose,
                betas=betas,
                trans=trans,
                box=prev_box.astype(np.float32),
                frame=path.stem,
                **parts,
            )
            n_saved += 1

    return n_saved, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT), help='Path to the 10demo/main folder')
    parser.add_argument('--person', default=DEFAULT_PERSON, help='Person id, e.g. 100831')
    parser.add_argument(
        '--cameras', default=None,
        help='Comma separated camera names to process (default: all cam_00..cam_15 found)',
    )
    parser.add_argument(
        '--model-path', default=str(DEFAULT_MODEL_PATH),
        help='Path to an NLF PyTorch checkpoint (.torchscript) that supports model_name="smplx". '
             'Downloaded automatically if missing.',
    )
    parser.add_argument(
        '--model-url', default=DEFAULT_MODEL_URL,
        help='URL to fetch the checkpoint from if --model-path does not exist yet',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help=f'Default: {DEFAULT_OUTPUT_ROOT}/<person> (kept independent of the 10demo/ folder)',
    )
    parser.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Frames per forward pass')
    parser.add_argument('--detector-threshold', type=float, default=DEFAULT_DETECTOR_THRESHOLD)
    parser.add_argument('--num-aug', type=int, default=DEFAULT_NUM_AUG)
    parser.add_argument('--beta-regularizer', type=float, default=DEFAULT_BETA_REGULARIZER)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    person_dir = data_root / args.person
    images_dir = person_dir / 'images'
    if not images_dir.is_dir():
        raise SystemExit(f'Could not find images dir: {images_dir}')

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / args.person
    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model_path = ensure_model(Path(args.model_path), args.model_url)

    print(f'Loading model from {model_path} onto {device} ...')
    model = torch.jit.load(str(model_path), map_location=device).to(device).eval()

    camera_dirs = list_camera_dirs(images_dir, args.cameras)
    if not camera_dirs:
        raise SystemExit(f'No camera folders found under {images_dir}')

    print(f'Found {len(camera_dirs)} camera(s) for person {args.person}: '
          f'{[c.name for c in camera_dirs]}')

    total_saved = 0
    all_missing = {}
    for camera_dir in camera_dirs:
        frame_paths = list_frame_paths(camera_dir)
        if not frame_paths:
            print(f'  [{camera_dir.name}] no frames found, skipping')
            continue
        print(f'  [{camera_dir.name}] processing {len(frame_paths)} frames '
              f'({frame_paths[0].name} .. {frame_paths[-1].name}) ...')
        n_saved, missing = process_camera(
            model,
            frame_paths,
            output_dir / camera_dir.name,
            device,
            args.batch_size,
            args.detector_threshold,
            args.num_aug,
            args.beta_regularizer,
            args.overwrite,
        )
        total_saved += n_saved
        if missing:
            all_missing[camera_dir.name] = missing
            print(f'  [{camera_dir.name}] WARNING: no person detected in {len(missing)} '
                  f'frame(s): {missing}')

    print(f'Done. Saved {total_saved} SMPL-X parameter files under {output_dir}')
    if all_missing:
        print('Frames with no detection (left unsaved):')
        for cam, frames in all_missing.items():
            print(f'  {cam}: {frames}')


if __name__ == '__main__':
    main()
