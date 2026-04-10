import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing.convert_grayscale_to_rgb import convert_dir
from preprocessing.mp4_to_frames import mp4_to_frames


def _frame_paths(path: Path):
    return sorted(list(path.glob("*.png")) + list(path.glob("*.jpg")))


def _clear_existing_frames(path: Path):
    for frame_path in _frame_paths(path):
        frame_path.unlink()


def detect_frame_mode(frames_dir: str) -> str:
    frame_paths = _frame_paths(Path(frames_dir))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in '{frames_dir}' for input-mode detection.")

    img = Image.open(frame_paths[0])
    if img.mode != "RGB":
        return "grayscale"

    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return "grayscale"
    if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
        return "grayscale"
    return "rgb"


def prepare_video_frames(video_path: str, output_dir: str, input_mode: str = "auto", overwrite: bool = False, max_frames=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing_frames = _frame_paths(output_path)
    if existing_frames:
        if not overwrite:
            raise RuntimeError(
                f"Found {len(existing_frames)} existing frames in '{output_dir}'. Pass --overwrite to replace them."
            )
        _clear_existing_frames(output_path)

    mp4_to_frames(video_path, output_dir, max_frames=max_frames)

    resolved_mode = input_mode
    if input_mode == "auto":
        resolved_mode = detect_frame_mode(output_dir)

    if resolved_mode == "grayscale":
        print("Detected grayscale input; converting frames to RGB for SAM2 / RAFT / DINO.", flush=True)
        convert_dir(output_dir, output_dir)
    else:
        print("Detected RGB input; leaving extracted frames unchanged.", flush=True)

    return resolved_mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames and conditionally convert grayscale footage to RGB.")
    parser.add_argument("--video-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--input-mode", choices=["auto", "rgb", "grayscale"], default="auto")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    resolved_mode = prepare_video_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
    )
    print(f"Prepared frames in {args.output_dir} (resolved input mode: {resolved_mode})", flush=True)
