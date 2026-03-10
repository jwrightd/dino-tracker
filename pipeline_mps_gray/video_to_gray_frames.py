import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def _list_frame_files(path: Path):
    return sorted(list(path.glob("*.jpg")) + list(path.glob("*.png")))


def video_to_gray_frames(
    video_path: str,
    output_folder: str,
    overwrite: bool = False,
    max_frames: int | None = None,
    single_channel: bool = True,
):
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    existing_frames = _list_frame_files(output_path)
    if existing_frames:
        if not overwrite:
            raise RuntimeError(
                f"Found {len(existing_frames)} existing frame files in '{output_folder}'. "
                "Pass --overwrite to replace them."
            )
        for frame_file in existing_frames:
            frame_file.unlink()

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

    written = 0
    progress = tqdm(total=total_frames if total_frames > 0 else None, desc="Converting to grayscale frames")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames is not None and written >= max_frames:
                break

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            out_file = output_path / f"{written:05d}.jpg"
            out_frame = gray if single_channel else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if not cv2.imwrite(str(out_file), out_frame):
                raise RuntimeError(f"Failed writing frame: {out_file}")

            written += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()

    if written == 0:
        raise RuntimeError("No frames were extracted from the input video.")

    return written


def main():
    parser = argparse.ArgumentParser(description="Convert a video to grayscale JPG frames.")
    parser.add_argument("--video-path", required=True, type=str, help="Path to input video file.")
    parser.add_argument("--output-folder", required=True, type=str, help="Output folder for extracted frames.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Delete existing .jpg/.png frames in output folder before writing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap while extracting grayscale frames.",
    )
    parser.add_argument(
        "--three-channel",
        action="store_true",
        default=False,
        help="Write 3-channel grayscale frames instead of single-channel grayscale.",
    )
    args = parser.parse_args()

    written = video_to_gray_frames(
        video_path=args.video_path,
        output_folder=args.output_folder,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
        single_channel=not args.three_channel,
    )
    print(f"Saved {written} grayscale frames to {os.path.abspath(args.output_folder)}")


if __name__ == "__main__":
    main()
