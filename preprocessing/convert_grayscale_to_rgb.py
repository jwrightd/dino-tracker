import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _get_frame_paths(input_dir: str):
    root = Path(input_dir)
    return sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))


def _is_effectively_grayscale(img: Image.Image) -> bool:
    if img.mode != "RGB":
        return True
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return True
    return np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2])


def convert_dir(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    frames = _get_frame_paths(input_dir)
    if not frames:
        raise FileNotFoundError(f"No frame images found in '{input_dir}'.")

    n_converted = 0
    for frame_path in frames:
        img = Image.open(frame_path)
        if _is_effectively_grayscale(img):
            img = img.convert("RGB")
            n_converted += 1
        img.save(Path(output_dir) / frame_path.name)

    print(f"Converted {n_converted}/{len(frames)} frames to RGB in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    convert_dir(args.input_dir, args.output_dir)
