import os
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from device_utils import get_device


def load_sam2_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read()) or {}
    return config


def get_sorted_frame_paths(video_dir: str):
    root = Path(video_dir)
    frame_paths = sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in '{video_dir}'.")
    return frame_paths


def load_frames(video_dir: str, grayscale_to_rgb: bool = True):
    frames = []
    frame_paths = get_sorted_frame_paths(video_dir)
    for frame_path in frame_paths:
        img = Image.open(frame_path)
        if grayscale_to_rgb and img.mode != "RGB":
            img = img.convert("RGB")
        frames.append(np.array(img))
    return frames, frame_paths


def get_sam2_device(log: bool = True):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return get_device(log=log)


def import_sam2_dependencies():
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 is required for the optimized pipeline. Install it separately, for example "
            "with `pip install sam2`, and ensure the checkpoint is available."
        ) from exc
    return build_sam2, build_sam2_video_predictor, SAM2AutomaticMaskGenerator


def validate_sam2_inputs(checkpoint_path: str, model_cfg: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at '{checkpoint_path}'. Download the checkpoint before running the optimized pipeline."
        )
    if not model_cfg:
        raise ValueError("SAM2 model_cfg must be provided.")


def normalize_model_cfg_name(model_cfg: str):
    if model_cfg.startswith("configs/"):
        return model_cfg
    if model_cfg.startswith("sam2.1_"):
        return f"configs/sam2.1/{model_cfg}"
    if model_cfg.startswith("sam2_"):
        return f"configs/sam2/{model_cfg}"
    return model_cfg


def ensure_clean_output_dir(output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for existing_mask in sorted(list(output_path.glob("*.png")) + list(output_path.glob("*.jpg"))):
        existing_mask.unlink()


def mask_to_box(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Could not derive a bounding box from the selected SAM2 mask.")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def save_binary_mask(mask: np.ndarray, output_path: Path):
    binary_mask = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(binary_mask).save(output_path)
