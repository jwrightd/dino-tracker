from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sam2_masking.sam2_utils import (
    ensure_clean_output_dir,
    get_sam2_device,
    import_sam2_dependencies,
    load_frames,
    load_sam2_config,
    mask_to_box,
    normalize_model_cfg_name,
    save_binary_mask,
    validate_sam2_inputs,
)


def prepare_rgb_proxy_frames(gray_video_dir: str | Path, rgb_video_dir: str | Path, overwrite: bool = False):
    gray_dir = Path(gray_video_dir)
    rgb_dir = Path(rgb_video_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(list(gray_dir.glob("*.png")) + list(gray_dir.glob("*.jpg")))
    if overwrite:
        for existing in sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg"))):
            existing.unlink()
    for frame_path in frame_paths:
        out_path = rgb_dir / f"{frame_path.stem}.jpg"
        if out_path.exists() and not overwrite:
            continue
        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read grayscale frame '{frame_path}'.")
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return rgb_dir


def _build_auto_mask_generator(build_sam2, sam2_automatic_mask_generator_cls, config, device):
    image_model = build_sam2(
        config["model_cfg"],
        config["checkpoint"],
        device=device,
        apply_postprocessing=config.get("apply_postprocessing", False),
    )
    return sam2_automatic_mask_generator_cls(
        image_model,
        points_per_side=config.get("points_per_side", 32),
        pred_iou_thresh=config.get("pred_iou_thresh", 0.8),
        stability_score_thresh=config.get("stability_score_thresh", 0.92),
        min_mask_region_area=config["min_mask_region_area"],
    )


def _build_video_predictor(build_sam2_video_predictor, config, device):
    return build_sam2_video_predictor(
        config["model_cfg"],
        config["checkpoint"],
        device=device,
    )


def _add_bbox_prompt(predictor, inference_state, bbox):
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=np.asarray(bbox, dtype=np.float32),
    )


def _add_point_prompt(predictor, inference_state, point):
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=np.asarray([point], dtype=np.float32),
        labels=np.asarray([1], dtype=np.int32),
    )


def _propagate_and_save(predictor, inference_state, frame_paths, output_dir):
    count = 0
    for frame_idx, _, mask_logits in predictor.propagate_in_video(inference_state):
        mask = (mask_logits[0] > 0.0).squeeze().detach().cpu().numpy()
        out_name = Path(frame_paths[frame_idx]).stem + ".png"
        save_binary_mask(mask, Path(output_dir) / out_name)
        count += 1
    return count


def generate_sam2_masks(video_dir: str | Path, output_dir: str | Path, config_path: str, mode: str = "bbox", bbox=None, point=None, overwrite: bool = False):
    config = load_sam2_config(config_path)
    config["model_cfg"] = normalize_model_cfg_name(config["model_cfg"])
    validate_sam2_inputs(config["checkpoint"], config["model_cfg"])
    build_sam2, build_sam2_video_predictor, sam2_automatic_mask_generator_cls = import_sam2_dependencies()
    device = get_sam2_device(log=True)

    frames, frame_paths = load_frames(str(video_dir), grayscale_to_rgb=True)
    config["min_mask_region_area"] = int(config.get("min_mask_area_ratio", 0.01) * frames[0].shape[0] * frames[0].shape[1])

    output_dir = Path(output_dir)
    if overwrite:
        ensure_clean_output_dir(str(output_dir))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "auto":
        mask_generator = _build_auto_mask_generator(build_sam2, sam2_automatic_mask_generator_cls, config, device)
        first_frame_masks = mask_generator.generate(frames[0])
        if not first_frame_masks:
            raise RuntimeError("SAM2 auto mode found no masks on frame 0.")
        first_frame_masks.sort(key=lambda item: item["area"], reverse=True)
        bbox = mask_to_box(first_frame_masks[0]["segmentation"])
        predictor = _build_video_predictor(build_sam2_video_predictor, config, device)
        inference_state = predictor.init_state(video_path=str(Path(frame_paths[0]).parent))
        with torch.inference_mode():
            _add_bbox_prompt(predictor, inference_state, bbox)
            _propagate_and_save(predictor, inference_state, frame_paths, output_dir)
        return output_dir

    predictor = _build_video_predictor(build_sam2_video_predictor, config, device)
    inference_state = predictor.init_state(video_path=str(Path(frame_paths[0]).parent))
    with torch.inference_mode():
        if mode == "bbox":
            if bbox is None:
                raise ValueError("bbox prompt is required for SAM2 bbox mode.")
            _add_bbox_prompt(predictor, inference_state, bbox)
        elif mode == "point":
            if point is None:
                raise ValueError("point prompt is required for SAM2 point mode.")
            _add_point_prompt(predictor, inference_state, point)
        else:
            raise ValueError(f"Unsupported SAM2 mode '{mode}'.")
        _propagate_and_save(predictor, inference_state, frame_paths, output_dir)
    return output_dir


def load_masks(mask_dir: str | Path, resize_hw=None, max_frames=None, device=None):
    mask_paths = sorted(list(Path(mask_dir).glob("*.png")) + list(Path(mask_dir).glob("*.jpg")))
    if max_frames is not None:
        mask_paths = mask_paths[:max_frames]
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in '{mask_dir}'.")

    tensors = []
    arrays = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask '{mask_path}'.")
        arrays.append(mask)
        if resize_hw is not None:
            resize_h, resize_w = resize_hw
            mask = cv2.resize(mask, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        tensors.append(torch.from_numpy((mask > 0).astype(np.float32)))

    masks_tensor = torch.stack(tensors, dim=0)
    if device is not None:
        masks_tensor = masks_tensor.to(device)
    return masks_tensor, np.stack(arrays, axis=0)


def mask_feature_map(mask_bhw: torch.Tensor, feature_hw):
    return F.interpolate(mask_bhw.unsqueeze(1), size=feature_hw, mode="nearest").squeeze(1)


def points_inside_mask(points_xy: np.ndarray, mask_hw: np.ndarray):
    h, w = mask_hw.shape[:2]
    xs = np.round(points_xy[..., 0]).astype(np.int32)
    ys = np.round(points_xy[..., 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    inside = np.zeros(points_xy.shape[:-1], dtype=bool)
    inside[valid] = mask_hw[ys[valid], xs[valid]] > 0
    return inside
