import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.interpolate import RBFInterpolator

from data.data_utils import save_video
from dinomotion_gray.landmark_dynamics import predict_future_points, train_landmark_dynamics
from dinomotion_gray.sam2_integration import load_masks, points_inside_mask


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f.read()) or {}


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_frame_paths(video_dir: str | Path):
    paths = sorted(list(Path(video_dir).glob("*.png")) + list(Path(video_dir).glob("*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No grayscale frames found in '{video_dir}'.")
    return paths


def load_gray_frames(video_dir: str | Path, max_frames: int | None = None):
    frames = []
    for frame_path in list_frame_paths(video_dir)[:max_frames]:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise FileNotFoundError(f"Could not read frame '{frame_path}'.")
        frames.append(frame)
    return np.stack(frames, axis=0)


def make_color_palette(n: int):
    palette = []
    for idx in range(max(n, 1)):
        hue = idx / max(n, 1)
        color = cv2.cvtColor(
            np.uint8([[[int(hue * 179), 220, 255]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        palette.append((int(color[0]), int(color[1]), int(color[2])))
    return palette


def weighted_linear_forecast(times: np.ndarray, values_xy: np.ndarray, target_times: np.ndarray):
    times = times.astype(np.float64)
    values_xy = values_xy.astype(np.float64)
    target_times = target_times.astype(np.float64)

    if times.shape[0] == 0:
        raise ValueError("Cannot forecast with zero observations.")
    if times.shape[0] == 1:
        return np.repeat(values_xy[:1], target_times.shape[0], axis=0)

    weights = np.linspace(0.5, 1.0, times.shape[0], dtype=np.float64) ** 2
    design = np.stack([times, np.ones_like(times)], axis=1)
    weighted_design = design * weights[:, None]
    preds = []
    for coord_idx in range(values_xy.shape[1]):
        coef, _, _, _ = np.linalg.lstsq(weighted_design, values_xy[:, coord_idx] * weights, rcond=None)
        preds.append(target_times * coef[0] + coef[1])
    return np.stack(preds, axis=1)


def compute_mask_distance_sequence(masks: np.ndarray | None):
    if masks is None:
        return None
    dist_maps = []
    for mask in masks:
        mask_u8 = (mask > 0).astype(np.uint8)
        if mask_u8.max() == 0:
            dist = np.zeros_like(mask_u8, dtype=np.float32)
        else:
            dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3).astype(np.float32)
            if dist.max() > 0:
                dist /= dist.max()
        dist_maps.append(dist)
    return np.stack(dist_maps, axis=0)


def compute_track_scores(tracks: np.ndarray, occlusions: np.ndarray, masks: np.ndarray | None, history: int):
    _, num_frames, _ = tracks.shape
    start_idx = max(0, num_frames - history)
    mask_dist_seq = compute_mask_distance_sequence(masks)
    scores = []
    for point_idx in range(tracks.shape[0]):
        visible = ~occlusions[point_idx, start_idx:]
        if visible.sum() < 2:
            scores.append((-np.inf, 0.0, np.inf, 0.0, 0.0))
            continue
        visible_pts = tracks[point_idx, start_idx:][visible]
        jumps = np.linalg.norm(np.diff(visible_pts, axis=0), axis=-1)
        p90_jump = float(np.percentile(jumps, 90)) if jumps.size else 0.0
        median_jump = float(np.median(jumps)) if jumps.size else 0.0
        visible_ratio = float(visible.mean())
        mask_ratio = 1.0
        mask_margin = 0.0
        if masks is not None:
            visible_times = np.where(visible)[0] + start_idx
            inside = np.array(
                [
                    bool(points_inside_mask(tracks[point_idx, time_idx : time_idx + 1], masks[time_idx])[0])
                    for time_idx in visible_times
                ],
                dtype=bool,
            )
            mask_ratio = float(inside.mean()) if inside.size else 0.0
            if mask_dist_seq is not None and inside.any():
                dvals = []
                for time_idx in visible_times:
                    x = int(np.clip(round(float(tracks[point_idx, time_idx, 0])), 0, mask_dist_seq.shape[2] - 1))
                    y = int(np.clip(round(float(tracks[point_idx, time_idx, 1])), 0, mask_dist_seq.shape[1] - 1))
                    dvals.append(float(mask_dist_seq[time_idx, y, x]))
                if dvals:
                    mask_margin = float(np.median(dvals))
        score = 2.4 * visible_ratio + 2.2 * mask_ratio + 1.0 * mask_margin - 0.05 * p90_jump - 0.04 * median_jump
        scores.append((score, visible_ratio, p90_jump, mask_ratio, mask_margin))
    return scores


def select_anchor_tracks(
    tracks: np.ndarray,
    occlusions: np.ndarray,
    masks: np.ndarray | None,
    history: int,
    max_anchors: int,
    min_visible_ratio: float,
    min_anchor_distance: float,
):
    scores = compute_track_scores(tracks, occlusions, masks, history)
    num_frames = tracks.shape[1]
    last_positions = tracks[:, num_frames - 1]
    last_visible = ~occlusions[:, num_frames - 1]
    last_inside = np.ones(tracks.shape[0], dtype=bool)
    if masks is not None:
        last_inside = points_inside_mask(last_positions, masks[num_frames - 1])

    order = np.argsort([-score[0] for score in scores])
    selected = []
    selected_positions = []

    def can_add(idx: int, required_visible_ratio: float, distance_threshold: float):
        score, visible_ratio, _, mask_ratio, _ = scores[idx]
        if not np.isfinite(score):
            return False
        if not last_visible[idx] or not last_inside[idx]:
            return False
        if visible_ratio < required_visible_ratio:
            return False
        if masks is not None and mask_ratio < 0.55:
            return False
        pos = last_positions[idx]
        if selected_positions:
            dists = np.linalg.norm(np.asarray(selected_positions) - pos[None, :], axis=1)
            if np.any(dists < distance_threshold):
                return False
        return True

    for idx in order:
        if not can_add(idx, min_visible_ratio, min_anchor_distance):
            continue
        score, visible_ratio, p90_jump, mask_ratio, mask_margin = scores[idx]
        pos = last_positions[idx]
        selected.append(
            {
                "index": int(idx),
                "score": float(score),
                "visible_ratio": float(visible_ratio),
                "p90_jump": float(p90_jump),
                "mask_ratio": float(mask_ratio),
                "mask_margin": float(mask_margin),
            }
        )
        selected_positions.append(pos)
        if len(selected) >= max_anchors:
            break

    if len(selected) < 4:
        for relaxed_visible_ratio in (min(0.4, min_visible_ratio), 0.0):
            for idx in order:
                if any(item["index"] == int(idx) for item in selected):
                    continue
                if not can_add(idx, relaxed_visible_ratio, max(8.0, min_anchor_distance * 0.6)):
                    continue
                score, visible_ratio, p90_jump, mask_ratio, mask_margin = scores[idx]
                selected.append(
                    {
                        "index": int(idx),
                        "score": float(score),
                        "visible_ratio": float(visible_ratio),
                        "p90_jump": float(p90_jump),
                        "mask_ratio": float(mask_ratio),
                        "mask_margin": float(mask_margin),
                    }
                )
                selected_positions.append(last_positions[idx])
                if len(selected) >= max(4, max_anchors):
                    break
            if len(selected) >= 4:
                break

    selected_indices = [item["index"] for item in selected]
    return selected_indices, selected


def forecast_future_points(
    tracks: np.ndarray,
    occlusions: np.ndarray,
    selected_indices: list[int],
    history: int,
    horizon: int,
    affine_blend: float,
):
    num_frames = tracks.shape[1]
    last_t = num_frames - 1
    hist_start = max(0, num_frames - history)
    source_points = tracks[selected_indices, last_t].astype(np.float32)
    target_times = np.arange(last_t + 1, last_t + horizon + 1, dtype=np.float64)

    raw_predictions = []
    for track_idx in selected_indices:
        visible_times = np.where(~occlusions[track_idx, hist_start:])[0] + hist_start
        if visible_times.size == 0:
            visible_times = np.array([last_t], dtype=np.int64)
        values = tracks[track_idx, visible_times]
        raw_predictions.append(weighted_linear_forecast(visible_times, values, target_times))
    raw_predictions = np.stack(raw_predictions, axis=0).astype(np.float32)  # N x H x 2

    blended_predictions = []
    affine_matrices = []
    for step_idx in range(horizon):
        target_points = raw_predictions[:, step_idx]
        affine, _ = cv2.estimateAffinePartial2D(
            source_points.astype(np.float32),
            target_points.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=6.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=50,
        )
        if affine is None:
            dxdy = np.median(target_points - source_points, axis=0)
            affine = np.array([[1.0, 0.0, dxdy[0]], [0.0, 1.0, dxdy[1]]], dtype=np.float32)
        affine_matrices.append(affine.astype(np.float32))
        affine_target = np.concatenate(
            [source_points, np.ones((source_points.shape[0], 1), dtype=np.float32)],
            axis=1,
        ) @ affine.T
        blended = affine_blend * affine_target + (1.0 - affine_blend) * target_points
        blended_predictions.append(blended.astype(np.float32))

    return source_points, np.stack(blended_predictions, axis=0), np.stack(affine_matrices, axis=0)


def forecast_future_points_shih_tps(
    tracks: np.ndarray,
    occlusions: np.ndarray,
    selected_indices: list[int],
    history: int,
    horizon: int,
    affine_blend: float,
    image_hw,
    dynamics_cfg: dict,
):
    selected_tracks = tracks[selected_indices]
    selected_occlusions = occlusions[selected_indices]

    model, training_summary = train_landmark_dynamics(
        tracks=selected_tracks,
        occlusions=selected_occlusions,
        history=history,
        horizon=horizon,
        image_hw=image_hw,
        hidden_dim=int(dynamics_cfg.get("hidden_dim", 128)),
        num_layers=int(dynamics_cfg.get("num_layers", 2)),
        dropout=float(dynamics_cfg.get("dropout", 0.1)),
        lr=float(dynamics_cfg.get("lr", 1e-3)),
        weight_decay=float(dynamics_cfg.get("weight_decay", 1e-5)),
        batch_size=int(dynamics_cfg.get("batch_size", 32)),
        steps=int(dynamics_cfg.get("steps", 600)),
        seed=int(dynamics_cfg.get("seed", 7)),
    )
    learned_predictions = predict_future_points(
        model=model,
        tracks=selected_tracks,
        occlusions=selected_occlusions,
        history=history,
        horizon=horizon,
        image_hw=image_hw,
    )  # H x N x 2

    source_points = selected_tracks[:, -1].astype(np.float32)
    blended_predictions = []
    affine_matrices = []
    for step_idx in range(horizon):
        target_points = learned_predictions[step_idx]
        affine, _ = cv2.estimateAffinePartial2D(
            source_points.astype(np.float32),
            target_points.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=6.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=50,
        )
        if affine is None:
            dxdy = np.median(target_points - source_points, axis=0)
            affine = np.array([[1.0, 0.0, dxdy[0]], [0.0, 1.0, dxdy[1]]], dtype=np.float32)
        affine_matrices.append(affine.astype(np.float32))
        affine_target = np.concatenate(
            [source_points, np.ones((source_points.shape[0], 1), dtype=np.float32)],
            axis=1,
        ) @ affine.T
        blended = affine_blend * affine_target + (1.0 - affine_blend) * target_points
        blended_predictions.append(blended.astype(np.float32))

    return source_points, np.stack(blended_predictions, axis=0), np.stack(affine_matrices, axis=0), training_summary


def estimate_background(frames: np.ndarray, masks: np.ndarray, history: int):
    use_frames = frames[-history:]
    use_masks = masks[-history:] > 0
    fallback = frames[-1].astype(np.float32)
    masked_frames = np.where(use_masks, np.nan, use_frames.astype(np.float32))
    background = np.nanmedian(masked_frames, axis=0)
    background = np.where(np.isnan(background), fallback, background)
    return np.clip(background, 0, 255).astype(np.uint8)


def dilate_mask(mask: np.ndarray, kernel_size: int):
    if kernel_size <= 1:
        return (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return (cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0).astype(np.uint8)


def inpaint_background(source_frame: np.ndarray, source_mask: np.ndarray, radius: int = 9):
    return cv2.inpaint(
        source_frame.astype(np.uint8),
        ((source_mask > 0).astype(np.uint8) * 255),
        radius,
        cv2.INPAINT_TELEA,
    )


def add_boundary_anchors(points_src: np.ndarray, points_tgt: np.ndarray, image_hw, border_fraction: float = 0.06):
    h, w = image_hw
    border_x = max(8.0, border_fraction * w)
    border_y = max(8.0, border_fraction * h)
    boundary = np.array(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [0.0, h - 1.0],
            [w - 1.0, h - 1.0],
            [w * 0.5, 0.0],
            [w * 0.5, h - 1.0],
            [0.0, h * 0.5],
            [w - 1.0, h * 0.5],
            [border_x, border_y],
            [w - 1.0 - border_x, border_y],
            [border_x, h - 1.0 - border_y],
            [w - 1.0 - border_x, h - 1.0 - border_y],
        ],
        dtype=np.float32,
    )
    return (
        np.concatenate([points_src.astype(np.float32), boundary], axis=0),
        np.concatenate([points_tgt.astype(np.float32), boundary], axis=0),
    )


def warp_with_tps(
    source_frame: np.ndarray,
    source_mask: np.ndarray,
    background_frame: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    smoothing: float,
    feather_sigma: float,
):
    h, w = source_frame.shape[:2]
    src_aug, tgt_aug = add_boundary_anchors(source_points, target_points, (h, w))
    interpolator = RBFInterpolator(
        tgt_aug.astype(np.float64),
        src_aug.astype(np.float64),
        kernel="thin_plate_spline",
        smoothing=float(smoothing),
    )
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    query = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    src_coords = interpolator(query).reshape(h, w, 2).astype(np.float32)
    map_x = src_coords[..., 0]
    map_y = src_coords[..., 1]

    warped_frame = cv2.remap(
        source_frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.remap(
        source_mask.astype(np.uint8) * 255,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_f = (warped_mask > 127).astype(np.float32)
    if feather_sigma > 0:
        mask_f = cv2.GaussianBlur(mask_f, (0, 0), feather_sigma)
        mask_f = np.clip(mask_f, 0.0, 1.0)
    composite = warped_frame.astype(np.float32) * mask_f + background_frame.astype(np.float32) * (1.0 - mask_f)
    return np.clip(composite, 0, 255).astype(np.uint8), (mask_f > 0.5).astype(np.uint8)


def render_overlay_frames(frames_gray: np.ndarray, predicted_points: np.ndarray):
    colors = make_color_palette(predicted_points.shape[1])
    rendered = []
    for frame_idx in range(frames_gray.shape[0]):
        frame_bgr = cv2.cvtColor(frames_gray[frame_idx], cv2.COLOR_GRAY2BGR)
        for point_idx in range(predicted_points.shape[1]):
            x, y = predicted_points[frame_idx, point_idx]
            cv2.circle(frame_bgr, (int(round(x)), int(round(y))), 4, colors[point_idx], -1, lineType=cv2.LINE_AA)
        rendered.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(rendered, axis=0)


def generate_future_frames(args):
    config = load_config(args.config)
    cfg = dict(config.get("future_generation", {}))
    method = str(cfg.get("method", "shih_tps"))
    history = int(args.history or cfg.get("history", 24))
    horizon = int(args.horizon or cfg.get("horizon", 16))
    min_visible_ratio = float(cfg.get("min_visible_ratio", 0.6))
    min_anchor_distance = float(cfg.get("min_anchor_distance", 20.0))
    max_anchors = int(cfg.get("max_anchors", 12))
    affine_blend = float(cfg.get("affine_blend", 0.8))
    background_history = int(cfg.get("background_history", 32))
    tps_smoothing = float(cfg.get("tps_smoothing", 1e-3))
    feather_sigma = float(cfg.get("feather_sigma", 1.5))
    mask_dilation = int(cfg.get("mask_dilation", 1))
    output_tag = str(cfg.get("output_tag", "tps_future")).strip()

    data_path = Path(args.data_path)
    outputs_dir = data_path / "outputs"
    video_dir = data_path / "video_gray"
    frame_paths = list_frame_paths(video_dir)

    frames = load_gray_frames(video_dir)
    tracks = np.load(outputs_dir / "tracks_smoothed.npy").astype(np.float32)
    occlusions = np.load(outputs_dir / "occlusions.npy").astype(bool)
    mask_dir = data_path / cfg.get("mask_dirname", "masks_sam2")
    masks = None
    if mask_dir.exists():
        _, masks = load_masks(mask_dir, max_frames=frames.shape[0])
        masks = (masks > 0).astype(np.uint8)
        if mask_dilation > 1:
            masks = np.stack([dilate_mask(mask, mask_dilation) for mask in masks], axis=0)

    selected_indices, selected_meta = select_anchor_tracks(
        tracks=tracks,
        occlusions=occlusions,
        masks=masks,
        history=history,
        max_anchors=max_anchors,
        min_visible_ratio=min_visible_ratio,
        min_anchor_distance=min_anchor_distance,
    )
    dynamics_summary = None
    if method == "shih_tps":
        source_points, future_points, affine_matrices, dynamics_summary = forecast_future_points_shih_tps(
            tracks=tracks,
            occlusions=occlusions,
            selected_indices=selected_indices,
            history=history,
            horizon=horizon,
            affine_blend=affine_blend,
            image_hw=frames.shape[1:3],
            dynamics_cfg=dict(cfg.get("dynamics", {})),
        )
    elif method == "linear_tps":
        source_points, future_points, affine_matrices = forecast_future_points(
            tracks=tracks,
            occlusions=occlusions,
            selected_indices=selected_indices,
            history=history,
            horizon=horizon,
            affine_blend=affine_blend,
        )
    else:
        raise ValueError(f"Unsupported future-generation method '{method}'.")

    source_frame = frames[-1]
    source_mask = masks[-1] if masks is not None else np.ones_like(source_frame, dtype=np.uint8)
    temporal_background = estimate_background(frames, masks if masks is not None else np.zeros_like(frames, dtype=np.uint8), background_history)
    inpainted_background = inpaint_background(source_frame, source_mask)
    background_frame = temporal_background.copy()
    background_frame[source_mask > 0] = inpainted_background[source_mask > 0]

    generated_frames = []
    generated_masks = []
    for step_idx in range(horizon):
        frame, frame_mask = warp_with_tps(
            source_frame=source_frame,
            source_mask=source_mask,
            background_frame=background_frame,
            source_points=source_points,
            target_points=future_points[step_idx],
            smoothing=tps_smoothing,
            feather_sigma=feather_sigma,
        )
        generated_frames.append(frame)
        generated_masks.append(frame_mask)

    generated_frames = np.stack(generated_frames, axis=0)
    generated_masks = np.stack(generated_masks, axis=0)

    future_dir = data_path / "future_generation"
    ensure_dir(future_dir)
    frame_dir = future_dir / f"frames_{output_tag}"
    ensure_dir(frame_dir)
    for frame_idx, frame in enumerate(generated_frames, start=1):
        cv2.imwrite(str(frame_dir / f"{frame_idx:05d}.png"), frame)

    np.save(future_dir / f"future_frames_{output_tag}.npy", generated_frames)
    np.save(future_dir / f"future_masks_{output_tag}.npy", generated_masks)
    np.save(future_dir / f"future_points_{output_tag}.npy", future_points)
    np.save(future_dir / f"source_points_{output_tag}.npy", source_points)
    np.save(future_dir / f"affine_matrices_{output_tag}.npy", affine_matrices)

    summary = {
        "paper_choice": "Shih-style landmark prediction + TPS warp" if method == "shih_tps" else "TPSMM-inspired sparse motion warping",
        "method": method,
        "history": history,
        "horizon": horizon,
        "selected_anchor_indices": selected_indices,
        "selected_anchor_meta": selected_meta,
        "output_tag": output_tag,
        "source_frame_index": int(frames.shape[0] - 1),
        "source_frame_path": str(frame_paths[-1]),
    }
    if dynamics_summary is not None:
        summary["dynamics_training"] = {
            "num_samples": int(dynamics_summary.num_samples),
            "final_loss": float(dynamics_summary.final_loss),
            "best_loss": float(dynamics_summary.best_loss),
            "device": dynamics_summary.device,
        }
    with open(future_dir / f"summary_{output_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    video_rgb = np.repeat(generated_frames[..., None], 3, axis=-1)
    overlay_rgb = render_overlay_frames(generated_frames, future_points)
    vis_dir = data_path / "visualizations"
    ensure_dir(vis_dir)
    fps = int(cfg.get("fps", config.get("video", {}).get("vis_fps", 10)))
    save_video(video_rgb, str(vis_dir / f"dinomotion_gray_future_{output_tag}_fps_{fps}.mp4"), fps=fps)
    save_video(overlay_rgb, str(vis_dir / f"dinomotion_gray_future_{output_tag}_overlay_fps_{fps}.mp4"), fps=fps)
    print(f"Saved future-frame outputs to {future_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate future grayscale frames from tracked points without diffusion.")
    parser.add_argument("--config", default="./dinomotion_gray/configs/plane_gray_future.yaml", type=str)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_future_frames(args)


if __name__ == "__main__":
    main()
