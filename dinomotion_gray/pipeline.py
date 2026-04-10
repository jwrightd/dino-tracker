import argparse
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.signal import savgol_filter
from tqdm import tqdm

from device_utils import get_device
from device_utils import grid_sample_border_safe
from data.data_utils import save_video
from dinomotion_gray.grayscale_dinov2 import GrayscaleDINOv2
from dinomotion_gray.sam2_integration import (
    generate_sam2_masks,
    load_masks,
    points_inside_mask,
    prepare_rgb_proxy_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f.read()) or {}


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str | Path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def resolve_sam2_settings(config, args):
    seg_cfg = dict(config.get("segmentation", {}))
    mode = getattr(args, "sam2_mode", None) or seg_cfg.get("mode", "bbox")
    enabled = bool(seg_cfg.get("enabled", True)) and mode != "off"
    return {
        "enabled": enabled,
        "mode": mode,
        "config_path": getattr(args, "sam2_config", None) or seg_cfg.get("config_path", "./sam2_masking/sam2_config.yaml"),
        "mask_dirname": seg_cfg.get("mask_dirname", "masks_sam2"),
        "rgb_proxy_dirname": seg_cfg.get("rgb_proxy_dirname", "video_sam2_rgb"),
        "point": tuple(getattr(args, "sam2_point", None)) if getattr(args, "sam2_point", None) is not None else None,
    }


def extract_grayscale_frames(video_path: str, output_dir: str, max_frames: int | None = None, overwrite: bool = False, frame_ext: str = "png"):
    output_path = Path(output_dir)
    ensure_dir(output_path)
    if overwrite:
        for existing in sorted(output_path.glob(f"*.{frame_ext}")):
            existing.unlink()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video '{video_path}'.")

    written = 0
    progress = tqdm(total=max_frames, desc="Extracting grayscale frames")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            out_path = output_path / f"{written:05d}.{frame_ext}"
            cv2.imwrite(str(out_path), gray)
            written += 1
            progress.update(1)
            if max_frames is not None and written >= max_frames:
                break
    finally:
        progress.close()
        cap.release()
    return written


def list_frame_paths(video_dir: str):
    paths = sorted(list(Path(video_dir).glob("*.png")) + list(Path(video_dir).glob("*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No grayscale frames found in '{video_dir}'.")
    return paths


def load_gray_frame(path: str | Path, resize_hw=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read frame '{path}'.")
    if resize_hw is not None:
        resize_h, resize_w = resize_hw
        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    return tensor, img


def load_gray_video_stack(video_dir: str, resize_hw=None, max_frames=None):
    frames = []
    raw_frames = []
    for frame_path in list_frame_paths(video_dir)[:max_frames]:
        tensor, raw = load_gray_frame(frame_path, resize_hw=resize_hw)
        frames.append(tensor)
        raw_frames.append(raw)
    return torch.stack(frames), np.stack(raw_frames)


def resize_bbox(bbox_xyxy, from_hw, to_hw):
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    scale_x = to_w / float(from_w)
    scale_y = to_h / float(from_h)
    x1, y1, x2, y2 = bbox_xyxy
    return [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    ]


def bbox_to_mask(bbox_xyxy, hw):
    h, w = hw
    x1, y1, x2, y2 = bbox_xyxy
    mask = torch.zeros((h, w), dtype=torch.float32)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    mask[y1:y2, x1:x2] = 1.0
    return mask


def normalize_minmax(x, eps=1e-6):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def mask_distance_map(mask_b1hw: torch.Tensor):
    masks_np = (mask_b1hw.detach().cpu().numpy() > 0.5).astype(np.uint8)
    dist_maps = []
    for mask in masks_np[:, 0]:
        if mask.max() == 0:
            dist = np.zeros_like(mask, dtype=np.float32)
        else:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float32)
            if dist.max() > 0:
                dist /= dist.max()
        dist_maps.append(dist)
    dist_t = torch.from_numpy(np.stack(dist_maps, axis=0)).to(mask_b1hw.device, dtype=mask_b1hw.dtype)
    return dist_t.unsqueeze(1)


def image_to_feature_points(points_xy, image_hw, feature_hw):
    img_h, img_w = image_hw
    feat_h, feat_w = feature_hw
    x = points_xy[..., 0] / max(img_w - 1, 1) * max(feat_w - 1, 1)
    y = points_xy[..., 1] / max(img_h - 1, 1) * max(feat_h - 1, 1)
    return torch.stack([x, y], dim=-1)


def feature_to_image_points(points_xy, image_hw, feature_hw):
    img_h, img_w = image_hw
    feat_h, feat_w = feature_hw
    x = points_xy[..., 0] / max(feat_w - 1, 1) * max(img_w - 1, 1)
    y = points_xy[..., 1] / max(feat_h - 1, 1) * max(img_h - 1, 1)
    return torch.stack([x, y], dim=-1)


def sample_feature_vectors(feature_map: torch.Tensor, points_xy: torch.Tensor):
    b, c, h, w = feature_map.shape
    points_norm = points_xy.clone()
    points_norm[..., 0] = points_xy[..., 0] / max(w - 1, 1) * 2 - 1
    points_norm[..., 1] = points_xy[..., 1] / max(h - 1, 1) * 2 - 1
    grid = points_norm.view(b, 1, -1, 2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True)
    return sampled.squeeze(2).permute(0, 2, 1).contiguous()


def select_diverse_points(points_xy: torch.Tensor, scores: torch.Tensor, num_points: int, min_distance: float):
    if points_xy.shape[0] == 0:
        raise ValueError("No candidate points available for selection.")

    num_points = min(int(num_points), int(points_xy.shape[0]))
    selected = []
    score_norm = scores.clone()
    if score_norm.numel() > 1:
        score_norm = (score_norm - score_norm.min()) / (score_norm.max() - score_norm.min() + 1e-6)
    else:
        score_norm = torch.ones_like(score_norm)

    min_dists = torch.full((points_xy.shape[0],), float("inf"), device=points_xy.device, dtype=points_xy.dtype)
    chosen_mask = torch.zeros(points_xy.shape[0], dtype=torch.bool, device=points_xy.device)

    # First pass: enforce a hard spacing threshold where possible.
    while len(selected) < num_points:
        diversity = torch.clamp(min_dists / max(min_distance, 1e-3), max=2.0)
        combined = score_norm + 0.75 * diversity
        combined = combined.masked_fill(chosen_mask, -1e6)
        if selected:
            combined = combined.masked_fill(min_dists < min_distance, -1e6)
        best_idx = int(torch.argmax(combined).item())
        if combined[best_idx] <= -1e5:
            break
        selected.append(best_idx)
        chosen_mask[best_idx] = True
        cur_dists = torch.norm(points_xy - points_xy[best_idx : best_idx + 1], dim=-1)
        min_dists = torch.minimum(min_dists, cur_dists)

    # Second pass: still prefer spread, but fill any remaining slots.
    while len(selected) < num_points:
        diversity = torch.clamp(min_dists / max(min_distance, 1e-3), max=2.0)
        combined = score_norm + 0.75 * diversity
        combined = combined.masked_fill(chosen_mask, -1e6)
        best_idx = int(torch.argmax(combined).item())
        if combined[best_idx] <= -1e5:
            break
        selected.append(best_idx)
        chosen_mask[best_idx] = True
        cur_dists = torch.norm(points_xy - points_xy[best_idx : best_idx + 1], dim=-1)
        min_dists = torch.minimum(min_dists, cur_dists)

    return torch.as_tensor(selected, device=points_xy.device, dtype=torch.long)


def similarity_soft_argmax(similarity_maps: torch.Tensor, temperature: float):
    b, k, h, w = similarity_maps.shape
    flat = (similarity_maps / max(temperature, 1e-6)).reshape(b, k, -1)
    weights = torch.softmax(flat, dim=-1).reshape(b, k, h, w)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=similarity_maps.device, dtype=similarity_maps.dtype),
        torch.arange(w, device=similarity_maps.device, dtype=similarity_maps.dtype),
        indexing="ij",
    )
    x = (weights * grid_x.view(1, 1, h, w)).sum(dim=(-2, -1))
    y = (weights * grid_y.view(1, 1, h, w)).sum(dim=(-2, -1))
    return torch.stack([x, y], dim=-1), weights


def estimate_affine_theta(template_points_xy: torch.Tensor, moving_points_xy: torch.Tensor, image_hw):
    img_h, img_w = image_hw
    template_norm = template_points_xy.clone()
    moving_norm = moving_points_xy.clone()
    template_norm[..., 0] = template_points_xy[..., 0] / max(img_w - 1, 1) * 2 - 1
    template_norm[..., 1] = template_points_xy[..., 1] / max(img_h - 1, 1) * 2 - 1
    moving_norm[..., 0] = moving_points_xy[..., 0] / max(img_w - 1, 1) * 2 - 1
    moving_norm[..., 1] = moving_points_xy[..., 1] / max(img_h - 1, 1) * 2 - 1

    ones = torch.ones_like(template_norm[..., :1])
    template_aug = torch.cat([template_norm, ones], dim=-1)
    solutions = []
    for batch_idx in range(template_aug.shape[0]):
        # affine_grid expects a mapping from output(template) coords to input(moving) coords.
        sol = torch.linalg.lstsq(template_aug[batch_idx], moving_norm[batch_idx]).solution
        solutions.append(sol.T)
    return torch.stack(solutions, dim=0)


def warp_moving_affine(moving_image: torch.Tensor, theta: torch.Tensor):
    grid = F.affine_grid(theta, moving_image.shape, align_corners=True)
    return grid_sample_border_safe(moving_image, grid, mode="bilinear", align_corners=True)


def masked_mse_loss(template: torch.Tensor, registered: torch.Tensor, mask: torch.Tensor | None):
    if mask is None:
        return F.mse_loss(registered, template)
    while mask.ndim < template.ndim:
        mask = mask.unsqueeze(1)
    diff = (registered - template) ** 2 * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def make_color_palette(n):
    rng = np.random.default_rng(7)
    palette = []
    for idx in range(n):
        hue = idx / max(n, 1)
        color = cv2.cvtColor(
            np.uint8([[[int(hue * 179), 220, 255]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        color = (int(color[0]), int(color[1]), int(color[2]))
        if idx > 0:
            palette.append(color)
        else:
            palette.append((0, 255, 255))
    rng.shuffle(palette)
    return palette


class DINOMotionGrayModel(torch.nn.Module):
    def __init__(self, config, device, torch_home=None):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.encoder = GrayscaleDINOv2(
            model_name=config["model"]["dino_model_name"],
            lora_rank=config["model"]["lora_rank"],
            lora_alpha=float(config["model"]["lora_alpha"]),
            train_patch_embed=bool(config["model"]["train_patch_embed"]),
            torch_home=torch_home,
        )
        self.temperature = float(config["tracking"]["softmax_temperature"])
        self.num_landmarks = int(config["tracking"]["num_landmarks"])
        self.nms_kernel_size = int(config["tracking"]["nms_kernel_size"])
        self.candidate_multiplier = int(config["tracking"].get("candidate_multiplier", 6))
        self.min_landmark_distance = float(config["tracking"].get("min_landmark_distance", 12.0))
        self.interior_weight = float(config["tracking"].get("interior_weight", 0.35))
        self.grad_weight = float(config["tracking"].get("grad_weight", 0.5))
        self.feature_weight = float(config["tracking"].get("feature_weight", 0.5))

    def trainable_parameters(self):
        return self.encoder.trainable_parameters()

    def _select_masked_landmarks(
        self,
        saliency_peaks: torch.Tensor,
        saliency_raw: torch.Tensor,
        mask_small: torch.Tensor,
        image_hw,
        feature_hw,
    ):
        batch_size, _, feat_h, feat_w = saliency_raw.shape
        mask_flat = (mask_small.flatten(2) > 0.5)
        peaks_flat = saliency_peaks.flatten(2)
        raw_flat = saliency_raw.flatten(2)
        selected_points = []

        for batch_idx in range(batch_size):
            valid_idx = torch.nonzero(mask_flat[batch_idx, 0], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                valid_idx = torch.arange(feat_h * feat_w, device=saliency_raw.device)

            peak_candidates = valid_idx[peaks_flat[batch_idx, 0, valid_idx] > 0]
            candidate_pool = min(max(self.num_landmarks * self.candidate_multiplier, self.num_landmarks), valid_idx.numel())

            if peak_candidates.numel() > 0:
                peak_scores = peaks_flat[batch_idx, 0, peak_candidates]
                peak_order = torch.argsort(peak_scores, descending=True)
                selected_seed = peak_candidates[peak_order[:candidate_pool]]
            else:
                fallback_scores = raw_flat[batch_idx, 0, valid_idx]
                fallback_order = torch.argsort(fallback_scores, descending=True)
                selected_seed = valid_idx[fallback_order[:candidate_pool]]

            if selected_seed.numel() < candidate_pool:
                fallback_scores = raw_flat[batch_idx, 0, valid_idx]
                fallback_order = torch.argsort(fallback_scores, descending=True)
                supplement = valid_idx[fallback_order[:candidate_pool]]
                selected_seed = torch.unique(torch.cat([selected_seed, supplement], dim=0), sorted=False)[:candidate_pool]

            if selected_seed.numel() == 0:
                selected_seed = valid_idx[:1]

            xs = (selected_seed % feat_w).float()
            ys = (selected_seed // feat_w).float()
            candidate_points_feat = torch.stack([xs, ys], dim=-1)
            candidate_points_img = feature_to_image_points(candidate_points_feat, image_hw, feature_hw)
            candidate_scores = raw_flat[batch_idx, 0, selected_seed]
            selected_local = select_diverse_points(
                points_xy=candidate_points_img,
                scores=candidate_scores,
                num_points=min(self.num_landmarks, candidate_points_img.shape[0]),
                min_distance=self.min_landmark_distance,
            )
            points_img = candidate_points_img[selected_local]
            if points_img.shape[0] < self.num_landmarks:
                repeats = int(math.ceil(self.num_landmarks / float(points_img.shape[0])))
                jitter_base = points_img.repeat(repeats, 1)[: self.num_landmarks]
                jitter = torch.zeros_like(jitter_base)
                if points_img.shape[0] > 0:
                    jitter[:, 0] = torch.linspace(-0.75, 0.75, steps=jitter_base.shape[0], device=jitter_base.device, dtype=jitter_base.dtype)
                    jitter[:, 1] = torch.linspace(0.75, -0.75, steps=jitter_base.shape[0], device=jitter_base.device, dtype=jitter_base.dtype)
                points_img = jitter_base + jitter
            else:
                points_img = points_img[: self.num_landmarks]
            selected_points.append(points_img)

        return torch.stack(selected_points, dim=0)

    def extract_landmarks(self, template_image: torch.Tensor, template_features: torch.Tensor, template_mask: torch.Tensor):
        feat_norm = template_features.norm(dim=1, keepdim=True)
        grad_x = F.pad(template_image[..., :, 1:] - template_image[..., :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(template_image[..., 1:, :] - template_image[..., :-1, :], (0, 0, 0, 1))
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad_small = F.interpolate(grad_mag, size=template_features.shape[-2:], mode="bilinear", align_corners=False)
        mask_small = F.interpolate(template_mask.unsqueeze(1), size=template_features.shape[-2:], mode="nearest")
        mask_distance_small = mask_distance_map(mask_small)

        feature_term = normalize_minmax(feat_norm) * self.feature_weight
        grad_term = normalize_minmax(grad_small) * self.grad_weight
        interior_term = mask_distance_small * self.interior_weight
        saliency_raw = (feature_term + grad_term + interior_term) * mask_small
        pooled = F.max_pool2d(saliency_raw, kernel_size=self.nms_kernel_size, stride=1, padding=self.nms_kernel_size // 2)
        is_peak = (saliency_raw == pooled) & (mask_small > 0)
        saliency_peaks = saliency_raw * is_peak
        return self._select_masked_landmarks(
            saliency_peaks=saliency_peaks,
            saliency_raw=saliency_raw,
            mask_small=mask_small,
            image_hw=template_image.shape[-2:],
            feature_hw=template_features.shape[-2:],
        )

    def match_landmarks(
        self,
        template_features: torch.Tensor,
        moving_features: torch.Tensor,
        template_points_img: torch.Tensor,
        image_hw,
        moving_mask: torch.Tensor | None = None,
    ):
        template_points_feat = image_to_feature_points(
            template_points_img,
            image_hw,
            template_features.shape[-2:],
        )
        template_vectors = sample_feature_vectors(template_features, template_points_feat)
        template_vectors = F.normalize(template_vectors, dim=-1)
        moving_features_norm = F.normalize(moving_features, dim=1)
        similarity = torch.einsum("bkc,bchw->bkhw", template_vectors, moving_features_norm)
        if moving_mask is not None:
            mask_small = F.interpolate(
                moving_mask.unsqueeze(1),
                size=moving_features.shape[-2:],
                mode="nearest",
            ).squeeze(1) > 0.5
            valid_batches = mask_small.flatten(1).any(dim=1)
            if valid_batches.any():
                similarity = similarity.clone()
                similarity[valid_batches] = similarity[valid_batches].masked_fill(
                    ~mask_small[valid_batches].unsqueeze(1),
                    -1e4,
                )
        points_feat, weights = similarity_soft_argmax(similarity, self.temperature)
        moving_points_img = feature_to_image_points(
            points_feat,
            image_hw,
            moving_features.shape[-2:],
        )
        moving_vectors = sample_feature_vectors(moving_features_norm, points_feat)
        moving_vectors = F.normalize(moving_vectors, dim=-1)
        return moving_points_img, similarity, weights, template_vectors, moving_vectors

    def forward(self, template_image: torch.Tensor, moving_image: torch.Tensor, template_mask: torch.Tensor, moving_mask: torch.Tensor | None = None):
        template_features = self.encoder.forward_feature_map(template_image)
        moving_features = self.encoder.forward_feature_map(moving_image)
        template_points = self.extract_landmarks(template_image, template_features, template_mask)
        moving_points, similarity, _, template_vectors, moving_vectors = self.match_landmarks(
            template_features,
            moving_features,
            template_points,
            template_image.shape[-2:],
            moving_mask=moving_mask,
        )
        theta = estimate_affine_theta(template_points, moving_points, template_image.shape[-2:])
        registered = warp_moving_affine(moving_image, theta)

        return {
            "template_points": template_points,
            "moving_points": moving_points,
            "registered": registered,
            "theta": theta,
            "similarity": similarity,
            "template_vectors": template_vectors,
            "moving_vectors": moving_vectors,
        }


def choose_moving_indices(num_frames: int, batch_size: int):
    return np.random.randint(0, num_frames, size=batch_size)


def choose_temporal_starts(num_frames: int, batch_size: int):
    if num_frames < 3:
        return np.zeros((0,), dtype=np.int64)
    return np.random.randint(0, num_frames - 2, size=batch_size)


def train_model(args):
    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device(log=True)
    sam2_settings = resolve_sam2_settings(config, args)

    output_dir = Path(args.data_path)
    video_dir = output_dir / "video_gray"
    frame_paths = list_frame_paths(video_dir)
    max_frames = config["video"]["max_frames"]
    if max_frames is not None:
        frame_paths = frame_paths[: int(max_frames)]

    template_tensor, template_raw = load_gray_frame(frame_paths[0], resize_hw=config["video"]["train_resize"])
    orig_template = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    bbox_train = resize_bbox(args.bbox, orig_template.shape[:2], config["video"]["train_resize"])
    if sam2_settings["enabled"]:
        mask_dir = output_dir / sam2_settings["mask_dirname"]
        train_masks, _ = load_masks(mask_dir, resize_hw=config["video"]["train_resize"], max_frames=len(frame_paths))
        template_mask = train_masks[0:1]
    else:
        train_masks = None
        template_mask = bbox_to_mask(bbox_train, config["video"]["train_resize"]).unsqueeze(0)

    model = DINOMotionGrayModel(
        config=config,
        device=device,
        torch_home=str(PROJECT_ROOT / ".torch_cache_dinomotion"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    ckpt_dir = output_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    template_tensor = template_tensor.unsqueeze(0).to(device)
    template_mask = template_mask.to(device)

    training_state = {
        "bbox_original": list(map(int, args.bbox)),
        "bbox_train": list(map(int, bbox_train)),
        "train_resize": list(map(int, config["video"]["train_resize"])),
        "num_frames": len(frame_paths),
        "model_name": config["model"]["dino_model_name"],
    }
    save_json(training_state, output_dir / "metadata.json")

    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    log_every = int(config["training"]["log_every"])
    save_every = int(config["training"]["save_every"])
    descriptor_identity_weight = float(config["training"].get("descriptor_identity_weight", 0.0))
    temporal_triplet_weight = float(config["training"].get("temporal_triplet_weight", 0.0))
    temporal_descriptor_weight = float(config["training"].get("temporal_descriptor_weight", 0.0))

    progress = tqdm(range(1, steps + 1), desc="Training DINOMotion-gray")
    for step in progress:
        moving_indices = choose_moving_indices(len(frame_paths), batch_size)
        moving_batch = []
        for idx in moving_indices:
            moving_tensor, _ = load_gray_frame(frame_paths[int(idx)], resize_hw=config["video"]["train_resize"])
            moving_batch.append(moving_tensor)
        moving_tensor = torch.stack(moving_batch, dim=0).to(device)
        template_batch = template_tensor.expand(batch_size, -1, -1, -1)
        mask_batch = template_mask.expand(batch_size, -1, -1)
        moving_mask_batch = None
        if train_masks is not None:
            moving_mask_batch = train_masks[moving_indices.tolist()].to(device)

        model.train()
        out = model(template_batch, moving_tensor, mask_batch, moving_mask=moving_mask_batch)
        photometric_loss = masked_mse_loss(template_batch, out["registered"], mask_batch)
        descriptor_identity_loss = (1.0 - F.cosine_similarity(out["template_vectors"], out["moving_vectors"], dim=-1)).mean()

        identity_template_features = model.encoder.forward_feature_map(template_batch[:1])
        identity_points = model.extract_landmarks(template_batch[:1], identity_template_features, mask_batch[:1])
        identity_moving_points, _, _, _, _ = model.match_landmarks(
            identity_template_features,
            identity_template_features,
            identity_points,
            template_batch[:1].shape[-2:],
            moving_mask=mask_batch[:1],
        )
        identity_loss = F.smooth_l1_loss(identity_moving_points, identity_points)

        temporal_triplet_loss = torch.tensor(0.0, device=device)
        temporal_descriptor_loss = torch.tensor(0.0, device=device)
        temporal_starts = choose_temporal_starts(len(frame_paths), batch_size)
        if temporal_starts.size > 0:
            triplet_frames = [[], [], []]
            triplet_masks = [[], [], []] if train_masks is not None else None
            for start_idx in temporal_starts.tolist():
                for offset in range(3):
                    moving_tensor_triplet, _ = load_gray_frame(frame_paths[start_idx + offset], resize_hw=config["video"]["train_resize"])
                    triplet_frames[offset].append(moving_tensor_triplet)
                    if triplet_masks is not None:
                        triplet_masks[offset].append(train_masks[start_idx + offset])

            triplet_outputs = []
            triplet_batch = template_tensor.expand(len(temporal_starts), -1, -1, -1)
            triplet_template_mask = template_mask.expand(len(temporal_starts), -1, -1)
            for offset in range(3):
                triplet_tensor = torch.stack(triplet_frames[offset], dim=0).to(device)
                moving_mask_triplet = None
                if triplet_masks is not None:
                    moving_mask_triplet = torch.stack(triplet_masks[offset], dim=0).to(device)
                triplet_outputs.append(
                    model(triplet_batch, triplet_tensor, triplet_template_mask, moving_mask=moving_mask_triplet)
                )

            vel_prev = triplet_outputs[1]["moving_points"] - triplet_outputs[0]["moving_points"]
            vel_next = triplet_outputs[2]["moving_points"] - triplet_outputs[1]["moving_points"]
            temporal_triplet_loss = F.smooth_l1_loss(vel_next, vel_prev)
            desc01 = 1.0 - F.cosine_similarity(triplet_outputs[0]["moving_vectors"], triplet_outputs[1]["moving_vectors"], dim=-1)
            desc12 = 1.0 - F.cosine_similarity(triplet_outputs[1]["moving_vectors"], triplet_outputs[2]["moving_vectors"], dim=-1)
            temporal_descriptor_loss = 0.5 * (desc01.mean() + desc12.mean())

        loss = (
            float(config["training"]["photometric_weight"]) * photometric_loss
            + float(config["training"]["identity_weight"]) * identity_loss
            + descriptor_identity_weight * descriptor_identity_loss
            + temporal_triplet_weight * temporal_triplet_loss
            + temporal_descriptor_weight * temporal_descriptor_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                photometric=f"{photometric_loss.item():.4f}",
                identity=f"{identity_loss.item():.4f}",
                desc=f"{descriptor_identity_loss.item():.4f}",
                temp=f"{temporal_triplet_loss.item():.4f}",
            )
        if step % save_every == 0 or step == steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "bbox_original": args.bbox,
                    "bbox_train": bbox_train,
                    "step": step,
                },
                ckpt_dir / "dinomotion_gray_latest.pt",
            )


@torch.no_grad()
def infer_tracks(args):
    config = load_config(args.config)
    device = get_device(log=True)
    sam2_settings = resolve_sam2_settings(config, args)
    output_dir = Path(args.data_path)
    ckpt_path = output_dir / "checkpoints" / "dinomotion_gray_latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    metadata = json.loads((output_dir / "metadata.json").read_text())

    model = DINOMotionGrayModel(
        config=config,
        device=device,
        torch_home=str(PROJECT_ROOT / ".torch_cache_dinomotion"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    train_resize = tuple(config["video"]["train_resize"])
    video_dir = output_dir / "video_gray"
    frame_paths = list_frame_paths(video_dir)[: int(metadata["num_frames"])]

    template_tensor, template_raw_resized = load_gray_frame(frame_paths[0], resize_hw=train_resize)
    if sam2_settings["enabled"]:
        mask_dir = output_dir / sam2_settings["mask_dirname"]
        train_masks, orig_masks = load_masks(mask_dir, resize_hw=train_resize, max_frames=len(frame_paths))
        train_masks_np = (train_masks.cpu().numpy() > 0).astype(np.uint8) * 255
        template_mask = train_masks[0:1].to(device)
    else:
        train_masks = None
        train_masks_np, orig_masks = None, None
        template_mask = bbox_to_mask(metadata["bbox_train"], train_resize).unsqueeze(0).to(device)
    template_tensor = template_tensor.unsqueeze(0).to(device)

    template_features = model.encoder.forward_feature_map(template_tensor)
    template_points = model.extract_landmarks(template_tensor, template_features, template_mask)

    tracks = []
    similarities = []
    resized_frames = []
    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Inferring landmarks")):
        moving_tensor, moving_raw = load_gray_frame(frame_path, resize_hw=train_resize)
        moving_tensor = moving_tensor.unsqueeze(0).to(device)
        moving_features = model.encoder.forward_feature_map(moving_tensor)
        moving_mask = train_masks[frame_idx : frame_idx + 1].to(device) if train_masks is not None else None
        moving_points, similarity, _, _, _ = model.match_landmarks(
            template_features,
            moving_features,
            template_points,
            template_tensor.shape[-2:],
            moving_mask=moving_mask,
        )
        tracks.append(moving_points.squeeze(0).cpu().numpy())
        similarities.append(similarity.squeeze(0).amax(dim=(-2, -1)).cpu().numpy())
        resized_frames.append(moving_raw)

    tracks = np.stack(tracks, axis=0).transpose(1, 0, 2)  # K x T x 2
    scores = np.stack(similarities, axis=0).transpose(1, 0)  # K x T
    resized_frames = np.stack(resized_frames, axis=0)

    refined_tracks, occluded = temporal_refine_tracks(
        resized_frames,
        tracks,
        scores,
        config["refinement"],
        masks_uint8=train_masks_np,
    )
    smoothed_tracks = smooth_tracks(refined_tracks, occluded, config["refinement"])

    orig_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = orig_frame.shape[:2]
    scale_x = orig_w / float(train_resize[1])
    scale_y = orig_h / float(train_resize[0])
    tracks_initial_out = tracks.copy()
    tracks_initial_out[..., 0] *= scale_x
    tracks_initial_out[..., 1] *= scale_y
    smoothed_tracks_out = smoothed_tracks.copy()
    smoothed_tracks_out[..., 0] *= scale_x
    smoothed_tracks_out[..., 1] *= scale_y
    refined_tracks_out = refined_tracks.copy()
    refined_tracks_out[..., 0] *= scale_x
    refined_tracks_out[..., 1] *= scale_y
    template_points_np = template_points.squeeze(0).cpu().numpy()
    template_points_np[:, 0] *= scale_x
    template_points_np[:, 1] *= scale_y

    out_dir = output_dir / "outputs"
    ensure_dir(out_dir)
    np.save(out_dir / "tracks_initial.npy", tracks_initial_out)
    np.save(out_dir / "tracks_refined.npy", refined_tracks_out)
    np.save(out_dir / "tracks_smoothed.npy", smoothed_tracks_out)
    np.save(out_dir / "occlusions.npy", occluded.astype(np.uint8))
    np.save(out_dir / "template_landmarks.npy", template_points_np)
    print(f"Saved inference outputs to {out_dir}", flush=True)


def temporal_refine_tracks(frames_uint8: np.ndarray, initial_tracks: np.ndarray, scores: np.ndarray, refinement_cfg, masks_uint8: np.ndarray | None = None):
    num_points, num_frames, _ = initial_tracks.shape
    refined = initial_tracks.copy()
    occluded = np.zeros((num_points, num_frames), dtype=bool)
    occluded[:, 0] = False if masks_uint8 is None else ~points_inside_mask(refined[:, 0], masks_uint8[0])

    lk_params = dict(
        winSize=(int(refinement_cfg["lk_window_size"]), int(refinement_cfg["lk_window_size"])),
        maxLevel=int(refinement_cfg["lk_max_level"]),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    alpha = float(refinement_cfg["blend_alpha"])
    max_jump = float(refinement_cfg["max_jump"])

    prev_points = refined[:, 0].astype(np.float32).reshape(-1, 1, 2)
    for t in range(1, num_frames):
        prev_frame = frames_uint8[t - 1]
        cur_frame = frames_uint8[t]
        global_points = initial_tracks[:, t].astype(np.float32)
        lk_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame, prev_points, None, **lk_params)
        if lk_points is None:
            refined[:, t] = global_points
            occluded[:, t] = True
            prev_points = refined[:, t].astype(np.float32).reshape(-1, 1, 2)
            continue

        lk_points = lk_points.reshape(-1, 2)
        status = status.reshape(-1).astype(bool)
        disagreement = np.linalg.norm(lk_points - global_points, axis=-1)
        good = status & (disagreement <= max_jump)
        blended = global_points.copy()
        blended[good] = alpha * lk_points[good] + (1.0 - alpha) * global_points[good]
        if masks_uint8 is not None:
            inside_mask = points_inside_mask(blended, masks_uint8[t])
            good = good & inside_mask
        refined[:, t] = blended
        occluded[:, t] = ~good
        prev_points = blended.astype(np.float32).reshape(-1, 1, 2)

    low_score = scores < np.percentile(scores, 20)
    occluded = np.logical_or(occluded, low_score)
    if masks_uint8 is not None:
        for t in range(num_frames):
            occluded[:, t] = np.logical_or(occluded[:, t], ~points_inside_mask(refined[:, t], masks_uint8[t]))
    return refined, occluded


def smooth_tracks(tracks: np.ndarray, occluded: np.ndarray, refinement_cfg):
    smoothed = tracks.copy()
    window_length = int(refinement_cfg["savgol_window_length"])
    polyorder = int(refinement_cfg["savgol_polyorder"])
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    if window_length > tracks.shape[1]:
        window_length = tracks.shape[1] if tracks.shape[1] % 2 == 1 else max(3, tracks.shape[1] - 1)

    for point_idx in range(tracks.shape[0]):
        for coord_idx in range(2):
            signal = tracks[point_idx, :, coord_idx]
            if window_length >= 3:
                smoothed[point_idx, :, coord_idx] = savgol_filter(signal, window_length=window_length, polyorder=min(polyorder, window_length - 1), mode="interp")
        smoothed[point_idx, occluded[point_idx], :] = tracks[point_idx, occluded[point_idx], :]
    return smoothed


def render_tracks(args):
    config = load_config(args.config)
    output_dir = Path(args.data_path)
    sam2_settings = resolve_sam2_settings(config, args)
    render_cfg = dict(config.get("rendering", {}))
    frame_paths = list_frame_paths(output_dir / "video_gray")
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frame_paths]
    frames = np.stack(frames, axis=0)

    outputs_dir = output_dir / "outputs"
    tracks = np.load(outputs_dir / "tracks_smoothed.npy")
    occluded = np.load(outputs_dir / "occlusions.npy").astype(bool)
    frame_masks = None
    if sam2_settings["enabled"]:
        mask_dir = output_dir / sam2_settings["mask_dirname"]
        if mask_dir.exists():
            _, frame_masks = load_masks(mask_dir, max_frames=frames.shape[0])
    colors = make_color_palette(tracks.shape[0])
    min_visible_frames = int(render_cfg.get("min_visible_frames", 0))
    max_track_p95_jump = float(render_cfg.get("max_track_p95_jump_px", float("inf")))
    max_trail_jump = float(render_cfg.get("max_trail_jump_px", float("inf")))
    keep_track = np.ones(tracks.shape[0], dtype=bool)
    for point_idx in range(tracks.shape[0]):
        visible_mask = ~occluded[point_idx]
        pts = tracks[point_idx][visible_mask]
        if pts.shape[0] < min_visible_frames:
            keep_track[point_idx] = False
            continue
        if pts.shape[0] >= 2 and np.isfinite(max_track_p95_jump):
            jumps = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
            if jumps.size and float(np.percentile(jumps, 95)) > max_track_p95_jump:
                keep_track[point_idx] = False

    dotted = []
    trails = []
    trail_length = int(render_cfg.get("trail_length", args.trail_length))

    for t in tqdm(range(frames.shape[0]), desc="Rendering DINOMotion-gray"):
        frame_bgr_dots = cv2.cvtColor(frames[t], cv2.COLOR_GRAY2BGR)
        frame_bgr_trails = frame_bgr_dots.copy()
        for point_idx in range(tracks.shape[0]):
            if not keep_track[point_idx]:
                continue
            color = colors[point_idx]
            visible_now = not occluded[point_idx, t]
            if visible_now and frame_masks is not None:
                visible_now = bool(points_inside_mask(tracks[point_idx : point_idx + 1, t], frame_masks[t])[0])
            if visible_now:
                x, y = tracks[point_idx, t]
                cv2.circle(frame_bgr_dots, (int(round(x)), int(round(y))), 4, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(frame_bgr_trails, (int(round(x)), int(round(y))), 4, color, -1, lineType=cv2.LINE_AA)

            start_t = max(0, t - trail_length)
            for prev_t in range(start_t + 1, t + 1):
                if occluded[point_idx, prev_t] or occluded[point_idx, prev_t - 1]:
                    continue
                if frame_masks is not None:
                    inside_prev = bool(points_inside_mask(tracks[point_idx : point_idx + 1, prev_t - 1], frame_masks[prev_t - 1])[0])
                    inside_cur = bool(points_inside_mask(tracks[point_idx : point_idx + 1, prev_t], frame_masks[prev_t])[0])
                    if not (inside_prev and inside_cur):
                        continue
                p1 = tracks[point_idx, prev_t - 1]
                p2 = tracks[point_idx, prev_t]
                if np.isfinite(max_trail_jump):
                    if float(np.linalg.norm(p2 - p1)) > max_trail_jump:
                        continue
                alpha = 0.25 + 0.75 * ((prev_t - start_t) / max(t - start_t, 1))
                fade_color = tuple(int(alpha * c) for c in color)
                cv2.line(
                    frame_bgr_trails,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])), int(round(p2[1]))),
                    fade_color,
                    2,
                    lineType=cv2.LINE_AA,
                )
        dotted.append(cv2.cvtColor(frame_bgr_dots, cv2.COLOR_BGR2RGB))
        trails.append(cv2.cvtColor(frame_bgr_trails, cv2.COLOR_BGR2RGB))

    vis_dir = output_dir / "visualizations"
    ensure_dir(vis_dir)
    output_tag = str(render_cfg.get("output_tag", "")).strip()
    tag_suffix = f"_{output_tag}" if output_tag else ""
    dotted_path = vis_dir / f"dinomotion_gray_dots{tag_suffix}_fps_{config['video']['vis_fps']}.mp4"
    trails_path = vis_dir / f"dinomotion_gray_trails{tag_suffix}_fps_{config['video']['vis_fps']}.mp4"
    save_video(np.stack(dotted, axis=0), str(dotted_path), fps=int(config["video"]["vis_fps"]))
    save_video(np.stack(trails, axis=0), str(trails_path), fps=int(config["video"]["vis_fps"]))
    print(f"Saved DINOMotion-gray visualizations to {vis_dir}", flush=True)


def run_future_generation(data_path: str, future_config: str, history: int | None = None, horizon: int | None = None):
    from types import SimpleNamespace
    from dinomotion_gray.future_frame_generator import generate_future_frames

    future_args = SimpleNamespace(
        config=future_config,
        data_path=data_path,
        history=history,
        horizon=horizon,
    )
    generate_future_frames(future_args)


def run_all(args):
    config = load_config(args.config)
    sam2_settings = resolve_sam2_settings(config, args)
    frame_ext = config["video"].get("frame_ext", "png")
    output_dir = Path(args.data_path)
    ensure_dir(output_dir)
    video_dir = output_dir / "video_gray"
    extract_grayscale_frames(
        video_path=args.video_path,
        output_dir=str(video_dir),
        max_frames=config["video"]["max_frames"],
        overwrite=args.overwrite,
        frame_ext=frame_ext,
    )
    save_json({"bbox": list(map(int, args.bbox))}, output_dir / "roi_bbox.json")
    if sam2_settings["enabled"]:
        rgb_proxy_dir = output_dir / sam2_settings["rgb_proxy_dirname"]
        mask_dir = output_dir / sam2_settings["mask_dirname"]
        prepare_rgb_proxy_frames(video_dir, rgb_proxy_dir, overwrite=args.overwrite)
        print(f"Generating SAM2 masks using mode: {sam2_settings['mode']}", flush=True)
        generate_sam2_masks(
            video_dir=rgb_proxy_dir,
            output_dir=mask_dir,
            config_path=sam2_settings["config_path"],
            mode=sam2_settings["mode"],
            bbox=args.bbox if sam2_settings["mode"] == "bbox" else None,
            point=sam2_settings["point"],
            overwrite=args.overwrite,
        )
    train_model(args)
    infer_tracks(args)
    render_tracks(args)
    if getattr(args, "future_config", None):
        run_future_generation(
            data_path=args.data_path,
            future_config=args.future_config,
            history=getattr(args, "future_history", None),
            horizon=getattr(args, "future_horizon", None),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="DINOMotion-inspired grayscale point tracking pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared(subparser):
        subparser.add_argument("--config", default="./dinomotion_gray/configs/plane_gray.yaml", type=str)
        subparser.add_argument("--data-path", required=True, type=str)
        subparser.add_argument("--bbox", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2"))
        subparser.add_argument("--sam2-mode", choices=["off", "auto", "bbox", "point"], default=None)
        subparser.add_argument("--sam2-point", nargs=2, type=int, metavar=("X", "Y"))
        subparser.add_argument("--sam2-config", type=str, default=None)
        subparser.add_argument("--seed", default=7, type=int)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--video-path", required=True, type=str)
    run_parser.add_argument("--overwrite", action="store_true", default=False)
    run_parser.add_argument("--trail-length", type=int, default=20)
    run_parser.add_argument("--future-config", type=str, default=None)
    run_parser.add_argument("--future-history", type=int, default=None)
    run_parser.add_argument("--future-horizon", type=int, default=None)
    add_shared(run_parser)

    train_parser = subparsers.add_parser("train")
    add_shared(train_parser)

    infer_parser = subparsers.add_parser("infer")
    add_shared(infer_parser)

    vis_parser = subparsers.add_parser("visualize")
    add_shared(vis_parser)
    vis_parser.add_argument("--trail-length", type=int, default=20)

    future_parser = subparsers.add_parser("future")
    future_parser.add_argument("--data-path", required=True, type=str)
    future_parser.add_argument("--future-config", default="./dinomotion_gray/configs/plane_gray_future.yaml", type=str)
    future_parser.add_argument("--future-history", type=int, default=None)
    future_parser.add_argument("--future-horizon", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "run":
        run_all(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "infer":
        infer_tracks(args)
    elif args.command == "visualize":
        render_tracks(args)
    elif args.command == "future":
        run_future_generation(
            data_path=args.data_path,
            future_config=args.future_config,
            history=args.future_history,
            horizon=args.future_horizon,
        )


if __name__ == "__main__":
    main()
