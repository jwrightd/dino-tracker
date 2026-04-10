import argparse
from pathlib import Path

import numpy as np
import torch

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
    with torch.inference_mode():
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=np.asarray(bbox, dtype=np.float32),
        )


def _add_point_prompt(predictor, inference_state, point):
    points = np.asarray([point], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int32)
    with torch.inference_mode():
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )


def _propagate_and_save(predictor, inference_state, frame_paths, output_dir):
    n_saved = 0
    with torch.inference_mode():
        for frame_idx, _, mask_logits in predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0.0).squeeze().detach().cpu().numpy()
            out_name = Path(frame_paths[frame_idx]).stem + ".png"
            save_binary_mask(mask, Path(output_dir) / out_name)
            n_saved += 1
    print(f"Saved {n_saved} masks to {output_dir}", flush=True)


def _generate_auto_masks(build_sam2, build_sam2_video_predictor, sam2_automatic_mask_generator_cls, config, device, frames, frame_paths, output_dir):
    mask_generator = _build_auto_mask_generator(build_sam2, sam2_automatic_mask_generator_cls, config, device)
    first_frame_masks = mask_generator.generate(frames[0])
    if not first_frame_masks:
        raise RuntimeError("SAM2 auto mode found no masks on frame 0. Try bbox or point mode instead.")

    first_frame_masks.sort(key=lambda mask_info: mask_info["area"], reverse=True)
    best_mask = first_frame_masks[0]["segmentation"]
    selected_area = first_frame_masks[0]["area"]
    frame_area = frames[0].shape[0] * frames[0].shape[1]
    bbox = mask_to_box(best_mask)
    print(
        f"Auto mode selected a mask of area {selected_area} pixels "
        f"({100.0 * selected_area / frame_area:.1f}% of frame).",
        flush=True,
    )

    predictor = _build_video_predictor(build_sam2_video_predictor, config, device)
    inference_state = predictor.init_state(video_path=str(Path(frame_paths[0]).parent))
    _add_bbox_prompt(predictor, inference_state, bbox)
    _propagate_and_save(predictor, inference_state, frame_paths, output_dir)


def _generate_bbox_masks(build_sam2_video_predictor, config, device, frame_paths, output_dir, bbox):
    predictor = _build_video_predictor(build_sam2_video_predictor, config, device)
    inference_state = predictor.init_state(video_path=str(Path(frame_paths[0]).parent))
    _add_bbox_prompt(predictor, inference_state, bbox)
    _propagate_and_save(predictor, inference_state, frame_paths, output_dir)


def _generate_point_masks(build_sam2_video_predictor, config, device, frame_paths, output_dir, point):
    predictor = _build_video_predictor(build_sam2_video_predictor, config, device)
    inference_state = predictor.init_state(video_path=str(Path(frame_paths[0]).parent))
    _add_point_prompt(predictor, inference_state, point)
    _propagate_and_save(predictor, inference_state, frame_paths, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate SAM2 foreground masks for DINO-Tracker.")
    parser.add_argument("--video-dir", required=True, help="Directory with extracted RGB frames.")
    parser.add_argument("--output-dir", required=True, help="Directory to save binary mask PNGs.")
    parser.add_argument("--config", default="./sam2_masking/sam2_config.yaml", help="SAM2 config YAML path.")
    parser.add_argument("--mode", choices=["auto", "bbox", "point"], default="auto")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"))
    parser.add_argument("--sam2-checkpoint", default=None, help="Override SAM2 checkpoint path.")
    parser.add_argument("--sam2-model-cfg", default=None, help="Override SAM2 model config name.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    config = load_sam2_config(args.config)
    if args.sam2_checkpoint is not None:
        config["checkpoint"] = args.sam2_checkpoint
    if args.sam2_model_cfg is not None:
        config["model_cfg"] = args.sam2_model_cfg
    config["model_cfg"] = normalize_model_cfg_name(config["model_cfg"])

    frame_paths_dir = Path(args.video_dir)
    if not frame_paths_dir.exists():
        raise FileNotFoundError(f"Video frame directory not found: {args.video_dir}")

    validate_sam2_inputs(config["checkpoint"], config["model_cfg"])
    build_sam2, build_sam2_video_predictor, sam2_automatic_mask_generator_cls = import_sam2_dependencies()
    device = get_sam2_device(log=True)

    frames, frame_paths = load_frames(args.video_dir, grayscale_to_rgb=True)
    print(f"Loaded {len(frame_paths)} frames from {args.video_dir}", flush=True)

    min_mask_area_ratio = config.get("min_mask_area_ratio", 0.01)
    config["min_mask_region_area"] = int(min_mask_area_ratio * frames[0].shape[0] * frames[0].shape[1])

    output_dir = args.output_dir
    if args.overwrite:
        ensure_clean_output_dir(output_dir)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "auto":
        _generate_auto_masks(
            build_sam2,
            build_sam2_video_predictor,
            sam2_automatic_mask_generator_cls,
            config,
            device,
            frames,
            frame_paths,
            output_dir,
        )
    elif args.mode == "bbox":
        if args.bbox is None:
            raise ValueError("--bbox is required when --mode bbox is selected.")
        _generate_bbox_masks(build_sam2_video_predictor, config, device, frame_paths, output_dir, args.bbox)
    else:
        if args.point is None:
            raise ValueError("--point is required when --mode point is selected.")
        _generate_point_masks(build_sam2_video_predictor, config, device, frame_paths, output_dir, args.point)


if __name__ == "__main__":
    main()
