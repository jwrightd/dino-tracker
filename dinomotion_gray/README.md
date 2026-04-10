# DINOMotion-Gray

This directory contains a separate grayscale-only, DINOMotion-inspired point-tracking pipeline.

Core ideas mirrored from the uploaded presentation:

- single-channel grayscale input
- DINOv2 backbone adapted with LoRA
- template/moving landmark correspondences
- affine image registration loss
- TAPIR-style temporal refinement after per-frame initialization

Example:

```bash
python ./dinomotion_gray/pipeline.py run \
  --video-path /Users/jamesw/Desktop/attempt_2/source_video.mp4 \
  --data-path /Users/jamesw/Desktop/attempt_2_dinomotion_gray \
  --bbox 520 160 960 350 \
  --trail-length 25 \
  --overwrite
```

Future-frame generation (non-diffusion, sparse-motion/TPS-style) can be run after tracking:

```bash
python ./dinomotion_gray/future_frame_generator.py \
  --config ./dinomotion_gray/configs/plane_gray_future.yaml \
  --data-path /Users/jamesw/Desktop/attempt_2_dinomotion_gray_sam2
```

The default future generator uses a Shih-style landmark-space predictor plus TPS warping.
