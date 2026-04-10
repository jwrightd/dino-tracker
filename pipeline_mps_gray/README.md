# RGB-First Optimized Pipeline

This directory now contains two pipeline entrypoints:

- `run_full_pipeline.sh`: the main optimized RGB-first workflow
- `run_grayscale_pipeline.py`: the older grayscale-focused helper, kept for compatibility

The optimized runner keeps RGB footage unchanged, converts grayscale footage to RGB only when needed, generates SAM2 masks, runs the existing preprocessing and training stack, smooths grid trajectories, and then visualizes the result.

## Optimized full pipeline

```bash
bash ./pipeline_mps_gray/run_full_pipeline.sh \
  /absolute/path/to/input_video.mp4 \
  /absolute/path/to/output_dataset
```

Example with a medical-style first-frame bounding box prompt:

```bash
bash ./pipeline_mps_gray/run_full_pipeline.sh \
  /absolute/path/to/input_video.mp4 \
  /absolute/path/to/output_dataset \
  --sam2-mode bbox \
  --bbox 50 40 300 280
```

Useful options:

- `--input-mode auto|rgb|grayscale`
- `--max-frames 200`
- `--interval 32`
- `--query-chunk-size 256`
- `--batch-size 64`
- `--plot-trails`

SAM2 is required for the optimized runner and must be installed separately, along with a checkpoint configured in `./sam2_masking/sam2_config.yaml`.

## Legacy grayscale helper

```bash
python ./pipeline_mps_gray/run_grayscale_pipeline.py \
  --video-path /absolute/path/to/input_video.mp4 \
  --data-path /absolute/path/to/output_dataset \
  --preprocess-config ./config/preprocessing.yaml \
  --train-config ./config/train.yaml \
  --max-frames 200 \
  --overwrite-video-frames
```
