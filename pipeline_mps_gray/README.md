# Grayscale MPS/CPU Pipeline

This pipeline adds a new entry point that:

1. takes a video file,
2. converts it to grayscale frames (single-channel by default),
3. runs existing preprocessing logic,
4. runs existing training logic.

It uses MPS when available, otherwise CPU, and disables CUDA via environment configuration for subprocesses.
Internally, model loaders convert grayscale frames to RGB where required by RAFT/DINO.
Direct-flow trajectory filtering is disabled by default in this pipeline for MPS memory safety.

## Full pipeline

```bash
python ./pipeline_mps_gray/run_grayscale_pipeline.py \
  --video-path /absolute/path/to/input_video.mp4 \
  --data-path /absolute/path/to/output_dataset \
  --preprocess-config ./config/preprocessing.yaml \
  --train-config ./config/train.yaml \
  --max-frames 200 \
  --overwrite-video-frames
```

To save 3-channel grayscale frames instead of single-channel:

```bash
python ./pipeline_mps_gray/run_grayscale_pipeline.py \
  --video-path /absolute/path/to/input_video.mp4 \
  --data-path /absolute/path/to/output_dataset \
  --preprocess-config ./config/preprocessing.yaml \
  --train-config ./config/train.yaml \
  --max-frames 200 \
  --overwrite-video-frames \
  --three-channel-frames
```

If you intentionally want direct-flow filtering (higher memory usage):

```bash
python ./pipeline_mps_gray/run_grayscale_pipeline.py \
  --video-path /absolute/path/to/input_video.mp4 \
  --data-path /absolute/path/to/output_dataset \
  --preprocess-config ./config/preprocessing.yaml \
  --train-config ./config/train.yaml \
  --max-frames 200 \
  --overwrite-video-frames \
  --enable-direct-flow
```

## Preprocessing only

```bash
python ./pipeline_mps_gray/run_grayscale_pipeline.py \
  --video-path /absolute/path/to/input_video.mp4 \
  --data-path /absolute/path/to/output_dataset \
  --preprocess-config ./config/preprocessing.yaml \
  --max-frames 200 \
  --overwrite-video-frames \
  --skip-training
```

## Training only (reuse existing preprocessed artifacts)

```bash
python ./pipeline_mps_gray/run_grayscale_pipeline.py \
  --video-path /absolute/path/to/input_video.mp4 \
  --data-path /absolute/path/to/output_dataset \
  --train-config ./config/train.yaml \
  --skip-preprocessing
```
