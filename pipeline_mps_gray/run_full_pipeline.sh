#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PREPROCESS_CONFIG="./config/preprocessing.yaml"
TRAIN_CONFIG="./config/train.yaml"
SAM2_CONFIG="./sam2_masking/sam2_config.yaml"

INPUT_MODE="auto"
SAM2_MODE="auto"
MAX_FRAMES=""
INTERVAL="32"
QUERY_CHUNK_SIZE="256"
BATCH_SIZE="64"
VIS_FPS="10"
PLOT_TRAILS="false"
OVERWRITE_FRAMES="false"
SKIP_OCCLUSION="true"
BBOX=()
POINT=()

usage() {
  echo "Usage: $0 <video_path> <video_dir> [options]"
  echo "Options:"
  echo "  --input-mode auto|rgb|grayscale"
  echo "  --sam2-mode auto|bbox|point"
  echo "  --bbox X1 Y1 X2 Y2"
  echo "  --point X Y"
  echo "  --max-frames N"
  echo "  --interval N"
  echo "  --query-chunk-size N"
  echo "  --batch-size N"
  echo "  --fps N"
  echo "  --plot-trails"
  echo "  --overwrite-frames"
  echo "  --with-occlusion"
  echo "  --preprocess-config PATH"
  echo "  --train-config PATH"
  echo "  --sam2-config PATH"
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

VIDEO_PATH="$1"
VIDEO_DIR="$2"
shift 2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-mode)
      INPUT_MODE="$2"
      shift 2
      ;;
    --sam2-mode)
      SAM2_MODE="$2"
      shift 2
      ;;
    --bbox)
      BBOX=("$2" "$3" "$4" "$5")
      shift 5
      ;;
    --point)
      POINT=("$2" "$3")
      shift 3
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --query-chunk-size)
      QUERY_CHUNK_SIZE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --fps)
      VIS_FPS="$2"
      shift 2
      ;;
    --plot-trails)
      PLOT_TRAILS="true"
      shift
      ;;
    --overwrite-frames)
      OVERWRITE_FRAMES="true"
      shift
      ;;
    --with-occlusion)
      SKIP_OCCLUSION="false"
      shift
      ;;
    --preprocess-config)
      PREPROCESS_CONFIG="$2"
      shift 2
      ;;
    --train-config)
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --sam2-config)
      SAM2_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

mkdir -p "$VIDEO_DIR"
VIDEO_DIR_ABS="$(cd "$VIDEO_DIR" && pwd)"

DERIVED_PREPROCESS_CONFIG="$VIDEO_DIR_ABS/.generated_configs/preprocessing.optimized.yaml"

echo "=== Step 1: Extract frames and prepare RGB-first input ==="
PREP_CMD=(
  "$PYTHON_BIN" ./pipeline_mps_gray/prepare_video_frames.py
  --video-path "$VIDEO_PATH"
  --output-dir "$VIDEO_DIR_ABS/video"
  --input-mode "$INPUT_MODE"
)
if [[ "$OVERWRITE_FRAMES" == "true" ]]; then
  PREP_CMD+=(--overwrite)
fi
if [[ -n "$MAX_FRAMES" ]]; then
  PREP_CMD+=(--max-frames "$MAX_FRAMES")
fi
"${PREP_CMD[@]}"

echo "=== Step 2: Build optimized preprocessing config ==="
"$PYTHON_BIN" - "$PREPROCESS_CONFIG" "$DERIVED_PREPROCESS_CONFIG" "$MAX_FRAMES" <<'PY'
import sys
from pathlib import Path

import yaml

source_config = Path(sys.argv[1])
output_config = Path(sys.argv[2])
max_frames = sys.argv[3]

with open(source_config, "r") as f:
    config = yaml.safe_load(f.read()) or {}

config["filter_using_direct_flow"] = False
config["direct_flow_threshold"] = None
if max_frames:
    config["max_frames"] = int(max_frames)

output_config.parent.mkdir(parents=True, exist_ok=True)
with open(output_config, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

echo "=== Step 3: Generate SAM2 foreground masks ==="
SAM2_CMD=(
  "$PYTHON_BIN" ./sam2_masking/generate_sam2_masks.py
  --video-dir "$VIDEO_DIR_ABS/video"
  --output-dir "$VIDEO_DIR_ABS/masks"
  --config "$SAM2_CONFIG"
  --mode "$SAM2_MODE"
  --overwrite
)
if [[ "$SAM2_MODE" == "bbox" ]]; then
  if [[ ${#BBOX[@]} -ne 4 ]]; then
    echo "--bbox requires exactly 4 integers when --sam2-mode bbox is used."
    exit 1
  fi
  SAM2_CMD+=(--bbox "${BBOX[@]}")
fi
if [[ "$SAM2_MODE" == "point" ]]; then
  if [[ ${#POINT[@]} -ne 2 ]]; then
    echo "--point requires exactly 2 integers when --sam2-mode point is used."
    exit 1
  fi
  SAM2_CMD+=(--point "${POINT[@]}")
fi
"${SAM2_CMD[@]}"

echo "=== Step 4: Run preprocessing ==="
"$PYTHON_BIN" ./preprocessing/main_preprocessing.py \
  --config "$DERIVED_PREPROCESS_CONFIG" \
  --data-path "$VIDEO_DIR_ABS"

echo "=== Step 5: Train DINO-Tracker ==="
"$PYTHON_BIN" ./train.py \
  --config "$TRAIN_CONFIG" \
  --data-path "$VIDEO_DIR_ABS"

echo "=== Step 6: Run grid inference ==="
INFER_CMD=(
  "$PYTHON_BIN" ./inference_grid.py
  --config "$TRAIN_CONFIG"
  --data-path "$VIDEO_DIR_ABS"
  --use-segm-mask
  --interval "$INTERVAL"
  --query-chunk-size "$QUERY_CHUNK_SIZE"
  --batch-size "$BATCH_SIZE"
)
if [[ "$SKIP_OCCLUSION" == "true" ]]; then
  INFER_CMD+=(--skip-occlusion)
fi
"${INFER_CMD[@]}"

echo "=== Step 7: Smooth trajectories ==="
"$PYTHON_BIN" ./postprocessing/smooth_trajectories.py \
  --data-path "$VIDEO_DIR_ABS" \
  --config "$TRAIN_CONFIG"

echo "=== Step 8: Visualize ==="
VIS_CMD=(
  "$PYTHON_BIN" ./visualization/visualize_rainbow.py
  --data-path "$VIDEO_DIR_ABS"
  --trajectories-path "$VIDEO_DIR_ABS/grid_trajectories/grid_trajectories_smoothed.npy"
  --occlusions-path "$VIDEO_DIR_ABS/grid_occlusions/grid_occlusions.npy"
  --fps "$VIS_FPS"
)
if [[ "$PLOT_TRAILS" == "true" ]]; then
  VIS_CMD+=(--plot-trails)
fi
"${VIS_CMD[@]}"

echo "=== Pipeline complete. Results in $VIDEO_DIR_ABS/visualizations/ ==="
