import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from device_utils import get_device
from pipeline_mps_gray.video_to_gray_frames import video_to_gray_frames


def _run_cmd(cmd, env):
    print(f"----- Running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def _build_preprocessing_config(
    preprocessing_config_path: Path,
    output_config_path: Path,
    max_frames_override: int | None,
    enable_direct_flow: bool,
):
    with open(preprocessing_config_path, "r") as f:
        config = yaml.safe_load(f.read())

    if max_frames_override is not None:
        config["max_frames"] = int(max_frames_override)
    # Direct-flow filtering is the most memory-intensive step and commonly OOMs on MPS.
    # Keep it disabled by default for a Mac-safe pipeline unless explicitly requested.
    config["filter_using_direct_flow"] = bool(enable_direct_flow)
    if not enable_direct_flow:
        config["direct_flow_threshold"] = None

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return output_config_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end grayscale pipeline on macOS: "
            "video -> grayscale frames -> preprocessing -> training."
        )
    )
    parser.add_argument("--video-path", required=True, type=str, help="Path to source video file.")
    parser.add_argument("--data-path", required=True, type=str, help="Dataset root for generated frames/artifacts.")
    parser.add_argument("--preprocess-config", default="./config/preprocessing.yaml", type=str)
    parser.add_argument("--train-config", default="./config/train.yaml", type=str)
    parser.add_argument("--max-frames", default=None, type=int, help="Override max_frames for extraction/preprocessing.")
    parser.add_argument(
        "--enable-direct-flow",
        action="store_true",
        default=False,
        help="Enable direct-flow trajectory filtering (can be much slower/more memory intensive on MPS).",
    )
    parser.add_argument(
        "--overwrite-video-frames",
        action="store_true",
        default=False,
        help="Overwrite existing frame files in <data-path>/video.",
    )
    parser.add_argument(
        "--three-channel-frames",
        action="store_true",
        default=False,
        help="Write grayscale frames as 3-channel images (default writes single-channel).",
    )
    parser.add_argument("--skip-preprocessing", action="store_true", default=False)
    parser.add_argument("--skip-training", action="store_true", default=False)
    parser.add_argument("--python-bin", default=sys.executable, type=str, help="Python executable for subprocess steps.")
    args = parser.parse_args()

    device = get_device(log=True)
    if device.type == "cuda":
        raise RuntimeError("CUDA is not allowed in this pipeline. Expected MPS or CPU.")

    data_path = Path(args.data_path).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    video_frames_path = data_path / "video"

    written = video_to_gray_frames(
        video_path=args.video_path,
        output_folder=str(video_frames_path),
        overwrite=args.overwrite_video_frames,
        max_frames=args.max_frames,
        single_channel=not args.three_channel_frames,
    )
    print(f"Prepared grayscale dataset frames: {written} -> {video_frames_path}", flush=True)

    generated_cfg_path = data_path / ".generated_configs" / "preprocessing.gray.yaml"
    preprocess_cfg = _build_preprocessing_config(
        preprocessing_config_path=Path(args.preprocess_config),
        output_config_path=generated_cfg_path,
        max_frames_override=args.max_frames,
        enable_direct_flow=args.enable_direct_flow,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = env.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if not args.skip_preprocessing:
        _run_cmd(
            [
                args.python_bin,
                "./preprocessing/main_preprocessing.py",
                "--config",
                str(preprocess_cfg),
                "--data-path",
                str(data_path),
            ],
            env=env,
        )

    if not args.skip_training:
        _run_cmd(
            [
                args.python_bin,
                "./train.py",
                "--config",
                args.train_config,
                "--data-path",
                str(data_path),
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
