import argparse
import os
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
import yaml


def _resolve_smoothing_params(config_path: str | None, window_length: int | None, polyorder: int | None):
    config_window = 11
    config_polyorder = 3
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f.read()) or {}
        smoothing_cfg = config.get("smoothing", {})
        config_window = int(smoothing_cfg.get("window_length", config_window))
        config_polyorder = int(smoothing_cfg.get("polyorder", config_polyorder))

    return window_length if window_length is not None else config_window, polyorder if polyorder is not None else config_polyorder


def _validate_window(T: int, window_length: int, polyorder: int):
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 2)
    if window_length > T:
        window_length = T if T % 2 == 1 else T - 1
    if window_length <= polyorder:
        raise ValueError(f"Invalid Savitzky-Golay params for sequence length {T}: window_length={window_length}, polyorder={polyorder}")
    return window_length


def smooth_trajectory_file(input_path: str, output_path: str, window_length: int, polyorder: int, chunk_size: int = 1024):
    trajectories = np.load(input_path, mmap_mode="r")
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError(f"Expected trajectory array of shape (N, T, 2), got {trajectories.shape} from '{input_path}'.")

    n_points, n_frames, _ = trajectories.shape
    window_length = _validate_window(n_frames, window_length, polyorder)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=trajectories.dtype,
        shape=trajectories.shape,
    )

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        chunk = np.asarray(trajectories[start:end], dtype=np.float32)
        output[start:end, :, 0] = savgol_filter(chunk[:, :, 0], window_length=window_length, polyorder=polyorder, axis=1, mode="nearest")
        output[start:end, :, 1] = savgol_filter(chunk[:, :, 1], window_length=window_length, polyorder=polyorder, axis=1, mode="nearest")

    output.flush()
    return output_path, window_length


def load_and_smooth(data_path: str, config_path: str | None = None, input_path: str | None = None, output_path: str | None = None, window_length: int | None = None, polyorder: int | None = None, in_place: bool = False, chunk_size: int = 1024):
    resolved_window_length, resolved_polyorder = _resolve_smoothing_params(config_path, window_length, polyorder)

    data_root = Path(data_path)
    default_input_path = data_root / "grid_trajectories" / "grid_trajectories.npy"
    input_path = str(default_input_path if input_path is None else Path(input_path))
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No grid trajectory file found at '{input_path}'. Run inference_grid.py first.")

    if output_path is None:
        output_path = input_path if in_place else str(Path(input_path).with_name("grid_trajectories_smoothed.npy"))

    smoothed_path, used_window = smooth_trajectory_file(
        input_path=input_path,
        output_path=output_path,
        window_length=resolved_window_length,
        polyorder=resolved_polyorder,
        chunk_size=chunk_size,
    )
    print(
        f"Smoothed trajectories saved to {smoothed_path} "
        f"(window_length={used_window}, polyorder={resolved_polyorder})",
        flush=True,
    )
    return smoothed_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Savitzky-Golay smoothing to grid trajectory .npy outputs.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--config", default=None, help="Optional train config path for smoothing defaults.")
    parser.add_argument("--input-path", default=None, help="Optional explicit input .npy trajectory path.")
    parser.add_argument("--output-path", default=None, help="Optional explicit output .npy trajectory path.")
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--polyorder", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--in-place", action="store_true", default=False)
    args = parser.parse_args()

    load_and_smooth(
        data_path=args.data_path,
        config_path=args.config,
        input_path=args.input_path,
        output_path=args.output_path,
        window_length=args.window_length,
        polyorder=args.polyorder,
        in_place=args.in_place,
        chunk_size=args.chunk_size,
    )
