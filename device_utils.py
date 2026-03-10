import torch
import torch.nn.functional as nnf

_DEVICE_LOGGED = False


def get_device(log: bool = False) -> torch.device:
    global _DEVICE_LOGGED
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if log and not _DEVICE_LOGGED:
        print(f"Using device: {device.type}")
        _DEVICE_LOGGED = True
    return device


def clear_device_cache(device: torch.device) -> None:
    device = torch.device(device)
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def grid_sample_border_safe(input_tensor, grid, mode="bilinear", align_corners=True):
    """
    MPS currently does not support padding_mode='border' in grid_sample.
    Emulate border behavior by clamping the grid and using zero padding.
    """
    if input_tensor.device.type == "mps":
        grid = torch.clamp(grid, -1.0, 1.0)
        if input_tensor.dim() in (4, 5):
            # MPS currently misses grid_sampler_{2d,3d}_backward in many builds.
            # Run 4D/5D grid_sample on CPU so autograd can complete.
            input_cpu = input_tensor.to("cpu")
            grid_cpu = grid.to("cpu")
            output_cpu = nnf.grid_sample(
                input_cpu,
                grid_cpu,
                mode=mode,
                padding_mode="zeros",
                align_corners=align_corners,
            )
            return output_cpu.to(input_tensor.device)
        return nnf.grid_sample(
            input_tensor,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=align_corners,
        )
    return nnf.grid_sample(
        input_tensor,
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=align_corners,
    )
