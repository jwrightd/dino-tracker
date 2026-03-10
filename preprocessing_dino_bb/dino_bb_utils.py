import torch
from device_utils import get_device

def create_meshgrid(h, w, step=7, patch_size=14, return_hw=False, device=None):
        if device is None:
            device = get_device()
        start_coord = patch_size//2
        x = torch.arange(start_coord, w, step=step, device=device).float()
        y = torch.arange(start_coord, h, step=step, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        grid = torch.stack([xx, yy], dim=-1)
        if return_hw:
            return grid, len(y), len(x)
        return grid

def xy_to_fxy(xy, stride=7, patch_size=14):
    fxy = (xy - (patch_size // 2)) / stride
    return fxy
