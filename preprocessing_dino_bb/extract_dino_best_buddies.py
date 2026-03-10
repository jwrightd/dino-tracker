import os 
import torch
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from einops import rearrange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_dino_bb.dino_bb_utils import create_meshgrid
from device_utils import clear_device_cache, get_device


@torch.no_grad()
def compute_best_buddies_chunked(source_features, target_features, chunk_size=1024):
    """Compute mutual nearest-neighbor matches with chunked similarity to avoid huge einsum tensors."""
    source_features = torch.nn.functional.normalize(source_features.float(), dim=1)
    target_features = torch.nn.functional.normalize(target_features.float(), dim=1)

    n_source = source_features.shape[0]
    n_target = target_features.shape[0]
    device = source_features.device
    target_t = target_features.transpose(0, 1).contiguous()  # C x N_target

    row_max_vals = torch.empty(n_source, device=device, dtype=source_features.dtype)
    row_max_idx = torch.empty(n_source, device=device, dtype=torch.long)

    col_max_vals = torch.full((n_target,), -float("inf"), device=device, dtype=source_features.dtype)
    col_max_idx = torch.zeros(n_target, device=device, dtype=torch.long)

    for start in range(0, n_source, chunk_size):
        end = min(start + chunk_size, n_source)
        sims = source_features[start:end] @ target_t  # chunk x N_target

        best_row_vals, best_row_idx = sims.max(dim=1)
        row_max_vals[start:end] = best_row_vals
        row_max_idx[start:end] = best_row_idx

        best_col_vals, best_col_idx_local = sims.max(dim=0)
        improved = best_col_vals > col_max_vals
        col_max_vals[improved] = best_col_vals[improved]
        col_max_idx[improved] = start + best_col_idx_local[improved]

    feature_range = torch.arange(n_source, device=device)
    source_bb_indices = feature_range == col_max_idx[row_max_idx]
    target_bb_indices = row_max_idx[source_bb_indices]
    affinities = row_max_vals[source_bb_indices]
    return source_bb_indices, target_bb_indices, affinities


@torch.no_grad()
def run(args):
    device = get_device(log=True)
    dino_embed_video_path = args.dino_emb_path
    h, w = args.h, args.w
    out_path = args.out_path
    
    best_buddies = {}
    coords_grid = create_meshgrid(h, w, step=args.stride, device=torch.device("cpu"))
    
    features = torch.load(dino_embed_video_path, map_location="cpu") # T x C x H x W
    features = rearrange(features, 't c h w -> t (h w) c')
    clear_device_cache(device)

    t = features.shape[0]
    for source_t in tqdm(range(t), desc="source time"):
        for target_t in tqdm(range(t), desc="target time"):
            if source_t == target_t:
                continue
            
            source_features = features[source_t].to(device) # (h x w) x c
            target_features = features[target_t].to(device) # (h x w) x c

            try:
                source_bb_indices, target_bb_indices, affinities = compute_best_buddies_chunked(
                    source_features=source_features,
                    target_features=target_features,
                    chunk_size=args.affinity_chunk_size,
                )
            except RuntimeError as e:
                err_msg = str(e)
                if "MPSGaph" not in err_msg and "MPSGraph" not in err_msg:
                    raise
                source_cpu = source_features.cpu()
                target_cpu = target_features.cpu()
                source_bb_indices, target_bb_indices, affinities = compute_best_buddies_chunked(
                    source_features=source_cpu,
                    target_features=target_cpu,
                    chunk_size=args.affinity_chunk_size,
                )

            source_bb_indices_cpu = source_bb_indices.cpu()
            target_bb_indices_cpu = target_bb_indices.cpu()
            source_coords = coords_grid[source_bb_indices_cpu]
            target_coords = coords_grid[target_bb_indices_cpu]
            affinities = affinities.cpu()
            
            best_buddies[f'{source_t}_{target_t}'] = {
                "source_coords": source_coords,
                "target_coords": target_coords,
                "cos_sims": affinities
            }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(best_buddies, out_path)
    print(f"Saved best buddies to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-emb-path", type=str, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--affinity-chunk-size", type=int, default=1024)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()
    run(args)
