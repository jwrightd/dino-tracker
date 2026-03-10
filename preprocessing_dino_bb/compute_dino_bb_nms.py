import argparse
import os
import torch
import sys
from pathlib import Path
from torchvision.ops import batched_nms #, nms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_dino_bb.dino_bb_utils import create_meshgrid, xy_to_fxy
from device_utils import get_device


def _compute_source_target_sim_flat(source_f, target_fmap, chunk_size=1024):
    """
    source_f: C x N
    target_fmap: C x H x W
    returns: N x (H*W) cosine similarities
    """
    c = source_f.shape[0]
    target_flat = target_fmap.reshape(c, -1)  # C x HW
    target_norm = torch.clamp(target_flat.norm(dim=0, keepdim=True), min=1e-08)
    target_flat = target_flat / target_norm

    n = source_f.shape[1]
    sims = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        source_chunk = source_f[:, start:end]  # C x chunk
        source_norm = torch.clamp(source_chunk.norm(dim=0, keepdim=True), min=1e-08)
        source_chunk = source_chunk / source_norm
        sim_chunk = source_chunk.transpose(0, 1) @ target_flat  # chunk x HW
        sims.append(sim_chunk)

    return torch.cat(sims, dim=0)


def get_bb_sim_indices(affs_batched, coords, box_size=50, iou_thresh=0.5, topk=400, device=None):
    """  affs_batched: B x N """
    if device is None:
        device = affs_batched.device
    topk = torch.topk(affs_batched, k=topk, sorted=False, dim=1)
    filt_idx = topk.indices # B x topk
    affs_filt = topk.values # B x topk

    if affs_filt.shape[0] == 0:
        return None, None, None
    
    filt_coords = coords[filt_idx]
    xmin = filt_coords[:, :, 0] - box_size # B x topk
    xmax = filt_coords[:, :, 0] + box_size # B x topk
    ymin = filt_coords[:, :, 1] - box_size # B x topk
    ymax = filt_coords[:, :, 1] + box_size # B x topk
    # concat to get boxes shaped B x topk x 4
    boxes = torch.cat([xmin[:, :, None], ymin[:, :, None], xmax[:, :, None], ymax[:, :, None]], dim=-1) # B x topk x 4
    # get idxs shaped (B x topk) representing the batch index
    idxs = torch.arange(filt_idx.shape[0], device=device)[:, None].repeat(1, filt_idx.shape[1]).reshape(-1) # (B x topk)
    peak_indices = batched_nms(boxes.reshape(-1, 4), affs_filt.reshape(-1), idxs, iou_thresh)
    # convert peak_indices to the original indices to the  not flat indices
    peak_indices_original = torch.stack([peak_indices // filt_idx.shape[1], peak_indices % filt_idx.shape[1]], dim=-1)
    # retrieve the first two elements of the peak_indices_original for the first axis
    filt_idx_mask = torch.zeros_like(filt_idx, device=device) # B x topk
    filt_idx_mask[peak_indices_original[:, 0], peak_indices_original[:, 1]] = 1
    peak_aff_batched = affs_filt * filt_idx_mask # B x topk
    # retrieve the highest and second highest affinities for each batch
    top2 = torch.topk(peak_aff_batched, k=2, dim=1)
    top2_values, top2_indices = top2.values, top2.indices # B x 2, B x 2
    highest_affs, highest_affs_idx = top2_values[:, 0], top2_indices[:, 0] # B, B
    second_highest_affs, second_highest_affs_idx = top2_values[:, 1], top2_indices[:, 1] # B, B
    r = second_highest_affs / highest_affs 
    return None, top2_values, r


def compute_bb_nms(dino_bb_sf_tf, sf, tf, dino_emb, coords, stride, box_size, iou_thresh, affinity_chunk_size=1024):
    source_xy = dino_bb_sf_tf['source_coords']
    source_fxy = xy_to_fxy(source_xy, stride=stride) # N x 2

    target_fmap = dino_emb[tf] # C x H x W
    source_f = dino_emb[sf, :, source_fxy[:, 1].int(), source_fxy[:, 0].int()] # C x N

    source_target_sim_flat = _compute_source_target_sim_flat(
        source_f=source_f,
        target_fmap=target_fmap,
        chunk_size=affinity_chunk_size,
    ) # N x (HxW)
    
    _, peak_aff, r = get_bb_sim_indices(source_target_sim_flat, coords, box_size=box_size, iou_thresh=iou_thresh)
    dino_bb_sf_tf['peak_coords'] = None
    dino_bb_sf_tf['peak_affs'] = peak_aff # N x 2
    dino_bb_sf_tf['r'] = r # N

    return dino_bb_sf_tf

def compute_max_r(bb, bb_rev):
    for i in range(bb['target_coords'].shape[0]):
        r = bb['r'][i]
        target_coord = bb['target_coords'][i]
        rev_idx = torch.norm(bb_rev['source_coords'] - target_coord[None, :], dim=1).argmin(0)
        assert torch.norm(bb_rev['target_coords'][rev_idx] - bb['source_coords'][i]) == 0
        rev_r = bb_rev['r'][rev_idx]
        max_r = max(rev_r, r)
        bb['r'][i] = max_r
        bb_rev['r'][rev_idx] = max_r
    return bb, bb_rev


def run(args):
    device = get_device(log=True)
    compute_device = device
    if device.type == "mps":
        compute_device = torch.device("cpu")
        print("MPS detected: using CPU for DINO-BB NMS compatibility.", flush=True)

    if compute_device.type == "cpu" and args.num_cpu_threads is not None:
        torch.set_num_threads(args.num_cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, args.num_cpu_threads // 2))
        except RuntimeError:
            # Can raise if interop threads were already initialized in this process.
            pass
        print(
            f"Using CPU threads: intra_op={torch.get_num_threads()}, inter_op={torch.get_num_interop_threads()}",
            flush=True,
        )

    dino_bb = torch.load(args.dino_bb_path, map_location=compute_device) # { 'i_j': { source_coords: [N x 2] } }
    dino_emb = torch.load(args.dino_emb_path, map_location=compute_device) # t x c x h x w
    coords = create_meshgrid(h=args.h, w=args.w, step=args.stride, device=compute_device) # N x 2

    for key in tqdm(dino_bb.keys()):
        # no dino-bbs for this frame pair
        if dino_bb[key]['source_coords'] is None:
            dino_bb[key]['peak_coords'] = None
            dino_bb[key]['peak_affs'] = None
            dino_bb[key]['r'] = None
            continue
        
        # nms already computed as bb_rev
        if dino_bb[key].get('r', None) is not None:
            continue

        sf, tf = int(key.split("_")[0]), int(key.split("_")[1])
        
        bb = compute_bb_nms(
            dino_bb[f'{sf}_{tf}'],
            sf,
            tf,
            dino_emb,
            coords,
            args.stride,
            args.box_size,
            args.iou_thresh,
            affinity_chunk_size=args.affinity_chunk_size,
        )
        bb_rev = compute_bb_nms(
            dino_bb[f'{tf}_{sf}'],
            tf,
            sf,
            dino_emb,
            coords,
            args.stride,
            args.box_size,
            args.iou_thresh,
            affinity_chunk_size=args.affinity_chunk_size,
        )
        
        bb, bb_rev = compute_max_r(bb, bb_rev)
        
        dino_bb[key] = bb
        dino_bb[f'{tf}_{sf}'] = bb_rev

    dino_bb_nms_path = args.out_path
    os.makedirs(os.path.dirname(dino_bb_nms_path), exist_ok=True)
    torch.save(dino_bb, dino_bb_nms_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dino-bb-path", type=str, required=True)
    parser.add_argument("--dino-emb-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--h", type=int, default=476)
    parser.add_argument("--w", type=int, default=854)
    parser.add_argument("--box-size", type=int, default=50)
    parser.add_argument("--iou-thresh", type=float, default=0.2)
    parser.add_argument("--affinity-chunk-size", type=int, default=1024)
    parser.add_argument("--num-cpu-threads", type=int, default=None)
    args = parser.parse_args()
    run(args)
