import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from dino_tracker import DINOTracker
from models.model_inference import ModelInference
from data.data_utils import get_grid_query_points
from device_utils import get_device


@torch.no_grad()
def run(args):
    device = get_device(log=True)
    dino_tracker = DINOTracker(args)
    dino_tracker.load_fg_masks()
    model = dino_tracker.get_model()
    if args.iter is not None:
        model.load_weights(args.iter)

    grid_trajectories_dir = dino_tracker.grid_trajectories_dir
    grid_occlusions_dir = dino_tracker.grid_occlusions_dir
    os.makedirs(grid_trajectories_dir, exist_ok=True)
    os.makedirs(grid_occlusions_dir, exist_ok=True)

    model_inference = ModelInference(
        model=model,
        range_normalizer=dino_tracker.range_normalizer,
        anchor_cosine_similarity_threshold=dino_tracker.config['anchor_cosine_similarity_threshold'],
        cosine_similarity_threshold=dino_tracker.config['cosine_similarity_threshold'],
    )

    orig_video_h, orig_video_w = dino_tracker.orig_video_res_h, dino_tracker.orig_video_res_w
    model_video_h, model_video_w = model.video.shape[-2], model.video.shape[-1]

    segm_mask = dino_tracker.fg_masks[args.start_frame].to(device) if args.use_segm_mask else None # H x W / None
    grid_query_points = get_grid_query_points(
        (orig_video_h, orig_video_w),
        segm_mask=segm_mask,
        device=device,
        interval=args.interval,
        query_frame=args.start_frame,
    ).to(dtype=torch.float32)
    grid_query_points = grid_query_points * torch.tensor(
        [model_video_w / orig_video_w, model_video_h / orig_video_h, 1.0],
        device=device,
        dtype=torch.float32,
    ) # resizes query points to model resolution

    total_queries = grid_query_points.shape[0]
    num_frames = model.video.shape[0]
    query_chunk_size = total_queries if args.query_chunk_size is None else max(1, args.query_chunk_size)
    save_every_chunks = max(1, args.save_every_chunks)

    final_traj_path = os.path.join(grid_trajectories_dir, "grid_trajectories.npy")
    final_occ_path = os.path.join(grid_occlusions_dir, "grid_occlusions.npy")
    partial_traj_path = os.path.join(grid_trajectories_dir, "grid_trajectories.partial.npy")
    partial_occ_path = os.path.join(grid_occlusions_dir, "grid_occlusions.partial.npy")
    progress_path = os.path.join(grid_trajectories_dir, "grid_inference_progress.txt")

    traj_memmap = np.lib.format.open_memmap(
        partial_traj_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_queries, num_frames, 2),
    )
    occ_memmap = np.lib.format.open_memmap(
        partial_occ_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_queries, num_frames),
    )

    chunk_starts = list(range(0, total_queries, query_chunk_size))
    for chunk_idx, start in enumerate(tqdm(chunk_starts, desc="Grid inference chunks")):
        end = min(start + query_chunk_size, total_queries)
        query_points_chunk = grid_query_points[start:end]
        if args.skip_occlusion:
            trajectories_full = model_inference.compute_trajectories(
                query_points_chunk,
                batch_size=args.batch_size,
            )
            grid_trajectories_chunk = trajectories_full[..., :2]
            grid_occlusions_chunk = torch.zeros(
                (grid_trajectories_chunk.shape[0], grid_trajectories_chunk.shape[1]),
                device=grid_trajectories_chunk.device,
                dtype=grid_trajectories_chunk.dtype,
            )
        else:
            grid_trajectories_chunk, grid_occlusions_chunk = model_inference.infer(
                query_points_chunk,
                batch_size=args.batch_size,
            )
        traj_memmap[start:end] = grid_trajectories_chunk[..., :2].cpu().detach().numpy().astype(np.float32, copy=False)
        occ_memmap[start:end] = grid_occlusions_chunk.cpu().detach().numpy().astype(np.float32, copy=False)

        if (chunk_idx + 1) % save_every_chunks == 0 or end == total_queries:
            traj_memmap.flush()
            occ_memmap.flush()
            with open(progress_path, "w") as f:
                f.write(f"{end}/{total_queries}\n")
            print(f"Saved intermediate grid outputs: {end}/{total_queries} queries", flush=True)

    os.replace(partial_traj_path, final_traj_path)
    os.replace(partial_occ_path, final_occ_path)
    with open(progress_path, "w") as f:
        f.write(f"completed {total_queries}/{total_queries}\n")
    print(f"Saved final outputs:\n- {final_traj_path}\n- {final_occ_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--iter", type=int, default=None, help="Iteration number of the model to load, if None, the last checkpoint is loaded.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--use-segm-mask", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--query-chunk-size", type=int, default=1024, help="Number of query points per chunk for incremental inference/save.")
    parser.add_argument("--save-every-chunks", type=int, default=1, help="Flush partial outputs every N chunks.")
    parser.add_argument("--skip-occlusion", action="store_true", default=False, help="Skip expensive occlusion computation and save zero occlusions for faster trajectory-only inference.")
    args = parser.parse_args()
    run(args)
