import argparse
import os
import subprocess
import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.main_preprocessing import add_config_paths
from device_utils import get_device


if __name__ == "__main__":
    get_device(log=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config path")
    parser.add_argument("--data-path", default="./dataset/libby", type=str)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())
        config = add_config_paths(args.data_path, config)

    max_frames = config.get("max_frames", 400)
    
    dino_bb_dir = config['dino_bb_dir']
    dino_bb_path = os.path.join(dino_bb_dir, "dino_best_buddies.pt")
    dino_bb_filtered_path = os.path.join(dino_bb_dir, "dino_best_buddies_filtered.pt")
    traj_path = config['unfiltered_trajectories_file']
    dino_emb_path = config['dino_embed_video_path']
    dino_bb_stride = config['dino_stride']
    affinity_chunk_size = config.get('dino_bb_affinity_chunk_size', 1024)
    nms_affinity_chunk_size = config.get('dino_bb_nms_affinity_chunk_size', affinity_chunk_size)
    nms_num_cpu_threads = config.get('dino_bb_nms_num_cpu_threads', None)
    box_size = config['dino_bb_box_size']
    iou_thresh = config['dino_bb_iou_threshold']
    h, w = config['video_resh'], config['video_resw']
    
    # 1. extract_dino_feature_best_buddies.py
    args_list = ['python', 'preprocessing_dino_bb/extract_dino_best_buddies.py', 
                    '--dino-emb-path', dino_emb_path, '--stride', str(dino_bb_stride), '--out-path', dino_bb_path, '--h', str(h), '--w', str(w),
                    '--affinity-chunk-size', str(affinity_chunk_size)]
    print(f"----- Running {' '.join(args_list)}", flush=True)
    subprocess.run(args_list, check=True)

    # 2. compute raft optical flows trajectories without direct flow filtering, used for DINO-BB filtering
    args_list = ['python', './preprocessing/extract_trajectories.py', 
                    '--frames-path', config['video_folder'], '--output-path', traj_path, '--min-trajectory-length', str(config['min_trajectory_length']),
                    '--threshold', str(config['threshold']), '--infer-res-size', str(h), str(w)]
    if max_frames is not None:
        args_list += ['--max-frames', str(max_frames)]
    print(f"----- Running {' '.join(args_list)}", flush=True)
    subprocess.run(args_list, check=True)

    # 3. of_filter_dino_feature_best_buddies.py
    args_list = ['python', 'preprocessing_dino_bb/of_filter_dino_best_buddies.py',
                    '--dino-bb-path', dino_bb_path, '--traj-path', traj_path, '--out-path', dino_bb_filtered_path, 
                    '--dino-bb-stride', str(dino_bb_stride), '--h', str(h), '--w', str(w)]
    print(f"----- Running {' '.join(args_list)}", flush=True)
    subprocess.run(args_list, check=True)

    # 4. extract_bb_nms.py
    args_list = ['python', 'preprocessing_dino_bb/compute_dino_bb_nms.py', 
                 '--dino-bb-path', dino_bb_filtered_path, '--dino-emb-path', dino_emb_path, '--out-path', dino_bb_filtered_path,
                 '--stride', str(dino_bb_stride), '--h', str(h), '--w', str(w),
                 '--box-size', str(box_size), '--iou-thresh', str(iou_thresh),
                 '--affinity-chunk-size', str(nms_affinity_chunk_size)]
    if nms_num_cpu_threads is not None:
        args_list += ['--num-cpu-threads', str(nms_num_cpu_threads)]
    print(f"----- Running {' '.join(args_list)}", flush=True)
    subprocess.run(args_list, check=True)
