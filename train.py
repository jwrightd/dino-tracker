import argparse
from dino_tracker import DINOTracker
from models.utils import fix_random_seeds
from device_utils import get_device


if __name__ == "__main__":
    get_device(log=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--seed", default=2, type=int)
    args = parser.parse_args()

    fix_random_seeds(args.seed)
    dino_tracker = DINOTracker(args)
    dino_tracker.train()
