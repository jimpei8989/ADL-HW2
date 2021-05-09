import random
from argparse import ArgumentParser
from pathlib import Path

from utils import set_seed
from utils.io import json_load, json_dump
from utils.logger import logger


def main(args):
    set_seed(args.seed)

    logger.info(f"Loading training data from {args.dataset_dir / 'train.json'}...")
    all_data = json_load(args.dataset_dir / "train.json")

    logger.info("Random shuffling the data...")
    random.shuffle(all_data)

    train_size = int(args.train_ratio * len(all_data))
    val_size = len(all_data) - train_size
    logger.info(f"Splitting the dataset into [{train_size}, {val_size}] sizes")

    train_data, val_data = all_data[:train_size], all_data[train_size:]

    json_dump(train_data, args.dataset_dir / "train_splitted.json")
    json_dump(val_data, args.dataset_dir / "val_splitted.json")


def parse_arguments():
    parser = ArgumentParser()

    # Filesystem
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset/chineseQA"))
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # Misc
    parser.add_argument("--seed", default=0x06902029)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_arguments())
