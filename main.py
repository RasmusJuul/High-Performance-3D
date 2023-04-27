import argparse
import os
import sys

import torch

from src.models.train_model import main as train


def main(
    name: str,
    max_epochs: int,
    num_workers: int,
    lr: float,
    fast: bool,
    batch_size: int,
    num_devices: int,
    compiled: bool,
):
    torch.cuda.empty_cache()

    train(
        name=name,
        max_epochs=max_epochs,
        num_workers=num_workers,
        lr=lr,
        fast=fast,
        batch_size=batch_size,
        num_devices=num_devices,
        compiled=compiled,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a ResNet18 on BugNIST"
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="test",
        help="Name for wandb"
    )
    parser.add_argument(
        "--max-epochs",
        "-me",
        type=int,
        default=10,
        help="Number of max epochs"
    )
    parser.add_argument(
        "--num-workers",
        "-nw",
        type=int,
        default=0,
        help="Number of threads use in loading data"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=8,
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        help="Batch size",
        default=-1,
    )
    parser.add_argument(
        "-f",
        "--fast",
        action='store_true',
        help="enable all speed increasing methods"
    )
    parser.add_argument(
        "-c",
        "--compiled",
        action='store_true',
        help="compiles model"
    )
    args = parser.parse_args()

    main(
        name=args.name,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        lr=args.lr,
        fast=args.fast,
        batch_size=args.batch_size,
        num_devices=args.num_devices,
        compiled=args.compiled,
    )