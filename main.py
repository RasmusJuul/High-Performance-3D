import argparse
import os
import sys

import torch

from src.models.train_model import main as train


def main(
    name: str = "test",
    max_epochs: int = 10,
    num_workers: int = 0,
    lr: float = 1e-4,
):
    torch.cuda.empty_cache()

    train(
        name=name,
        max_epochs=max_epochs,
        num_workers=num_workers,
        lr=lr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a ResNet18 on BugNIST"
    )
    parser.add_argument("--name", "-n", type=str, help="Name for wandb")

    parser.add_argument("--max-epochs", "-me", type=int, help="Number of max epochs")
    parser.add_argument(
        "--num-workers", "-nw", type=int, help="Number of threads use in loading data"
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
    )
    args = parser.parse_args()

    name = "test"
    max_epochs = 10
    num_workers = 0
    lr = 1e-4

    if args.name:
        name = args.name
    if args.max_epochs:
        max_epochs = args.max_epochs
    if args.num_workers:
        num_workers = args.num_workers
    if args.lr:
        lr = args.lr


    main(
        name=name,
        max_epochs=max_epochs,
        num_workers=num_workers,
        lr=lr,
    )