import logging
import datetime

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

import wandb
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.data.make_dataset import BugNISTDataModule
from src.models.model import ResNet


def main(
    name: str = "test",
    max_epochs: int = 10,
    num_workers: int = 0,
    lr: float = 1e-4,
    fast: bool = False,
    batch_size: int = 16,
    ):
    
    time = str(datetime.datetime.now())[:-10].replace(" ","-").replace(":","")
    
    if fast:
        torch.set_float32_matmul_precision('medium')
    
    model = ResNet(block="basic", layers=[2, 2, 2, 2], block_inplanes=[32, 64, 128, 256],
                              num_classes=12, n_input_channels=1, fast=fast, lr=lr)
#     if fast:
#         model = torch.compile(model)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=_PATH_MODELS + "/" + time,
        filename='resnet-{epoch}',
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val/acc", patience=5, verbose=True, mode="max", strict=False, check_on_train_epoch_end=False,
    )

    bugnist = BugNISTDataModule(batch_size=batch_size, num_workers=num_workers,transforms=True)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(project="high-performance-3d", name=name)

    seed_everything(1234, workers=True)
    
    if fast:
        early_stopping_callback = EarlyStopping(
            monitor="val/acc", patience=5, verbose=True, mode="max", strict=False, stopping_threshold=0.9, check_on_train_epoch_end=False,
        )
        trainer = Trainer(
            max_epochs=max_epochs,
            devices=1,
            accelerator="gpu",
            deterministic=False,
            strategy=DeepSpeedStrategy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8,logging_batch_size_per_gpu=batch_size),
            precision="16-mixed",
            default_root_dir=_PROJECT_ROOT,
            callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
            log_every_n_steps=25,
            logger=wandb_logger,
        )
    else:
        trainer = Trainer(
            max_epochs=max_epochs,
            devices=1,
            accelerator="gpu",
            deterministic=False,
            default_root_dir=_PROJECT_ROOT,
            callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
            log_every_n_steps=25,
            logger=wandb_logger,
        )

    trainer.fit(model, datamodule=bugnist)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
