import logging
import datetime

import torch
import torch._dynamo
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
    deepspeed: bool = False,
    compiled: bool = False,
    offload: bool = False,
    ):
    
    seed_everything(1234, workers=True)
    
    time = str(datetime.datetime.now())[:-10].replace(" ","-").replace(":","")
    
    torch.set_float32_matmul_precision('medium')
    
    model = ResNet(block="basic", layers=[2, 2, 2, 2], block_inplanes=[32, 64, 128, 256],
                              num_classes=12, n_input_channels=1, offload=offload, lr=lr)
    if compiled:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=_PATH_MODELS + "/" + time,
        filename='resnet-{epoch}',
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    
    bugnist = BugNISTDataModule(batch_size=batch_size, num_workers=num_workers,transforms=True)

    wandb_logger = WandbLogger(project="high-performance-3d", name=name)
    
    
    if fast:
        early_stopping_callback = EarlyStopping(
            monitor="val/acc", patience=100, verbose=True, mode="max", strict=False, stopping_threshold=0.9, check_on_train_epoch_end=False,
        )
    else:
        early_stopping_callback = EarlyStopping(
        monitor="val/acc", patience=10, verbose=True, mode="max", strict=False, check_on_train_epoch_end=False,
    )
        
    if deepspeed:
        if offload:
            strat = DeepSpeedStrategy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8, logging_batch_size_per_gpu=batch_size)
        else:
            strat = DeepSpeedStrategy(logging_batch_size_per_gpu=batch_size)
        
        trainer = Trainer(
            max_epochs=max_epochs,
            devices=-1,
            accelerator="gpu",
            deterministic=False,
            strategy=strat,
            precision="16-mixed",
            default_root_dir=_PROJECT_ROOT,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=25,
            logger=wandb_logger,
        )
    else:
        trainer = Trainer(
            max_epochs=max_epochs,
            devices=-1,
            accelerator="gpu",
            deterministic=False,
            default_root_dir=_PROJECT_ROOT,
            precision="16-mixed",
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=25,
            logger=wandb_logger,
        )

    trainer.fit(model, datamodule=bugnist)
