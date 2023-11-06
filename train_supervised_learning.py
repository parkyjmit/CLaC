from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
import torch
import pytorch_lightning as pl
from lightning.pytorch.strategies import DDPStrategy
# from configs import CLaMPConfigs
from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from typing import List
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from model.clamp import GraphSupervisedLearning

import os
os.environ['HYDRA_FULL_ERROR'] = '1'
torch.set_float32_matmul_precision('medium')

def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.trainer:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.trainer.monitor_metric,
                mode=cfg.trainer.monitor_metric_mode,
                patience=cfg.trainer.early_stopping.patience,
                verbose=cfg.trainer.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.trainer:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.trainer.monitor_metric,
                mode=cfg.trainer.monitor_metric_mode,
                save_top_k=cfg.trainer.model_checkpoints.save_top_k,
                verbose=cfg.trainer.model_checkpoints.verbose,
            )
        )

    return callbacks


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: DictConfig):

    if cfg.trainer.deterministic:
        seed_everything(cfg.trainer.random_seed)

    if cfg.debug:
        hydra.utils.log.info(
            f"Debug mode <{cfg.debug=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        # cfg.trainer.pl_trainer.gpus = 0
        # cfg.data.datamodule.num_workers.train = 0
        # cfg.data.datamodule.num_workers.val = 0
        # cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # DataModule loading
    print('Loading DataModule')
    dm = hydra.utils.instantiate(cfg.model.datamodule, _recursive_=False)
    dm.setup()

    # Model loading
    task = dm.task
    if task == 'classification':
        
    model = GraphSupervisedLearning(cfg.model)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )
    
    # Trainer loading
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=build_callbacks(cfg),
        default_root_dir=hydra_dir,
        **cfg.trainer.pl_trainer
    )

    # Training
    trainer.fit(model, dm)

    # Testing
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()