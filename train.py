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
import wandb

import os
os.environ['HYDRA_FULL_ERROR'] = '1'
torch.set_float32_matmul_precision('medium')


class FormulaReplacementCallback(Callback):
    """Callback to enable chemical formula replacement at a specific epoch."""

    def __init__(self, start_epoch: int = 0, prob: float = 0.5):
        super().__init__()
        self.start_epoch = start_epoch
        self.prob = prob

    def on_train_epoch_start(self, trainer, pl_module):
        """Enable formula replacement when reaching the start epoch."""
        if trainer.current_epoch >= self.start_epoch:
            if hasattr(trainer.datamodule, 'enable_formula_replacement'):
                if not trainer.datamodule.enable_formula_replacement:
                    trainer.datamodule.enable_formula_replacement = True
                    print(f"\n{'='*80}")
                    print(f"[Epoch {trainer.current_epoch}] Chemical formula replacement ENABLED")
                    print(f"  Replacement probability: {self.prob}")
                    print(f"{'='*80}\n")
        else:
            if hasattr(trainer.datamodule, 'enable_formula_replacement'):
                trainer.datamodule.enable_formula_replacement = False


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
                save_last=cfg.trainer.model_checkpoints.save_last,
            )
        )

    # Chemical formula replacement callback
    if cfg.hyperparams.replace_formula_prob > 0:
        hydra.utils.log.info(
            f"Adding callback <FormulaReplacementCallback> "
            f"(start_epoch={cfg.hyperparams.replace_formula_start_epoch}, "
            f"prob={cfg.hyperparams.replace_formula_prob})"
        )
        callbacks.append(
            FormulaReplacementCallback(
                start_epoch=cfg.hyperparams.replace_formula_start_epoch,
                prob=cfg.hyperparams.replace_formula_prob,
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

        # Set max_epochs for debug mode
        # If resuming from checkpoint, we need to adjust max_epochs
        if cfg.resume_from_checkpoint is not None:
            # Load checkpoint to get current epoch
            ckpt = torch.load(cfg.resume_from_checkpoint, map_location='cpu', weights_only=False)
            current_epoch = ckpt.get('epoch', 0)
            # current_epoch is the last completed epoch (0-indexed)
            # To run 1 more epoch, we need max_epochs = current_epoch + 2
            cfg.trainer.pl_trainer.max_epochs = current_epoch + 2
            print(f"[Debug Mode] Resuming from epoch {current_epoch} (completed), will run epoch {current_epoch + 1}, setting max_epochs={current_epoch + 2}")
        else:
            cfg.trainer.pl_trainer.max_epochs = 1

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # DataModule loading
    print('Loading DataModule')
    dm = hydra.utils.instantiate(cfg.model.datamodule, _recursive_=False)
    dm.setup()

    # Model loading
    if hasattr(dm, 'task'):
        if dm.task == 'classification':
            cfg.model.num_classes = len(dm.categories)
        else:
            cfg.model.num_classes = 1

    # Check if resuming from checkpoint
    ckpt_path = None
    if cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        if Path(ckpt_path).exists():
            hydra.utils.log.info(f"Will resume training from checkpoint: {ckpt_path}")
            print(f"\n{'='*80}")
            print(f"RESUMING FROM CHECKPOINT: {ckpt_path}")
            print(f"  - Model weights will be restored")
            print(f"  - Optimizer state will be restored")
            print(f"  - Training epoch will continue from checkpoint")
            print(f"{'='*80}\n")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Always instantiate model (Lightning will load weights from ckpt_path if provided)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    if hasattr(dm, 'task'):
        model.task = dm.task
        if dm.task == 'classification':
            model.label_encoder = dm.label_encoder
            model.categories = dm.categories

    # Logger instantiation/configuration
    loggers = []
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = OmegaConf.to_container(cfg.logging.wandb, resolve=True)
        settings_config = wandb_config.get("settings")
        if settings_config is not None:
            wandb_config["settings"] = wandb.Settings(**settings_config)
        wandb_logger = WandbLogger(
            **wandb_config,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )
        loggers.append(wandb_logger)

    # Always add CSV logger for local metrics collection per run
    csv_logger = CSVLogger(save_dir=str(hydra_dir), name="")
    loggers.append(csv_logger)
    
    # Trainer loading
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=build_callbacks(cfg),
        default_root_dir=hydra_dir,
        **cfg.trainer.pl_trainer
    )

    # Training
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    # Testing
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
