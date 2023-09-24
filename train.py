import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
import torch
from data.datamodule import LaMDataModule
from model.clamp import CLaMP
from model.graph_encoder import load_graph_encoder
from model.text_encoder import load_text_encoder
from configs import CLaMPConfigs

cs = ConfigStore.instance()
cs.store(name='clamp_config', node=CLaMPConfigs)

@hydra.main(config_path='config', config_name='config')
def main(cfg: CLaMPConfigs):
    # DataModule loading
    dm = LaMDataModule(cfg.paths.data, cfg.hyperparams.batch_size)

    # Model loading
    graph_encoder = load_graph_encoder(cfg.graph_encoder)
    text_encoder = load_text_encoder(cfg.text_encoder)
    model = CLaMP(graph_encoder, text_encoder, cfg.hyperparams)

    # Trainer loading
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        progress_bar_refresh_rate=20,
        logger=pl.loggers.TensorBoardLogger('logs/', name='my_model'),
        default_root_dir='logs/'
    )

    # Training
    trainer.fit(model, dm)

    # Testing
    trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    main()