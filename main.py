import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import hydra
from omegaconf import OmegaConf

from blind_robot.data import CalvinDataset, CalvinDataModule
from blind_robot.model import gpt


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    config = pl.utilities.parsing.AttributeDict(config)
    pl.seed_everything(config["seed"])
    print(config)

    datamodule = CalvinDataModule(
        train_data_dir=config.data["train_data_dir"], 
        val_data_dir=config.data["val_data_dir"], 
        keys=config.data["keys"],
        batch_size=config.data["batch_size"], 
        num_workers=config.data["num_workers"], 
        pin_memory=config.data["pin_memory"],
        )
    datamodule.setup(stage='fit')

    model = gpt(config=config)
    # Initialize a trainer
    trainer = pl.Trainer(
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )

    # Train the model ⚡
    #trainer.fit(mnist_model, train_loader)

if __name__ == "__main__":
    main()
