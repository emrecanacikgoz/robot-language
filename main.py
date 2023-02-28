import os

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from blind_robot.data import CalvinDatasetGPT
from blind_robot.data import CalvinDatasetMLP
from blind_robot.models.gpt import GPT
from blind_robot.models.mlp import MLP


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    # set hydra
    config = omegaconf.OmegaConf.to_container(config)
    config = pl.utilities.parsing.AttributeDict(config)
    pl.seed_everything(config["seed"])
    print(config)

    # initialize dataloader
    if config.data["task"] == "gpt":
        train_data = CalvinDatasetGPT(
            data=config.data["train_data_file_tsv"],
            max_length=config.data["max_length"],
            keys=config.data["keys"],
        )
        val_data = CalvinDatasetGPT(
            data=config.data["val_data_file_tsv"],
            max_length=config.data["max_length"],
            keys=config.data["keys"],
        )
    elif config.data["task"] == "mlp":
        train_data = CalvinDatasetMLP(
            np_data=config.data["train_data_dir_npy"],
            tsv_data=config.data["train_data_file_tsv"],
            keys=config.data["keys"],
        )
        val_data = CalvinDatasetMLP(
            np_data=config.data["val_data_dir_npy"],
            tsv_data=config.data["val_data_file_tsv"],
            keys=config.data["keys"],
        )
    else:
        raise NotImplementedError("Only gpt and mlp dataloaders supported!")

    # load data
    train_loader = DataLoader(
        train_data,
        batch_size=config.data["batch_size"],
        shuffle=config.data["shuffle_train"],
        num_workers=config.data["num_workers"],
        pin_memory=config.data["pin_memory"],
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.data["batch_size"],
        shuffle=config.data["shuffle_val"],
        num_workers=config.data["num_workers"],
        pin_memory=config.data["pin_memory"],
    )

    # initialize model
    if config.data["task"] == "gpt":
        model = GPT(config=config)
    elif config.data["task"] == "mlp":
        model = MLP(config=config)
    else:
        raise NotImplementedError("Only gpt and mlp models supported!")

    # initialize logger
    if config.trainer["logger"] == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=os.getcwd(), version=1, name="lightning_logs"
        )
    elif config.trainer["logger"] == "wandb":
        logger = WandbLogger(
            save_dir=os.getcwd(),
            name=config.trainer["wandb_name"],
            project=config.trainer["wandb_project"],
        )
    else:
        logger = False

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=config.trainer["accelerator"],
        devices=config.trainer["devices"],
        max_epochs=config.trainer["max_epochs"],
        max_steps=config.trainer["max_steps"],
        gradient_clip_val=config.trainer["gradient_clip_val"],
        precision=config.trainer["precision"],
        strategy=config.trainer["strategy"],
        accumulate_grad_batches=config.trainer["accumulate_grad_batches"],
        check_val_every_n_epoch=config.trainer["check_val_every_n_epoch"],
        auto_scale_batch_size=config.trainer["auto_scale_batch_size"],
        auto_lr_find=config.trainer["auto_lr_find"],
        resume_from_checkpoint=config.trainer["resume_from_checkpoint"],
        enable_progress_bar=True,
        logger=logger,
    )

    # Train the model ⚡
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
