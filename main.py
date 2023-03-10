import os

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from blind_robot.data import CalvinDataset
from blind_robot.data import CalvinDatasetGPT
from blind_robot.models.mlp import MLP
from blind_robot.models.rnn import RNN
from blind_robot.models.gpt import GPT


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):

    # set hydra
    config = omegaconf.OmegaConf.to_container(config)
    config = pl.utilities.parsing.AttributeDict(config)
    pl.seed_everything(config["seed"])
    print(config)

    # initialize dataloader
    if (config["task"] == "mlp") or (config["task"] == "rnn"):
        train_data = CalvinDataset(
            path=config.data["train_path"],
            config=config,
        )
        val_data = CalvinDataset(
            path=config.data["val_path"],
            config=config,
        )
    elif config["task"] == "gpt":
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
    else:
        raise NotImplementedError("Only gpt and mlp dataloaders supported!")

    # load data
    train_loader = DataLoader(
        train_data,
        batch_size=config.training["batch_size"],
        shuffle=config.data["shuffle_train"],
        num_workers=config.data["num_workers"],
        pin_memory=config.data["pin_memory"],
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.training["batch_size"],
        shuffle=config.data["shuffle_val"],
        num_workers=config.data["num_workers"],
        pin_memory=config.data["pin_memory"],
    )

    # initialize model
    if config["task"] == "mlp":
        model = MLP(config=config)
    elif config["task"] == "rnn":
        model = RNN(config=config)
    elif config["task"] == "gpt":
        model = GPT(config=config)
    else:
        raise NotImplementedError("Only gpt, mlp, and rnn are supported!")

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

    # initialize trainer
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    trainer = pl.Trainer(
        accelerator=config.trainer["accelerator"],
        devices=config.trainer["devices"],
        max_epochs=config.trainer["max_epochs"],
        max_steps=config.trainer["max_steps"],
        gradient_clip_val=config.training["gradient_clip_val"],
        precision=config.trainer["precision"],
        strategy=config.trainer["strategy"],
        accumulate_grad_batches=config.trainer["accumulate_grad_batches"],
        check_val_every_n_epoch=config.trainer["check_val_every_n_epoch"],
        auto_scale_batch_size=config.trainer["auto_scale_batch_size"],
        auto_lr_find=config.trainer["auto_lr_find"],
        resume_from_checkpoint=config.trainer["resume_from_checkpoint"],
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        logger=logger,
    )

    # train the model ⚡
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()  # pylint: disable=E1120
