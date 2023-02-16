import hydra, os
from omegaconf import OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


from blind_robot.data import CalvinDataset
from blind_robot.model import gpt


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    config = pl.utilities.parsing.AttributeDict(config)
    pl.seed_everything(config["seed"])
    print(config)
    
    train_data = CalvinDataset(root_data_dir=config.data["train_data_dir"], keys=config.data["keys"])
    val_data   = CalvinDataset(root_data_dir=config.data["val_data_dir"], keys=config.data["keys"])

    train_loader = DataLoader(train_data, 
                              batch_size=config.data["batch_size"], 
                              shuffle=config.data["shuffle_train"],
                              num_workers=config.data["num_workers"], 
                              pin_memory=config.data["pin_memory"],
                             )
    val_loader = DataLoader(val_data, 
                            batch_size=config.data["batch_size"], 
                            shuffle=config.data["shuffle_val"], 
                            num_workers=config.data["num_workers"], 
                            pin_memory=config.data["pin_memory"],
                           )

    model = gpt(config=config)
    # Initialize a logger
    if config.trainer["logger"] == "tensorboard":
        logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    elif config.trainer["logger"] == "wandb":
        logger = WandbLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    else:
        logger = False

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator=config.trainer["accelerator"],
        devices=config.trainer["devices"],
        max_epochs=config.trainer["max_epochs"],
        max_steps=config.trainer["max_steps"],
        gradient_clip_val=config.trainer["gradient_clip_val"],
        precision=config.trainer["precision"],
        accumulate_grad_batches=config.trainer["accumulate_grad_batches"],
        check_val_every_n_epoch=config.trainer["check_val_every_n_epoch"],
        auto_scale_batch_size=config.trainer["auto_scale_batch_size"],
        auto_lr_find=config.trainer["auto_lr_find"],
        strategy=config.trainer["strategy"],
        resume_from_checkpoint=config.trainer["resume_from_checkpoint"],
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        default_root_dir=config.trainer["default_root_dir"],
        logger=logger,
    )

    # Train the model ⚡
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
