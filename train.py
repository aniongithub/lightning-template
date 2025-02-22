import os

from pytorch_lightning.cli import LightningCLI

from datamodules.mnist_datamodule import MNISTDataModule
from models.mnist_litmodule import MNISTLitModule


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=MNISTLitModule,
        datamodule_class=MNISTDataModule,
        seed_everything_default=42,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None}
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # fit() runs the training
    cli.trainer.fit(
        cli.model, 
        datamodule = cli.datamodule)

if __name__ == "__main__":
    main()