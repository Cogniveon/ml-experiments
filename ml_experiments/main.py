import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI


def lightning_cli():
    LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=False,
        trainer_defaults={
            "log_every_n_steps": 5,
            "enable_model_summary": False,
        },
    )


if __name__ == "__main__":
    lightning_cli()
