import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

if __name__ == "__main__":
    cli = LightningCLI(
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
