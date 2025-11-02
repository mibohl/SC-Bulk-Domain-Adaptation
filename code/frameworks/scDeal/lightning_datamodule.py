import os
import pytorch_lightning as pl
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_utils import CombinedDataLoader, create_dataloader, load_data


class scDEALDataModule(pl.LightningDataModule):
    def __init__(self, drug, batch_size, balancing_strategy, data_split, data_path="datasets", seed: int = 42, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.drug_data_path = os.path.join(self.hparams.data_path, self.hparams.drug)
        self.source_path = os.path.join(self.drug_data_path, "source_splits")
        self.target_path = os.path.join(self.drug_data_path, "target_splits")
        self._seed = seed

        # Internal caches populated in setup()
        self._source_train_loader = None
        self._source_val_loader = None
        self._source_test_loader = None
        self._target_train_loader = None
        self._target_val_loader = None
        self._target_test_loader = None

    def setup(self, stage=None):
        (
            self.x_train_source,
            self.y_train_source,
            self.x_val_source,
            self.y_val_source,
            self.x_test_source,
            self.y_test_source,
            self.x_train_target,
            self.y_train_target,
            self.x_val_target,
            self.y_val_target,
            self.x_test_target,
            self.y_test_target,
            _,
            _,
        ) = load_data(
            self.source_path,
            self.target_path,
            batch_size_source=self.hparams.batch_size,
            batch_size_target=self.hparams.batch_size,
            framework="scdeal",
            balancing=self.hparams.balancing_strategy,
        )

        self._build_dataloaders()

    def _build_dataloaders(self):
        self._source_train_loader = create_dataloader(
            self.x_train_source,
            self.y_train_source,
            self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            balancing_strategy=self.hparams.balancing_strategy,
            seed=self._seed,
        )
        self._source_val_loader = create_dataloader(
            self.x_val_source,
            self.y_val_source,
            self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            seed=self._seed,
        )
        self._source_test_loader = create_dataloader(
            self.x_test_source,
            self.y_test_source,
            self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            seed=self._seed,
        )

        self._target_train_loader = create_dataloader(
            self.x_train_target,
            self.y_train_target,
            self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            balancing_strategy=self.hparams.balancing_strategy,
            seed=self._seed,
        )

        if self.x_val_target is None or self.y_val_target is None:
            # Mirror training_utils fallback: reuse training data when no dedicated target validation split exists.
            self._target_val_loader = create_dataloader(
                self.x_train_target,
                self.y_train_target,
                self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                seed=self._seed,
            )
        else:
            self._target_val_loader = create_dataloader(
                self.x_val_target,
                self.y_val_target,
                self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                seed=self._seed,
            )

        self._target_test_loader = create_dataloader(
            self.x_test_target,
            self.y_test_target,
            self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            seed=self._seed,
        )

    def train_dataloader(self):
        return CombinedDataLoader(
            self._source_train_loader,
            self._target_train_loader,
            length_mode="max",
            resample_target=True,
        )

    def val_dataloader(self):
        return self._source_val_loader

    def test_dataloader(self):
        return [self._source_test_loader, self._target_test_loader]

    # The following are needed for the pre-training in the setup hook of the LightningModule
    def source_train_dataloader(self):
        return self._source_train_loader

    def source_val_dataloader(self):
        return self._source_val_loader

    def target_train_dataloader(self):
        return self._target_train_loader

    def target_val_dataloader(self):
        return self._target_val_loader
