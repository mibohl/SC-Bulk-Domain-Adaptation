import os
import pytorch_lightning as pl

# Add project root to system path for module imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_utils import CombinedDataLoader, create_dataloader, load_data


class SCADDataModule(pl.LightningDataModule):
    def __init__(self, drug, mbS, mbT, balancing_strategy, data_split, data_path="datasets", seed: int = 42):
        super().__init__()
        self.save_hyperparameters()
        self.drug_data_path = os.path.join(self.hparams.data_path, self.hparams.drug)
        self._seed = seed

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
            os.path.join(self.drug_data_path, "source_splits"),
            os.path.join(self.drug_data_path, "target_splits"),
            batch_size_source=self.hparams.mbS,
            batch_size_target=self.hparams.mbT,
            framework="scad",
            balancing=self.hparams.balancing_strategy,
        )

        # Cache evaluation batch size to mirror training_utils behaviour
        self._eval_batch_size = max(
            len(self.x_val_source), len(self.x_test_source), len(self.x_test_target)
        )

    def train_dataloader(self):
        source_loader = create_dataloader(
            self.x_train_source,
            self.y_train_source,
            self.hparams.mbS,
            shuffle=True,
            drop_last=False,
            balancing_strategy=self.hparams.balancing_strategy,
            seed=self._seed,
        )
        target_loader = create_dataloader(
            self.x_train_target,
            self.y_train_target,
            self.hparams.mbT,
            shuffle=True,
            drop_last=False,
            balancing_strategy=self.hparams.balancing_strategy,
            seed=self._seed,
        )
        return CombinedDataLoader(
            source_loader,
            target_loader,
            length_mode="target",
            min_batch_size=5,
            resample_target=False,
        )

    def val_dataloader(self):
        return create_dataloader(
            self.x_val_source,
            self.y_val_source,
            self._eval_batch_size,
            shuffle=False,
            drop_last=False,
            seed=self._seed,
        )

    def test_dataloader(self):
        source_test_loader = create_dataloader(
            self.x_test_source,
            self.y_test_source,
            self._eval_batch_size,
            shuffle=False,
            drop_last=False,
            seed=self._seed,
        )
        target_test_loader = create_dataloader(
            self.x_test_target,
            self.y_test_target,
            self._eval_batch_size,
            shuffle=False,
            drop_last=False,
            seed=self._seed,
        )
        return [source_test_loader, target_test_loader]
