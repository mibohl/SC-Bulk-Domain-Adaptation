import os
import pytorch_lightning as pl

# Add project root to system path for module imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_utils import create_dataloader, create_shot_dataloaders, load_data



class SSDA4DrugDataModule(pl.LightningDataModule):
    def __init__(self, drug, data_path, batch_size, n_shot, seed, balancing_strategy):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        drug_data_path = os.path.join(self.hparams.data_path, self.hparams.drug)
        
        (
            self.x_train_source, self.y_train_source,
            self.x_val_source, self.y_val_source,
            self.x_test_source, self.y_test_source,
            self.x_train_target, self.y_train_target,
            self.x_val_target, self.y_val_target,
            self.x_test_target, self.y_test_target,
            _, _
        ) = load_data(
            os.path.join(drug_data_path, "source_splits"),
            os.path.join(drug_data_path, "target_splits"),
            framework="scad" # Use standard loader
        )

        self.dataloader_labeled_target, self.dataloader_unlabeled_target = create_shot_dataloaders(
            self.x_train_target,
            self.y_train_target,
            self.x_val_target,
            self.y_val_target,
            self.hparams.batch_size,
            self.hparams.n_shot,
            self.hparams.seed,
            balancing_strategy=self.hparams.balancing_strategy
        )

    def train_dataloader(self):
        source_train_loader = create_dataloader(self.x_train_source, self.y_train_source, self.hparams.batch_size, shuffle=True, balancing_strategy=self.hparams.balancing_strategy, drop_last=True)
        
        labeled_target_loader = self.dataloader_labeled_target["train"]
        unlabeled_target_loader = self.dataloader_unlabeled_target["train"]

        return {
            "source": source_train_loader,
            "labeled_target": labeled_target_loader,
            "unlabeled_target": unlabeled_target_loader
        }

    def val_dataloader(self):
        source_val_loader = create_dataloader(self.x_val_source, self.y_val_source, self.hparams.batch_size, shuffle=False)
        return source_val_loader

    def test_dataloader(self):
        source_test_loader = create_dataloader(self.x_test_source, self.y_test_source, self.hparams.batch_size, shuffle=False)
        target_test_loader = create_dataloader(self.x_test_target, self.y_test_target, self.hparams.batch_size, shuffle=False)
        return [source_test_loader, target_test_loader]
