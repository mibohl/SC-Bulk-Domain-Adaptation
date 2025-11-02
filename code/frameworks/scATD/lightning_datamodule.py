"""
Data module for scATD. Loads source and target data and provides combined dataloaders
for training, and separate dataloaders for validation and testing.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Add project root to system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_utils import (
    align_dataframe_to_gene_vocab,
    convert_to_symbols,
    create_dataloader,
    load_scatd_gene_vocab,
)

DRUG_NAME_MAP = {
    "PLX4720 (A375)": "PLX4720",
    "PLX4720 (451Lu)": "PLX4720_451Lu",
}

DEFAULT_SCAD_SPLIT_ROOT = Path(__file__).resolve().parents[3] / "original_git_repos" / "SCAD" / "data" / "split_norm"


class CombinedDataLoader:
    """Helper class to combine source and target dataloaders for training."""

    def __init__(self, source_loader, target_loader, should_include_target=None):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self._should_include_target = should_include_target

    def _include_target(self) -> bool:
        if self._should_include_target is None:
            return True
        if callable(self._should_include_target):
            return bool(self._should_include_target())
        return bool(self._should_include_target)

    def __iter__(self):
        include_target = self._include_target()
        target_iter = iter(self.target_loader) if include_target else None
        for source_batch in self.source_loader:
            include_target = self._include_target()
            if include_target:
                if target_iter is None:
                    target_iter = iter(self.target_loader)
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    target_batch = next(target_iter)
                yield source_batch, target_batch
            else:
                yield source_batch, None

    def __len__(self):
        # Match the number of source batches, aligning with the reference training loop
        return len(self.source_loader)


class scATDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        drug: str,
        data_path: Optional[str],
        batch_size: int,
        balancing_strategy: str,
        split: str = "split1",
        splits_root: Optional[str] = None,
        use_scad_splits: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._seed = kwargs.get("seed", 42)
        self.class_weights = None
        self.use_scad_splits = use_scad_splits
        self.target_pool: Optional[np.ndarray] = None

        self._train_loader = None
        self._source_val_loader = None
        self._source_test_loader = None
        self._target_val_loader = None
        self._target_test_loader = None
        self._gene_vocab: Optional[list[str]] = None

    def setup(self, stage: Optional[str] = None):
        self.class_weights = None

        if self.use_scad_splits:
            self._setup_from_scad_splits()
        else:
            self._setup_from_csv_sources()

        self._prepare_and_build_loaders()

    # ------------------------------------------------------------------
    # SCAD split loaders
    # ------------------------------------------------------------------
    def _setup_from_scad_splits(self):
        split_root = self.hparams.splits_root or str(DEFAULT_SCAD_SPLIT_ROOT)
        split_root_path = Path(split_root)
        dataset_key = DRUG_NAME_MAP.get(self.hparams.drug, self.hparams.drug.replace(" ", "_"))
        base = split_root_path / dataset_key / "stratified"
        if not base.exists():
            raise FileNotFoundError(f"SCAD split directory not found at {base}")

        source_dir = base / "source_5_folds" / self.hparams.split
        target_dir = base / "target_5_folds" / self.hparams.split

        def read_matrix(path: Path) -> pd.DataFrame:
            if not path.exists():
                raise FileNotFoundError(f"Expected matrix file not found: {path}")
            return pd.read_csv(path, sep="\t", index_col=0)

        def read_labels(path: Path) -> pd.DataFrame:
            if not path.exists():
                raise FileNotFoundError(f"Expected label file not found: {path}")
            labels = pd.read_csv(path, sep="\t", index_col=0)
            # Ensure column name is consistent with other utilities
            col = labels.columns[0]
            labels = labels.rename(columns={col: "response"})
            labels["response"] = labels["response"].astype(int)
            return labels

        # Source domain
        self.x_train_source = read_matrix(source_dir / "X_train_source.tsv")
        self.x_val_source = read_matrix(source_dir / "X_val_source.tsv")
        self.x_test_source = read_matrix(source_dir / "X_test_source.tsv")

        self.y_train_source = read_labels(source_dir / "Y_train_source.tsv")
        self.y_val_source = read_labels(source_dir / "Y_val_source.tsv")
        self.y_test_source = read_labels(source_dir / "Y_test_source.tsv")

        # Target domain
        self.x_train_target = read_matrix(target_dir / "X_train_target.tsv")
        self.x_test_target = read_matrix(target_dir / "X_test_target.tsv")
        self.y_train_target = read_labels(target_dir / "Y_train_target.tsv")
        self.y_test_target = read_labels(target_dir / "Y_test_target.tsv")

        # Target validation sets are not provided in the original splits.
        self.x_val_target = None
        self.y_val_target = None

        if self.hparams.balancing_strategy == "class_weights":
            y_array = self.y_train_source["response"].to_numpy()
            classes = np.array(sorted(np.unique(y_array)))
            self.class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_array)

    # ------------------------------------------------------------------
    # Fallback loader from processed CSVs (legacy behaviour)
    # ------------------------------------------------------------------
    def _setup_from_csv_sources(self):
        if not self.hparams.data_path:
            raise ValueError("data_path must be provided when use_scad_splits is False.")

        source_drug_path = os.path.join(self.hparams.data_path, f"X_{self.hparams.drug}_bulk.csv")
        source_label_path = os.path.join(self.hparams.data_path, f"y_{self.hparams.drug}_bulk.csv")
        target_drug_path = os.path.join(self.hparams.data_path, f"X_{self.hparams.drug}_SCC47.csv")
        target_label_path = os.path.join(self.hparams.data_path, f"y_{self.hparams.drug}_SCC47.csv")

        x_source = pd.read_csv(source_drug_path, index_col=0)
        y_source = pd.read_csv(source_label_path, index_col=0)
        x_target = pd.read_csv(target_drug_path, index_col=0)
        y_target = pd.read_csv(target_label_path, index_col=0)

        gene_vocab = load_scatd_gene_vocab()
        x_source = align_dataframe_to_gene_vocab(x_source, gene_vocab, f"{self.hparams.drug}_bulk")
        x_target = align_dataframe_to_gene_vocab(x_target, gene_vocab, "SCC47")

        (
            self.x_train_source,
            self.x_val_source,
            self.y_train_source,
            self.y_val_source,
        ) = train_test_split(x_source, y_source, test_size=0.3, random_state=42)
        (
            self.x_val_source,
            self.x_test_source,
            self.y_val_source,
            self.y_test_source,
        ) = train_test_split(self.x_val_source, self.y_val_source, test_size=0.5, random_state=42)

        (
            self.x_train_target,
            self.x_val_target,
            self.y_train_target,
            self.y_val_target,
        ) = train_test_split(x_target, y_target, test_size=0.3, random_state=42)
        (
            self.x_val_target,
            self.x_test_target,
            self.y_val_target,
            self.y_test_target,
        ) = train_test_split(self.x_val_target, self.y_val_target, test_size=0.5, random_state=42)

        if self.hparams.balancing_strategy == "class_weights":
            y_array = self.y_train_source.squeeze().values
            classes = sorted(np.unique(y_array))
            self.class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_array)

    def _prepare_and_build_loaders(self):
        """Convert identifiers, align vocabularies, and build dataloaders to mirror training_utils."""
        self._gene_vocab = load_scatd_gene_vocab()

        def _prep_features(features: Optional[pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
            if features is None:
                return None
            converted = convert_to_symbols(features)
            return align_dataframe_to_gene_vocab(converted, self._gene_vocab, name)

        self.x_train_source = _prep_features(self.x_train_source, "x_train_source")
        self.x_val_source = _prep_features(self.x_val_source, "x_val_source")
        self.x_test_source = _prep_features(self.x_test_source, "x_test_source")
        self.x_train_target = _prep_features(self.x_train_target, "x_train_target")
        self.x_val_target = _prep_features(self.x_val_target, "x_val_target")
        self.x_test_target = _prep_features(self.x_test_target, "x_test_target")

        if self.hparams.balancing_strategy == "class_weights":
            dataloader_balancing = None
        else:
            dataloader_balancing = self.hparams.balancing_strategy

        self._source_train_loader = create_dataloader(
            self.x_train_source,
            self.y_train_source,
            self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            balancing_strategy=dataloader_balancing,
            seed=self._seed,
        )
        self._target_train_loader = create_dataloader(
            self.x_train_target,
            self.y_train_target,
            self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
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

        if self.x_val_target is None or self.y_val_target is None:
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

        self._source_test_loader = create_dataloader(
            self.x_test_source,
            self.y_test_source,
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

        self._train_loader = CombinedDataLoader(self._source_train_loader, self._target_train_loader)
        self.target_pool = self.x_train_target.values if self.x_train_target is not None else None

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        if self._train_loader is None:
            raise RuntimeError("Data loaders not prepared. Call setup() before requesting train dataloader.")
        return self._train_loader

    def val_dataloader(self):
        if self._source_val_loader is None:
            raise RuntimeError("Validation loader not prepared. Call setup() before requesting dataloaders.")
        return self._source_val_loader

    def test_dataloader(self):
        if self._source_test_loader is None or self._target_test_loader is None:
            raise RuntimeError("Test loaders not prepared. Call setup() before requesting dataloaders.")
        return [self._source_test_loader, self._target_test_loader]

    def get_class_weights(self):
        return self.class_weights
