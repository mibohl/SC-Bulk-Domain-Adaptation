# Standard library imports
import os
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# other imports
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Module constants and globals
SCATD_GENE_VOCAB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "datasets", "reference", "OS_scRNA_gene_index.19264.tsv")
)
SYMBOL_MAP_DF = pd.DataFrame()
symbol_to_ensembl = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "datasets", "reference","symbol_ensembl_map.txt")
)


def drop_all_nan_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-NaN columns, merge duplicate columns (sum), and fill remaining NaNs with 0."""
    df = df.dropna(axis=1, how="all")
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).sum().T
    return df.fillna(0.0)


def intersect_genes(*dfs: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]:
    """Intersect passed dataframes on their column names and return aligned copies plus the shared vocabulary."""
    common = set(dfs[0].columns)
    for frame in dfs[1:]:
        common &= set(frame.columns)
    common_genes = sorted(list(common))
    return [frame[common_genes].copy() for frame in dfs], common_genes


def normalize_cpm_log1p_if_counts(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """Detect raw counts, apply CPM normalisation and log1p; otherwise return dataframe unchanged."""
    df = df.apply(pd.to_numeric, errors="coerce")
    is_counts = (df.min().min() >= 0) and (df.max().max() > 20)
    if not is_counts:
        print(f"{df_name} appears already normalized/scaled; skipping CPM.")
        return df

    print(f"Applying CPM and log1p to {df_name}...")
    row_sums = df.sum(axis=1).replace(0, np.nan)
    df_cpm = df.div(row_sums, axis=0) * 1_000_000.0
    df_cpm = df_cpm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.log1p(df_cpm)


def maybe_restore_raw_counts(
    df: Optional[pd.DataFrame],
    df_name: str,
    log_threshold: float = 20.0,
    fractional_threshold: float = 0.2,
) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Detect log-scaled expression matrices and revert them back to approximate count space.

    Returns a tuple of (processed_dataframe, expm1_applied_flag).
    """
    if df is None or df.empty:
        return df, False

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=np.float64, copy=True)

    with np.errstate(invalid="ignore"):
        finite_mask = np.isfinite(values)

    if not finite_mask.any():
        warnings.warn(f"{df_name}: no finite values found; returning zeros.")
        return numeric_df.fillna(0.0), False

    max_val = np.nanmax(values)
    min_val = np.nanmin(values)
    if np.isnan(max_val) or np.isnan(min_val):
        return numeric_df.fillna(0.0), False

    fractional = np.abs(values - np.round(values))
    frac_ratio = np.mean(fractional[finite_mask] > 1e-3)
    is_log_scaled = min_val >= 0 and max_val <= log_threshold and frac_ratio > fractional_threshold

    if not is_log_scaled:
        return numeric_df.fillna(0.0), False

    print(
        f"scATD benchmark: detected log-scaled inputs for {df_name}; applying expm1 to restore raw-count scale."
    )
    restored = np.expm1(values)
    restored[~finite_mask] = 0.0
    restored = np.clip(restored, a_min=0.0, a_max=None)
    restored_df = pd.DataFrame(restored, index=numeric_df.index, columns=numeric_df.columns)
    return restored_df, True


def initialize_symbol_map(map_path: str) -> None:
    """Load the symbol-to-Ensembl mapping into the module-level cache."""
    global SYMBOL_MAP_DF
    try:
        if not os.path.exists(map_path):
            warnings.warn(f"symbol_ensembl_map not found at: {map_path}. Proceeding without mapping.")
            SYMBOL_MAP_DF = pd.DataFrame()
            return
        SYMBOL_MAP_DF = pd.read_csv(map_path)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Failed to load symbol map at {map_path}: {exc}. Proceeding without mapping.")
        SYMBOL_MAP_DF = pd.DataFrame()


def load_scatd_gene_vocab() -> list[str]:
    """Load the scFoundation/scATD gene vocabulary (19,264 genes) from the reference TSV."""
    if not os.path.exists(SCATD_GENE_VOCAB_PATH):
        raise FileNotFoundError(
            f"scATD gene vocabulary not found at {SCATD_GENE_VOCAB_PATH}. "
            "Place OS_scRNA_gene_index.19264.tsv in datasets/reference."
        )

    vocab_df = pd.read_csv(SCATD_GENE_VOCAB_PATH, sep="\t")
    gene_col = "gene_name" if "gene_name" in vocab_df.columns else vocab_df.columns[0]
    gene_vocab = (
        vocab_df[gene_col].dropna().astype(str).str.strip().tolist()
    )
    return list(dict.fromkeys(gene_vocab))


def align_dataframe_to_gene_vocab(df: Optional[pd.DataFrame], gene_vocab: list[str], df_name: str = "dataset"):
    """
    Align a DataFrame to the scATD gene vocabulary by dropping, imputing, and reordering columns.
    """
    if df is None:
        return None

    df_aligned = df.copy()
    df_aligned.columns = df_aligned.columns.astype(str).str.strip()

    if df_aligned.columns.duplicated().any():
        duplicates = df_aligned.columns[df_aligned.columns.duplicated()].unique()
        print(
            f"scATD alignment ({df_name}): found {len(duplicates)} duplicate gene(s), summing values: {list(duplicates)[:5]}..."
        )
        df_aligned = df_aligned.groupby(df_aligned.columns, axis=1).sum()

    vocab_set = set(gene_vocab)
    matched = len([col for col in df_aligned.columns if col in vocab_set])
    extra = max(len(df_aligned.columns) - matched, 0)
    missing = len(gene_vocab) - matched

    if extra > 0:
        print(f"scATD alignment ({df_name}): dropping {extra} genes not in pretrained vocab.")
    if missing > 0:
        print(f"scATD alignment ({df_name}): imputing {missing} missing genes with zeros.")

    df_aligned = df_aligned.reindex(columns=gene_vocab, fill_value=0.0)
    return df_aligned.astype(np.float32)


def create_dataloader(
    x,
    y,
    batch_size,
    shuffle: bool = True,
    sampler=None,
    drop_last: bool = False,
    balancing_strategy: Optional[str] = None,
    seed: int = 42,
):
    """
    Create a PyTorch DataLoader from pandas inputs with optional SMOTE/weighting strategies.
    """
    x_tensor = torch.FloatTensor(x.values)

    if y.empty:
        y_values = np.array([])
    elif isinstance(y, pd.DataFrame):
        y_values = y["response"].values if "response" in y.columns else y.iloc[:, 0].values
    elif isinstance(y, pd.Series):
        y_values = y.values
    else:
        raise TypeError(f"Unsupported type for y: {type(y)}. Expected pandas DataFrame or Series.")

    if balancing_strategy == "smote":
        print("Applying SMOTE for class balancing...")
        try:
            smote = SMOTE(random_state=seed)
            x_resampled, y_resampled = smote.fit_resample(x.values, y_values)

            tl = TomekLinks(sampling_strategy="auto")
            x_resampled, y_resampled = tl.fit_resample(x_resampled, y_resampled)

            print(f"Original dataset shape: {Counter(y_values)}")
            print(f"Resampled dataset shape: {Counter(y_resampled)}")

            x_tensor = torch.FloatTensor(x_resampled)
            y_tensor = torch.FloatTensor(y_resampled).squeeze()
        except ValueError as exc:
            print(f"Warning: Could not apply SMOTE ({exc}). Falling back to standard sampling.")
            y_tensor = torch.FloatTensor(y_values).squeeze()
    else:
        y_tensor = torch.FloatTensor(y_values).squeeze()

    dataset = TensorDataset(x_tensor, y_tensor)

    if sampler:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last)

    if balancing_strategy == "weighted":
        y_array = np.array(y_values >= 0.5).astype(int)
        class_counts = Counter(y_array)
        total_samples = len(y_array)
        num_classes = len(class_counts)
        class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
        sample_weights = torch.DoubleTensor([class_weights[label] for label in y_array])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def create_shot_dataloaders(
    x_train,
    y_train,
    x_val,
    y_val,
    batch_size,
    n_shot,
    seed: int = 42,
    balancing_strategy: Optional[str] = None,
):
    """Create labeled and unlabeled dataloaders for n-shot learning in SSDA4Drug."""
    random.seed(seed)
    x_train.index = y_train.index

    sample_0_train = random.sample(y_train[y_train == 0].index.tolist(), n_shot)
    sample_1_train = random.sample(y_train[y_train == 1].index.tolist(), n_shot)
    train_labeled_idx = sample_0_train + sample_1_train

    y_train_labeled = y_train.loc[train_labeled_idx]
    x_train_labeled = x_train.loc[train_labeled_idx]

    drop_last_labeled = True
    if len(y_train_labeled) < batch_size * 2 or len(y_train_labeled) < batch_size:
        drop_last_labeled = False

    target_train_labeled_loader = create_dataloader(
        x_train_labeled,
        y_train_labeled,
        batch_size,
        shuffle=True,
        drop_last=drop_last_labeled,
        balancing_strategy=balancing_strategy,
        seed=seed,
    )

    train_unlabeled_idx = y_train.index.drop(train_labeled_idx)
    y_train_unlabeled = y_train.loc[train_unlabeled_idx]
    x_train_unlabeled = x_train.loc[train_unlabeled_idx]

    effective_batch_size = min(batch_size, len(y_train_unlabeled))
    drop_last_unlabeled = True
    if balancing_strategy == "weighted" and len(y_train_unlabeled) < batch_size * 2:
        drop_last_unlabeled = False
    elif len(y_train_unlabeled) < batch_size:
        drop_last_unlabeled = False

    target_train_unlabeled_loader = create_dataloader(
        x_train_unlabeled,
        y_train_unlabeled,
        effective_batch_size,
        shuffle=True,
        drop_last=drop_last_unlabeled,
        balancing_strategy=balancing_strategy,
        seed=seed,
    )

    target_val_labeled_loader = create_dataloader(
        x_val, y_val, effective_batch_size, shuffle=False, seed=seed
    )

    dataloader_labeled_target = {"train": target_train_labeled_loader, "val": target_val_labeled_loader}
    dataloader_unlabeled_target = {"train": target_train_unlabeled_loader}

    return dataloader_labeled_target, dataloader_unlabeled_target


def _load_data_splits(path, prefix):
    """Helper function to load train, val, and test splits."""
    x_train = pd.read_csv(os.path.join(path, f"{prefix}train.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(path, "y_train.csv"), index_col=0)

    x_val_path = os.path.join(path, f"{prefix}val.csv")
    y_val_path = os.path.join(path, "y_val.csv")

    if os.path.exists(x_val_path) and os.path.exists(y_val_path):
        x_val = pd.read_csv(x_val_path, index_col=0)
        y_val = pd.read_csv(y_val_path, index_col=0)
        print(f"DEBUG: Loaded validation data from {path}")
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train["response"] if "response" in y_train.columns else y_train.iloc[:, 0],
        )
        print(f"DEBUG: Created validation split from training data in {path}")

    x_test = pd.read_csv(os.path.join(path, f"{prefix}test.csv"), index_col=0)
    y_test = pd.read_csv(os.path.join(path, "y_test.csv"), index_col=0)
    print(f"DEBUG: Loaded data from {path}")
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_data(
    source_path,
    target_path,
    batch_size_source: int = 32,
    batch_size_target: int = 32,
    framework: str = "scad",
    n_shot: int = 3,
    seed: int = 42,
    balancing: Optional[str] = "smote",
):
    """Load and prepare data for a given framework."""
    if framework in {"scad", "scdeal"}:
        return _load_data_standard(
            source_path,
            target_path,
            batch_size_source,
            batch_size_target,
            balancing,
            framework,
        )
    if framework == "ssda4drug_shot":
        return _load_data_for_shot(source_path, target_path, batch_size_source, n_shot, seed)
    raise ValueError(f"Unknown framework: {framework}")


def _load_data_standard(source_path, target_path, batch_size_source, batch_size_target, balancing, framework):
    """Load train/val/test splits for source and target domains and build dataloaders."""
    (
        x_train_source,
        y_train_source,
        x_val_source,
        y_val_source,
        x_test_source,
        y_test_source,
    ) = _load_data_splits(source_path, "X_")
    (
        x_train_target,
        y_train_target,
        x_val_target,
        y_val_target,
        x_test_target,
        y_test_target,
    ) = _load_data_splits(target_path, "X_")

    drop_last_train = framework != "scdeal"

    source_train_loader = create_dataloader(
        x_train_source,
        y_train_source,
        batch_size_source,
        shuffle=True,
        drop_last=drop_last_train,
        balancing_strategy=balancing,
    )
    target_train_loader = create_dataloader(
        x_train_target,
        y_train_target,
        batch_size_target,
        shuffle=True,
        drop_last=drop_last_train,
        balancing_strategy=balancing,
    )

    source_val_loader = create_dataloader(x_val_source, y_val_source, batch_size_source, shuffle=True, drop_last=False)
    target_val_loader = create_dataloader(x_val_target, y_val_target, batch_size_target, shuffle=True, drop_last=False)
    source_test_loader = create_dataloader(x_test_source, y_test_source, batch_size_source, shuffle=True, drop_last=False)
    target_test_loader = create_dataloader(x_test_target, y_test_target, batch_size_target, shuffle=True, drop_last=False)

    dataloader_source = {"train": source_train_loader, "val": source_val_loader, "test": source_test_loader}
    dataloader_target = {"train": target_train_loader, "val": target_val_loader, "test": target_test_loader}

    return (
        x_train_source,
        y_train_source,
        x_val_source,
        y_val_source,
        x_test_source,
        y_test_source,
        x_train_target,
        y_train_target,
        x_val_target,
        y_val_target,
        x_test_target,
        y_test_target,
        dataloader_source,
        dataloader_target,
    )


def _load_data_for_shot(source_path, target_path, batch_size, n_shot, seed: int = 42):
    """Specialised loader for SSDA4Drug n-shot experiments."""
    (
        x_train_source,
        y_train_source,
        x_val_source,
        y_val_source,
        x_test_source,
        y_test_source,
    ) = _load_data_splits(source_path, "X_")
    (
        x_train_target_full,
        y_train_target_full,
        x_val_target_full,
        y_val_target_full,
        x_test_target_full,
        y_test_target_full,
    ) = _load_data_splits(target_path, "X_")

    x_train_source.index = y_train_source.index
    x_val_source.index = y_val_source.index
    x_test_source.index = y_test_source.index

    source_train_loader = create_dataloader(
        x_train_source, y_train_source, batch_size, shuffle=True, balancing_strategy="weighted"
    )
    source_val_loader = create_dataloader(x_val_source, y_val_source, batch_size, shuffle=True)
    dataloader_source = {"train": source_train_loader, "val": source_val_loader}

    dataloader_labeled_target, dataloader_unlabeled_target = create_shot_dataloaders(
        x_train_target_full,
        y_train_target_full,
        x_val_target_full,
        y_val_target_full,
        batch_size,
        n_shot,
        seed,
        balancing_strategy="weighted",
    )

    target_test_loader = create_dataloader(
        x_test_target_full, y_test_target_full, batch_size, shuffle=True, drop_last=True
    )
    source_test_loader = create_dataloader(x_test_source, y_test_source, batch_size, shuffle=True, drop_last=True)

    return (
        dataloader_source,
        dataloader_labeled_target,
        dataloader_unlabeled_target,
        target_test_loader,
        source_test_loader,
        x_train_source.shape[1],
    )


def convert_to_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Ensembl ID columns to gene symbols using the loaded mapping."""
    if SYMBOL_MAP_DF.empty:
        warnings.warn("Symbol map is not initialized. Cannot convert to symbols.")
        return df

    ensembl_map = SYMBOL_MAP_DF.drop_duplicates(subset=["Gene stable ID"]).set_index("Gene stable ID")["Gene name"]
    new_columns = {}
    for col in df.columns:
        ensembl_id = col.split(".")[0]
        if ensembl_id in ensembl_map:
            new_columns[col] = ensembl_map[ensembl_id]

    df_renamed = df.rename(columns=new_columns)
    return df_renamed.groupby(df_renamed.columns, axis=1).sum()


def convert_to_ensembl(df: pd.DataFrame) -> pd.DataFrame:
    """Convert gene symbol columns to Ensembl IDs if needed, dropping unmatched columns."""
    if any(df.columns.str.startswith("ENSG")):
        df.columns = [col.split(".")[0] for col in df.columns]
        return df

    symbol_map: dict[str, str] = {}
    for _, row in symbol_to_ensembl.iterrows():
        if row["Gene name"] in df.columns:
            symbol_map[row["Gene name"]] = row["Gene stable ID"]
        elif row["Gene Synonym"] in df.columns:
            symbol_map[row["Gene Synonym"]] = row["Gene stable ID"]

    selected_cols = [col for col in df.columns if col in symbol_map]
    df = df[selected_cols]
    df.columns = [symbol_map[col] for col in selected_cols]
    return df


class CombinedDataLoader:
    """Helper to iterate paired source+target loaders with selectable epoch length."""

    def __init__(
        self,
        source_loader,
        target_loader,
        length_mode: str = "max",
        min_batch_size: Optional[int] = None,
        resample_target: bool = True,
    ):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.length_mode = length_mode
        self.min_batch_size = min_batch_size
        self.resample_target = resample_target
        self._it_s = None
        self._it_t = None

    def __iter__(self):
        self._it_s = iter(self.source_loader)
        self._it_t = iter(self.target_loader)
        self._count = 0
        self._epoch_len = self.__len__()
        return self

    def __next__(self):
        if self._epoch_len == 0 or self._count >= self._epoch_len:
            raise StopIteration

        attempts = 0
        max_attempts = max(self._epoch_len, 1)

        while True:
            xs, ys = self._next_source_batch()
            try:
                xt, yt = self._next_target_batch()
            except StopIteration:
                raise StopIteration

            if self.min_batch_size is not None:
                if xs.size(0) < self.min_batch_size or xt.size(0) < self.min_batch_size:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise StopIteration
                    continue

            self._count += 1
            return (xs, ys), (xt, yt)

    def __len__(self):
        if self.length_mode == "target":
            return len(self.target_loader)
        return max(len(self.source_loader), len(self.target_loader))

    def _next_source_batch(self):
        try:
            return next(self._it_s)
        except StopIteration:
            self._it_s = iter(self.source_loader)
            return next(self._it_s)

    def _next_target_batch(self):
        try:
            return next(self._it_t)
        except StopIteration:
            if self.resample_target:
                self._it_t = iter(self.target_loader)
                return next(self._it_t)
            raise


def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV table with sample index in the first column."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    suffix = path.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, index_col=0)


def _coerce_label_series(frame: pd.DataFrame | pd.Series) -> pd.Series:
    """Ensure labels are returned as a pandas Series."""
    if isinstance(frame, pd.Series):
        return frame.copy()
    if isinstance(frame, pd.DataFrame):
        if frame.shape[1] == 0:
            raise ValueError("Label dataframe has no columns.")
        if "response" in frame.columns:
            return frame["response"].copy()
        return frame.iloc[:, 0].copy()
    raise TypeError(f"Unsupported label container: {type(frame)}")


def _maybe_invert(series: pd.Series, enabled: bool) -> pd.Series:
    if not enabled:
        return series
    if not np.issubdtype(series.dtype, np.number):
        raise ValueError("Label inversion requested but labels are not numeric.")
    return 1.0 - series


def _maybe_stratify(labels: pd.Series, threshold: float) -> Optional[np.ndarray]:
    """Return binarized labels for stratified splits when both classes are present."""
    if labels.empty:
        return None
    if not np.issubdtype(labels.dtype, np.number):
        return None
    binarized = (labels.to_numpy(dtype=float) >= threshold).astype(int)
    if np.unique(binarized).size < 2:
        return None
    return binarized


@dataclass
class PreparedDatasets:
    """Container holding the processed splits consumed by the benchmark runners."""

    x_train_source: pd.DataFrame
    y_train_source: pd.Series
    x_val_source: Optional[pd.DataFrame]
    y_val_source: Optional[pd.Series]
    x_test_source: pd.DataFrame
    y_test_source: pd.Series
    x_train_target: pd.DataFrame
    y_train_target: pd.Series
    x_val_target: Optional[pd.DataFrame]
    y_val_target: Optional[pd.Series]
    x_test_target: pd.DataFrame
    y_test_target: pd.Series
    X_target_independent: Optional[pd.DataFrame]
    y_target_independent: Optional[pd.Series]

    def as_dict(self) -> Dict[str, Optional[pd.DataFrame]]:
        return {
            "x_train_source": self.x_train_source,
            "y_train_source": self.y_train_source,
            "x_val_source": self.x_val_source,
            "y_val_source": self.y_val_source,
            "x_test_source": self.x_test_source,
            "y_test_source": self.y_test_source,
            "x_train_target": self.x_train_target,
            "y_train_target": self.y_train_target,
            "x_val_target": self.x_val_target,
            "y_val_target": self.y_val_target,
            "x_test_target": self.x_test_target,
            "y_test_target": self.y_test_target,
            "X_target_independent": self.X_target_independent,
            "y_target_independent": self.y_target_independent,
        }


def prepare_source_target_datasets(
    source_features: str,
    source_labels: str,
    target_features: str,
    target_labels: str,
    *,
    symbol_map: Optional[str] = None,
    binarize_threshold: float = 0.5,
    source_val_fraction: float = 0.2,
    source_test_fraction: float = 0.2,
    target_val_fraction: float = 0.0,
    target_test_fraction: float = 0.2,
    seed: int = 42,
    invert_source_labels: bool = False,
    invert_target_labels: bool = False,
) -> PreparedDatasets:
    """
    Load, preprocess, and split source/target datasets to match the benchmark expectations.

    The preprocessing mirrors the logic used in `hyper_tuning_original.py`:
      * map gene symbols to Ensembl IDs (when possible)
      * drop all-NaN / duplicate columns and normalise via CPM+log1p when counts are detected
      * align source/target on the shared gene vocabulary
      * create train/val/test splits with stratification when labels are binary
      * fit a StandardScaler on source training data and apply it to every split
    """
    if symbol_map:
        initialize_symbol_map(symbol_map)

    source_features_df = drop_all_nan_and_deduplicate(
        convert_to_ensembl(_read_table(Path(source_features)))
    )
    target_features_df = drop_all_nan_and_deduplicate(
        convert_to_ensembl(_read_table(Path(target_features)))
    )

    source_features_df = normalize_cpm_log1p_if_counts(source_features_df, "source_features")
    target_features_df = normalize_cpm_log1p_if_counts(target_features_df, "target_features")

    source_labels_series = _maybe_invert(
        _coerce_label_series(_read_table(Path(source_labels))),
        invert_source_labels,
    )
    target_labels_series = _maybe_invert(
        _coerce_label_series(_read_table(Path(target_labels))),
        invert_target_labels,
    )

    (source_aligned, target_aligned), shared_genes = intersect_genes(
        source_features_df, target_features_df
    )
    if not shared_genes:
        raise ValueError("No overlapping genes between source and target datasets.")

    source_features_df = source_aligned
    target_features_df = target_aligned

    source_test_fraction = float(np.clip(source_test_fraction, 1e-6, 0.9))
    target_test_fraction = float(np.clip(target_test_fraction, 1e-6, 0.9))
    source_val_fraction = float(np.clip(source_val_fraction, 0.0, 0.9))
    target_val_fraction = float(np.clip(target_val_fraction, 0.0, 0.9))

    source_strat = _maybe_stratify(source_labels_series, binarize_threshold)
    X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
        source_features_df,
        source_labels_series,
        test_size=source_test_fraction,
        random_state=seed,
        stratify=source_strat,
    )

    X_source_val: Optional[pd.DataFrame] = None
    y_source_val: Optional[pd.Series] = None
    if source_val_fraction > 0.0:
        effective_val_fraction = source_val_fraction / (1.0 - source_test_fraction)
        effective_val_fraction = float(np.clip(effective_val_fraction, 1e-6, 0.9))
        source_strat_inner = _maybe_stratify(y_source_train, binarize_threshold)
        X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
            X_source_train,
            y_source_train,
            test_size=effective_val_fraction,
            random_state=seed,
            stratify=source_strat_inner,
        )

    target_strat = _maybe_stratify(target_labels_series, binarize_threshold)
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        target_features_df,
        target_labels_series,
        test_size=target_test_fraction,
        random_state=seed,
        stratify=target_strat,
    )

    X_target_val: Optional[pd.DataFrame] = None
    y_target_val: Optional[pd.Series] = None
    if target_val_fraction > 1e-6:
        effective_target_val = target_val_fraction / (1.0 - target_test_fraction)
        effective_target_val = float(np.clip(effective_target_val, 1e-6, 0.9))
        target_strat_inner = _maybe_stratify(y_target_train, binarize_threshold)
        X_target_train, X_target_val, y_target_train, y_target_val = train_test_split(
            X_target_train,
            y_target_train,
            test_size=effective_target_val,
            random_state=seed,
            stratify=target_strat_inner,
        )

    scaler = StandardScaler()
    scaler.fit(X_source_train)
    feature_order = list(scaler.feature_names_in_)

    def _scale(frame: pd.DataFrame) -> pd.DataFrame:
        data = scaler.transform(frame[feature_order])
        return pd.DataFrame(data, index=frame.index, columns=feature_order)

    X_source_train = _scale(X_source_train)
    X_source_test = _scale(X_source_test)
    if X_source_val is not None:
        X_source_val = _scale(X_source_val)

    def _scale_target(frame: pd.DataFrame | None) -> Optional[pd.DataFrame]:
        if frame is None:
            return None
        aligned = frame.reindex(columns=feature_order).fillna(0.0)
        data = scaler.transform(aligned)
        return pd.DataFrame(data, index=aligned.index, columns=feature_order)

    X_target_train = _scale_target(X_target_train)
    X_target_test = _scale_target(X_target_test)
    X_target_val = _scale_target(X_target_val)

    return PreparedDatasets(
        x_train_source=X_source_train.astype(np.float32),
        y_train_source=y_source_train,
        x_val_source=None if X_source_val is None else X_source_val.astype(np.float32),
        y_val_source=y_source_val,
        x_test_source=X_source_test.astype(np.float32),
        y_test_source=y_source_test,
        x_train_target=X_target_train.astype(np.float32),
        y_train_target=y_train_target,
        x_val_target=None if X_target_val is None else X_target_val.astype(np.float32),
        y_val_target=y_val_target,
        x_test_target=X_target_test.astype(np.float32),
        y_test_target=y_test_target,
        X_target_independent=None,
        y_target_independent=None,
    )
