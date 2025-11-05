import os

os.environ["SCIPY_ARRAY_API"] = "1"  # requested env var

import argparse
import sys
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training_utils import (  # pylint: disable=wrong-import-position
    run_catboost_benchmark,
    run_catboost_fewshot_baseline,
    run_scad_benchmark,
    run_scdeal_benchmark,
    run_scatd_benchmark,
    run_ssda4drug_benchmark,
    set_seed,
)
from data_utils import (  # pylint: disable=wrong-import-position
    convert_to_ensembl,
    drop_all_nan_and_deduplicate,
    initialize_symbol_map,
    intersect_genes,
    normalize_cpm_log1p_if_counts,
)


# ----------------------- Constants & Defaults -----------------------
SEED = 42
DATA_DIR = "/cluster/work/bewi/members/mibohl/master_thesis/paper/datasets/processed"
SYMBOL_ENSEMBL_MAP = os.path.join(DATA_DIR, "symbol_ensembl_map.txt")
WANDB_PROJECT = "independent_target_evaluation"

DEFAULT_CATBOOST_PARAMS = {
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3.0,
    "border_count": 254,
}


class ScadArgs:
    """Container for SCAD defaults."""

    def __init__(self, drug_name: str) -> None:
        self.drug = drug_name
        self.epochs = 100
        self.data_split = "test"
        self.log_normalized = True
        self.plot_umap = False
        self.mode = "SCAD"
        self.lr = 0.001
        self.mbS = 8
        self.mbT = 8
        self.dropout = 0.5
        self.lam1 = 1.0
        self.balancing_strategy = "weighted"
        self.h_dim = 1024
        self.predictor_z_dim = 128
        self.tune_threshold = True
        self.binarize_source = True
        self.patience = 10


class ScDealArgs:
    """Container for scDEAL defaults."""

    def __init__(self, drug_name: str) -> None:
        self.drug = drug_name
        self.result_path = (
            "/cluster/scratch/mibohl/master_thesis/code/"
            "replicate_and_benchmark/RF_benchmark/scDEAL_temp/"
        )
        self.model_path = self.result_path
        self.bulk_model = self.result_path
        self.sc_model = self.result_path
        self.epochs_bulk = 50
        self.epochs = 100
        self.patience = 10
        self.batch_size = 32
        self.bulk_h_dims = "512,256"
        self.predictor_h_dims = "64,32"
        self.dropout = 0.3
        self.dimreduce = "DAE"
        self.freeze_pretrain = 0
        self.pretrain = "True"
        self.regularization_weight = 1.0
        self.data_split = "test"
        self.log_normalized = True
        self.mode = "scDEAL"
        self.lr_DA = 0.01
        self.bottleneck = 128
        self.mmd_weight = 0.25
        self.balancing_strategy = "weighted"
        self.lr = 0.01
        self.tune_threshold = True
        self.binarize_source = True


class Ssda4DrugArgs:
    """Container for SSDA4Drug defaults."""

    def __init__(self, drug_name: str) -> None:
        self.drug = drug_name
        self.epochs = 100
        self.batch_size = 128
        self.n_shot = 3
        self.encoder = "DAE"
        self.method = "adv"
        self.data_split = "test"
        self.mode = "SSDA4Drug"
        self.log_normalized = True
        self.patience = 10
        self.epsilon = 0.0
        self.lr = 0.001
        self.dropout = 0.3
        self.encoder_h_dims = "512,256"
        self.predictor_h_dims = "64,32"
        self.balancing_strategy = "weighted"
        self.tune_threshold = True
        self.binarize_source = True


class ScAtdArgs:
    """Container for scATD defaults."""

    def __init__(self, drug_name: str) -> None:
        self.drug = drug_name
        self.epochs = 100
        self.epochs_classifier = 50
        self.patience = 10
        self.data_split = "test"
        self.log_normalized = True
        self.mode = "scATD"
        self.lr = 2e-5
        self.batch_size = 64
        self.mmd_weight = 0.1
        self.balancing_strategy = "weighted"
        self.z_dim = 421
        self.hidden_dim_layer0 = 1664
        self.hidden_dim_layer_out_Z = 359
        self.pretrained_model_path = (
            "/cluster/work/bewi/members/mibohl/master_thesis/paper/"
            "code/frameworks/scATD/pretrained_models/"
            "checkpoint_fold1_epoch_30.pth"
        )
        self.tune_threshold = True
        self.binarize_source = True


# Target datasets available per drug (mirrors hyperparameter tuning script)
TARGET_FILE_NAMES: Dict[str, list[str]] = {
    "Cisplatin": ["GSE117872_HN120", "GSE117872_HN137", "GSE138267"],
    "Paclitaxel": ["GSE163836_FCIBC02", "GSE131984"],
    "Docetaxel": ["GSE140440_PC3", "GSE140440_DU145"],
    "Sorafenib": ["SCC47", "GSE175716_HCC"],
    "Gefitinib": ["GSE162045_PC9", "GSE202234_H1975", "GSE202234_PC9", "JHU006", "GSE112274_PC9"],
    "Afatinib": ["GSE228154_LT", "SCC47"],
}


def _unique_targets(targets: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""
    seen = set()
    ordered_unique = []
    for tag in targets:
        if tag not in seen:
            ordered_unique.append(tag)
            seen.add(tag)
    return ordered_unique


def _build_target_combinations() -> Dict[str, list[tuple[str, str]]]:
    """
    Generate all ordered (target, independent) pairs for each drug with >1 target dataset.
    """
    combinations: Dict[str, list[tuple[str, str]]] = {}
    for drug, targets in TARGET_FILE_NAMES.items():
        unique_targets = _unique_targets(targets)
        if len(unique_targets) <= 1:
            continue
        pairs = []
        for source_tag in unique_targets:
            for independent_tag in unique_targets:
                if source_tag == independent_tag:
                    continue
                pairs.append((source_tag, independent_tag))
        combinations[drug] = pairs
    return combinations


DRUG_TARGETS = {drug: _unique_targets(tags) for drug, tags in TARGET_FILE_NAMES.items()}
TARGET_COMBINATIONS = _build_target_combinations()

WANDB_TUNING_PROJECT_PATH = "bohl/hyper_tuning_v5"
MODEL_TARGET_HYPERS: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
_MODEL_TARGET_HYPERS_PATH: Optional[str] = None


def _coerce_to_float(value: object) -> Optional[float]:
    """Attempt to convert a value to a finite float."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _clean_config(config: Dict[str, object]) -> Dict[str, object]:
    """Normalize W&B config values to plain Python types."""
    cleaned: Dict[str, object] = {}
    for key, value in config.items():
        if isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def _fetch_best_model_hyperparams(
    project_path: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    """
    Fetch the best hyperparameters per (drug, target, model) from W&B.
    
    Selection logic:
    1. First, select runs with the highest source_val_mcc.
    2. If there are ties, use source_val_mcc + source_val_auc + source_val_auprc as tiebreaker.
    3. If still tied, warn and take the first one.
    """
    try:
        api = wandb.Api()
        runs = api.runs(project_path)
    except Exception as exc:  # pylint: disable=broad-except
        warnings.warn(f"Failed to fetch W&B runs from '{project_path}': {exc}")
        return {}

    # Collect all candidate runs grouped by (drug, target, model)
    candidates: Dict[tuple, list[Dict[str, object]]] = {}
    for run in runs:
        if getattr(run, "state", None) and run.state != "finished":
            continue

        summary_obj = getattr(run, "summary", None)
        if summary_obj is None:
            continue
        summary_dict = dict(getattr(summary_obj, "_json_dict", summary_obj))

        config_raw = getattr(run, "config", {})
        config_items = getattr(config_raw, "items", None)
        if callable(config_items):
            config_iterable = config_items()
        else:
            config_iterable = []
        config_filtered = {
            str(key): value for key, value in config_iterable if not str(key).startswith("_")
        }
        config_clean = _clean_config(config_filtered)

        drug = summary_dict.get("drug") or config_clean.get("drug")
        target = (
            summary_dict.get("target")
            or summary_dict.get("target_dataset")
            or config_clean.get("target")
            or config_clean.get("target_dataset")
        )
        model_name = (
            summary_dict.get("model_name")
            or config_clean.get("model_name")
            or config_clean.get("model_type")
            or config_clean.get("mode")
        )

        source_val_mcc = _coerce_to_float(summary_dict.get("source_val_mcc"))
        if (
            drug is None
            or target is None
            or model_name is None
            or source_val_mcc is None
        ):
            continue
        if drug not in TARGET_COMBINATIONS or target not in DRUG_TARGETS.get(drug, []):
            continue

        config_clean.pop("drug", None)
        config_clean.pop("target", None)
        config_clean.pop("target_dataset", None)
        config_clean.pop("model_name", None)
        config_clean.pop("model_type", None)
        config_clean.pop("mode", None)

        if model_name.startswith("CatBoost") and "catboost_params" not in config_clean:
            config_clean = {"catboost_params": config_clean}

        key = (drug, target, model_name)
        candidates.setdefault(key, []).append({
            "config": config_clean,
            "source_val_mcc": source_val_mcc,
            "source_val_auc": _coerce_to_float(summary_dict.get("source_val_auc")),
            "source_val_auprc": _coerce_to_float(summary_dict.get("source_val_auprc")),
            "run_id": getattr(run, "id", "unknown"),
        })

    # Select best run for each (drug, target, model) using tiered logic
    best: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for (drug, target, model_name), runs_list in candidates.items():
        # Tier 1: Find max source_val_mcc
        max_mcc = max(r["source_val_mcc"] for r in runs_list)
        tier1_runs = [r for r in runs_list if r["source_val_mcc"] == max_mcc]
        
        if len(tier1_runs) == 1:
            selected_run = tier1_runs[0]
        else:
            # Tier 2: Use combined metric as tiebreaker
            tier2_scores = []
            for r in tier1_runs:
                auc = r["source_val_auc"] if r["source_val_auc"] is not None else 0.0
                auprc = r["source_val_auprc"] if r["source_val_auprc"] is not None else 0.0
                combined = max_mcc + auc + auprc
                tier2_scores.append((combined, r))
            
            max_combined = max(score for score, _ in tier2_scores)
            tier2_runs = [r for score, r in tier2_scores if score == max_combined]
            
            if len(tier2_runs) == 1:
                selected_run = tier2_runs[0]
            else:
                # Tier 3: Still tied, warn and take first
                warnings.warn(
                    f"Multiple runs with identical metrics for {drug}/{target}/{model_name}: "
                    f"source_val_mcc={max_mcc}, combined_score={max_combined}. "
                    f"Taking the first one (run_id={tier2_runs[0]['run_id']})."
                )
                selected_run = tier2_runs[0]
        
        best.setdefault(drug, {}).setdefault(target, {})[model_name] = selected_run["config"]

    return best


def _ensure_model_hyperparams_loaded() -> Dict[str, Dict[str, Dict[str, Dict[str, object]]]]:
    """
    Lazily load tuned hyperparameters, caching them for the configured W&B project.
    """
    global MODEL_TARGET_HYPERS, _MODEL_TARGET_HYPERS_PATH  # pylint: disable=global-statement
    if not MODEL_TARGET_HYPERS or _MODEL_TARGET_HYPERS_PATH != WANDB_TUNING_PROJECT_PATH:
        MODEL_TARGET_HYPERS = _fetch_best_model_hyperparams(WANDB_TUNING_PROJECT_PATH)
        _MODEL_TARGET_HYPERS_PATH = WANDB_TUNING_PROJECT_PATH
    return MODEL_TARGET_HYPERS


def _resolve_dataset(
    data_dir: str, all_files: list[str], drug: str, prefix: str, dataset_tag: str
) -> Optional[str]:
    """Find the matching CSV filename for a given dataset tag."""
    for candidate in all_files:
        if not candidate.startswith(prefix) or not candidate.endswith(".csv"):
            continue
        if drug in candidate and dataset_tag in candidate:
            return os.path.join(data_dir, candidate)
    return None


def prepare_data(
    drug: str,
    target_tag: str,
    independent_tag: str,
    all_files: list[str],
    data_dir: str,
) -> Optional[Dict[str, object]]:
    """Load, preprocess, and split data for a drug with target and independent cohorts."""

    # Locate source datasets
    X_source_path = _resolve_dataset(data_dir, all_files, drug, "X_", "bulk")
    y_source_path = _resolve_dataset(data_dir, all_files, drug, "y_", "bulk")
    if X_source_path is None or y_source_path is None:
        warnings.warn(f"[{drug}] Missing bulk source files. Skipping.")
        return None

    # Locate target datasets
    X_target_path = _resolve_dataset(data_dir, all_files, drug, "X_", target_tag)
    y_target_path = _resolve_dataset(data_dir, all_files, drug, "y_", target_tag)
    if X_target_path is None or y_target_path is None:
        warnings.warn(f"[{drug}] Missing target files for dataset {target_tag}. Skipping.")
        return None

    # Locate independent target datasets
    X_independent_path = _resolve_dataset(data_dir, all_files, drug, "X_", independent_tag)
    y_independent_path = _resolve_dataset(data_dir, all_files, drug, "y_", independent_tag)
    if X_independent_path is None or y_independent_path is None:
        warnings.warn(
            f"[{drug}] Missing independent target files for dataset {independent_tag}. Skipping."
        )
        return None

    # Load CSVs
    X_source_raw = pd.read_csv(X_source_path, index_col=0)
    y_source_raw = pd.read_csv(y_source_path, index_col=0)
    X_target_raw = pd.read_csv(X_target_path, index_col=0)
    y_target_raw = pd.read_csv(y_target_path, index_col=0)
    X_independent_raw = pd.read_csv(X_independent_path, index_col=0)
    y_independent_raw = pd.read_csv(y_independent_path, index_col=0)

    # Convert labels
    if "viability" in y_source_raw.columns:
        y_source_values = 1.0 - y_source_raw["viability"]
    else:
        y_source_values = y_source_raw.iloc[:, 0]
    y_source_bin = (y_source_values >= 0.5).astype(int)

    y_target_series = y_target_raw.iloc[:, 0]
    y_independent_series = y_independent_raw.iloc[:, 0]

    # Preprocess expression matrices
    X_source = normalize_cpm_log1p_if_counts(
        drop_all_nan_and_deduplicate(convert_to_ensembl(X_source_raw.copy())), "X_source"
    )
    X_target = normalize_cpm_log1p_if_counts(
        drop_all_nan_and_deduplicate(convert_to_ensembl(X_target_raw.copy())), "X_target"
    )
    X_independent = normalize_cpm_log1p_if_counts(
        drop_all_nan_and_deduplicate(convert_to_ensembl(X_independent_raw.copy())),
        "X_independent",
    )

    # Align genes across all three cohorts
    (X_source, X_target, X_independent), common_genes = intersect_genes(
        X_source, X_target, X_independent
    )
    if not common_genes:
        warnings.warn(
            f"[{drug}] No common genes among source, target, and independent datasets. Skipping."
        )
        return None

    # Split datasets
    X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
        X_source, y_source_values, test_size=0.2, random_state=SEED, stratify=y_source_bin
    )
    y_source_train_bin = (y_source_train >= 0.5).astype(int)
    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_train,
        y_source_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_source_train_bin,
    )

    try:
        X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
            X_target,
            y_target_series,
            test_size=0.2,
            random_state=SEED,
            stratify=y_target_series,
        )
    except ValueError as exc:
        warnings.warn(
            f"[{drug}] Target stratified split failed for {target_tag}: {exc}. Skipping."
        )
        return None

    # Independent target remains evaluation only
    X_target_independent = X_independent
    y_target_independent = y_independent_series

    # Fit scaler on source train, apply to remaining splits
    source_scaler = StandardScaler()
    X_source_train = pd.DataFrame(
        source_scaler.fit_transform(X_source_train),
        index=X_source_train.index,
        columns=X_source_train.columns,
    )
    X_source_val = pd.DataFrame(
        source_scaler.transform(X_source_val),
        index=X_source_val.index,
        columns=X_source_val.columns,
    )
    X_source_test = pd.DataFrame(
        source_scaler.transform(X_source_test),
        index=X_source_test.index,
        columns=X_source_test.columns,
    )

    expected_cols = list(source_scaler.feature_names_in_)
    X_target_train = pd.DataFrame(
        source_scaler.transform(X_target_train.reindex(columns=expected_cols)),
        index=X_target_train.index,
        columns=expected_cols,
    )
    X_target_test = pd.DataFrame(
        source_scaler.transform(X_target_test.reindex(columns=expected_cols)),
        index=X_target_test.index,
        columns=expected_cols,
    )
    X_target_independent = pd.DataFrame(
        source_scaler.transform(X_target_independent.reindex(columns=expected_cols)),
        index=X_target_independent.index,
        columns=expected_cols,
    )

    print(
        f"[{drug}::{target_tag}] Source train label counts "
        f"{np.bincount((y_source_train >= 0.5).astype(int))}, "
        f"val counts {np.bincount((y_source_val >= 0.5).astype(int))}, "
        f"test counts {np.bincount((y_source_test >= 0.5).astype(int))}"
    )
    print(
        f"[{drug}::{target_tag}] Target train label counts "
        f"{y_target_train.value_counts().to_dict()}, "
        f"test counts {y_target_test.value_counts().to_dict()}"
    )
    print(
        f"[{drug}::{target_tag}] Independent target label counts "
        f"{y_target_independent.value_counts().to_dict()}"
    )

    return {
        "x_train_source": X_source_train,
        "y_train_source": y_source_train,
        "x_val_source": X_source_val,
        "y_val_source": y_source_val,
        "x_test_source": X_source_test,
        "y_test_source": y_source_test,
        "x_train_target": X_target_train,
        "y_train_target": y_target_train,
        "x_test_target": X_target_test,
        "y_test_target": y_target_test,
        "X_target_independent": X_target_independent,
        "y_target_independent": y_target_independent,
        "target_file": target_tag,
        "independent_target_file": independent_tag,
    }


def build_args(model_name: str, drug: str, target_tag: str) -> Optional[Dict[str, object]]:
    """Construct the argument dictionary for a model/target combination."""
    hyper_lookup = _ensure_model_hyperparams_loaded()
    model_hypers = hyper_lookup.get(drug, {}).get(target_tag, {}).get(model_name)
    if model_hypers is None:
        return None

    if model_name == "SCAD":
        args = vars(ScadArgs(drug))
    elif model_name == "SSDA4Drug":
        args = vars(Ssda4DrugArgs(drug))
    elif model_name == "scDEAL":
        args = vars(ScDealArgs(drug))
    elif model_name == "scATD":
        args = vars(ScAtdArgs(drug))
    elif model_name == "CatBoost_source_only":
        args = {
            "model_type": model_name,
            "drug": drug,
            "catboost_params": dict(DEFAULT_CATBOOST_PARAMS),
        }
    elif model_name == "CatBoost_fs":
        args = {"drug": drug}
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "CatBoost_source_only":
        cat_params = model_hypers.get("catboost_params", model_hypers)
        if not isinstance(cat_params, dict):
            cat_params = {}
        args["catboost_params"] = {**args.get("catboost_params", {}), **cat_params}
    else:
        args.update(model_hypers)
    return args


def log_results(run, results: Dict[str, object]) -> None:
    """Flatten nested metric dictionaries and log them to Weights & Biases."""
    payload: Dict[str, float] = {}
    for split_name, split_metrics in results.items():
        if isinstance(split_metrics, dict):
            for metric_name, metric_value in split_metrics.items():
                payload[f"{split_name}/{metric_name}"] = metric_value
        else:
            payload[split_name] = split_metrics
    if payload:
        run.log(payload)


def _build_wandb_config(
    model_name: str,
    model_args: Dict[str, object],
    target_tag: str,
    independent_tag: str,
) -> Dict[str, object]:
    """Prepare a consistent wandb config payload across all model families."""
    config = dict(model_args)
    # Remove framework-specific identifiers to avoid duplicate model columns.
    config.pop("mode", None)
    config.pop("model_type", None)
    config["model_name"] = model_name
    config["target_dataset"] = target_tag
    config["independent_target_dataset"] = independent_tag
    return config


def evaluate_model(
    model_name: str,
    args_dict: Dict[str, object],
    data_bundle: Dict[str, object],
) -> Dict[str, object]:
    """Dispatch to the appropriate benchmark runner."""
    if model_name == "SCAD":
        return run_scad_benchmark(args_dict, **data_bundle, seed=SEED)
    if model_name == "SSDA4Drug":
        return run_ssda4drug_benchmark(args_dict, **data_bundle, seed=SEED)
    if model_name == "scDEAL":
        return run_scdeal_benchmark(args_dict, **data_bundle, seed=SEED)
    if model_name == "scATD":
        return run_scatd_benchmark(args_dict, **data_bundle, seed=SEED)
    if model_name == "CatBoost_source_only":
        return run_catboost_benchmark(
            args_dict.get("model_type", model_name),
            data_bundle["x_train_source"],
            data_bundle["y_train_source"],
            data_bundle["x_val_source"],
            data_bundle["y_val_source"],
            data_bundle["x_test_source"],
            data_bundle["y_test_source"],
            data_bundle["x_train_target"],
            data_bundle["y_train_target"],
            data_bundle["x_test_target"],
            data_bundle["y_test_target"],
            X_target_independent=data_bundle.get("X_target_independent"),
            y_target_independent=data_bundle.get("y_target_independent"),
            seed=SEED,
            catboost_params=args_dict.get("catboost_params"),
        )
    if model_name == "CatBoost_fs":
        return run_catboost_fewshot_baseline(
            data_bundle["x_val_source"],
            data_bundle["y_val_source"],
            data_bundle["x_test_source"],
            data_bundle["y_test_source"],
            data_bundle["x_train_target"],
            data_bundle["y_train_target"],
            data_bundle["x_test_target"],
            data_bundle["y_test_target"],
            X_target_independent=data_bundle.get("X_target_independent"),
            y_target_independent=data_bundle.get("y_target_independent"),
            seed=SEED,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate domain adaptation models on independent targets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["SCAD"], # "CatBoost_source_only", "CatBoost_fs" "SSDA4Drug",  "scDEAL","scATD"  "scDEAL"
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of target dataset tags (applied across drugs).",
    )
    parser.add_argument(
        "--wandb-project",
        default=WANDB_PROJECT,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--tuning-project",
        default="bohl/hyper_tuning_v4",
        help="Weights & Biases project path (<entity>/<project>) containing tuned hyperparameters.",
    )
    args = parser.parse_args()

    initialize_symbol_map(SYMBOL_ENSEMBL_MAP)
    set_seed(SEED)

    global WANDB_TUNING_PROJECT_PATH, MODEL_TARGET_HYPERS, _MODEL_TARGET_HYPERS_PATH  # pylint: disable=global-statement
    WANDB_TUNING_PROJECT_PATH = args.tuning_project
    MODEL_TARGET_HYPERS = {}
    _MODEL_TARGET_HYPERS_PATH = None
    hyper_lookup = _ensure_model_hyperparams_loaded()
    if not hyper_lookup:
        warnings.warn(
            f"No tuned hyperparameters found in W&B project '{WANDB_TUNING_PROJECT_PATH}'. "
            "Model evaluation will be skipped."
        )
        return

    requested_targets = set(args.targets) if args.targets else None
    eligible_drugs = list(TARGET_COMBINATIONS.keys())

    if not eligible_drugs:
        warnings.warn("No drugs with multiple targets configured; nothing to evaluate.")
        return

    for drug in eligible_drugs:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and drug in f]
        print(f'\n{"=" * 25} Processing drug: {drug} {"=" * 25}')

        for target_tag, independent_tag in TARGET_COMBINATIONS.get(drug, []):
            if requested_targets and target_tag not in requested_targets:
                continue

            print(f"\n--- Preparing data for target: {target_tag} | independent: {independent_tag} ---")
            data_bundle = prepare_data(drug, target_tag, independent_tag, csv_files, DATA_DIR)
            if data_bundle is None:
                print(f"Skipping {drug}::{target_tag} due to data issues.")
                continue

            for model_name in args.models:
                if model_name not in hyper_lookup.get(drug, {}).get(target_tag, {}):
                    warnings.warn(
                        f"[{drug}] Hyperparameters missing for {model_name} on target {target_tag}. Skipping model."
                    )
                    continue

                model_args = build_args(model_name, drug, target_tag)
                if model_args is None:
                    warnings.warn(
                        f"[{drug}] Failed to build args for {model_name} on {target_tag}. Skipping."
                    )
                    continue

                run_name = f"{model_name}_{drug}_{target_tag}_to_{independent_tag}"
                wandb_config = _build_wandb_config(model_name, model_args, target_tag, independent_tag)

                print(f"\n--- Evaluating {model_name} on {drug}::{target_tag} ---")
                with wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=wandb_config,
                    reinit=True,
                ) as run:
                    try:
                        results = evaluate_model(model_name, model_args, data_bundle)
                    except ValueError as exc:
                        warnings.warn(f"{model_name} failed on {drug}::{target_tag}: {exc}")
                        run.log({"error": str(exc)})
                        continue

                    log_results(run, results)
                    print(
                        f"{model_name} results for {drug}::{target_tag}: "
                        f"source_test MCC={results.get('source_test', {}).get('mcc', 'NA')}, "
                        f"target_test MCC={results.get('target_test', {}).get('mcc', 'NA')}, "
                        f"independent_test MCC={results.get('independent_target_test', {}).get('mcc', 'NA')}"
                    )


if __name__ == "__main__":
    main()
