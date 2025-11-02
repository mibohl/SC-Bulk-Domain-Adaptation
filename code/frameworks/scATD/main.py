import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
from pathlib import Path
import sys
from typing import Dict

# Ensure project root on sys.path for shared utilities.
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(FILE_PATH,  "..", ".."))

from data_utils import prepare_source_target_datasets
from training_utils import run_scatd_benchmark

DEFAULT_SYMBOL_MAP = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "reference", "symbol_ensembl_map.txt")
DEFAULT_SOURCE_FEATURES = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "processed", "X_Gefitinib_bulk.csv")
DEFAULT_SOURCE_LABELS = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "processed", "y_Gefitinib_bulk.csv")
DEFAULT_TARGET_FEATURES = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "processed", "X_Gefitinib_JHU006.csv")
DEFAULT_TARGET_LABELS = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "processed", "y_Gefitinib_JHU006.csv")
DEFAULT_PRETRAINED = os.path.join(FILE_PATH, "pretrained_models", "checkpoint_fold1_epoch_30.pth")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate scATD using custom source/target datasets."
    )

    # --- Data inputs ---
    parser.add_argument(
        "--source-features",
        default=DEFAULT_SOURCE_FEATURES,
        help=f"Path to source feature matrix (CSV/TSV). Defaults to {DEFAULT_SOURCE_FEATURES}.",
    )
    parser.add_argument(
        "--source-labels",
        default=DEFAULT_SOURCE_LABELS,
        help=f"Path to source labels (CSV/TSV). Defaults to {DEFAULT_SOURCE_LABELS}.",
    )
    parser.add_argument(
        "--target-features",
        default=DEFAULT_TARGET_FEATURES,
        help=f"Path to target feature matrix (CSV/TSV). Defaults to {DEFAULT_TARGET_FEATURES}.",
    )
    parser.add_argument(
        "--target-labels",
        default=DEFAULT_TARGET_LABELS,
        help=f"Path to target labels (CSV/TSV). Defaults to {DEFAULT_TARGET_LABELS}.",
    )
    parser.add_argument(
        "--symbol-map",
        default=DEFAULT_SYMBOL_MAP,
        help="Optional path to symbol↔Ensembl mapping file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binarize-threshold", type=float, default=0.5)
    parser.add_argument("--source-val-fraction", type=float, default=0.2)
    parser.add_argument("--source-test-fraction", type=float, default=0.2)
    parser.add_argument("--target-val-fraction", type=float, default=0.0)
    parser.add_argument("--target-test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--invert-source-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1 - y for source labels (viability → sensitivity).",
    )
    parser.add_argument(
        "--invert-target-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 1 - y for target labels.",
    )
    parser.add_argument(
        "--wandb-mode",
        default="offline",
        choices=["offline", "online", "disabled"],
        help="How to run Weights & Biases logging.",
    )

    # --- Model configuration ---
    parser.add_argument("--drug", default="Vorinostat")
    parser.add_argument("--epochs", type=int, default=100, help="Fine-tuning epochs.")
    parser.add_argument("--epochs-classifier", type=int, default=50, help="Classifier warm-up epochs.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pretrained-model-path", default=DEFAULT_PRETRAINED)
    parser.add_argument("--balancing-strategy", default="weighted", choices=["none", "weighted", "smote"])
    parser.add_argument("--z-dim", type=int, default=421)
    parser.add_argument("--hidden-dim-layer0", type=int, default=1664)
    parser.add_argument("--hidden-dim-layer-out-z", type=int, default=359)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--tune-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable validation-based threshold selection.",
    )
    parser.add_argument(
        "--binarize-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Binarize source labels before training.",
    )

    return parser.parse_args()


def _set_wandb_mode(mode: str) -> None:
    if mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        os.environ.setdefault("WANDB_START_METHOD", "thread")
    else:
        os.environ["WANDB_MODE"] = mode


def _print_results(results: Dict[str, Dict[str, float] | float]) -> None:
    if not results:
        print("No results returned from benchmark runner.")
        return

    print("\n=== Evaluation Results ===")
    for split, payload in results.items():
        if payload is None:
            continue
        if isinstance(payload, dict):
            print(f"{split}:")
            for metric, value in payload.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        else:
            print(f"{split}: {payload}")


def _resolve_pretrained(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / path
    return str(candidate.resolve())


def main() -> None:
    args = _parse_args()
    _set_wandb_mode(args.wandb_mode)

    prepared = prepare_source_target_datasets(
        source_features=args.source_features,
        source_labels=args.source_labels,
        target_features=args.target_features,
        target_labels=args.target_labels,
        symbol_map=args.symbol_map,
        binarize_threshold=args.binarize_threshold,
        source_val_fraction=args.source_val_fraction,
        source_test_fraction=args.source_test_fraction,
        target_val_fraction=args.target_val_fraction,
        target_test_fraction=args.target_test_fraction,
        seed=args.seed,
        invert_source_labels=args.invert_source_labels,
        invert_target_labels=args.invert_target_labels,
    ).as_dict()

    benchmark_args = {
        "drug": args.drug,
        "epochs": args.epochs,
        "epochs_classifier": args.epochs_classifier,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mmd_weight": args.mmd_weight,
        "weight_decay": args.weight_decay,
        "pretrained_model_path": _resolve_pretrained(args.pretrained_model_path),
        "balancing_strategy": args.balancing_strategy,
        "z_dim": args.z_dim,
        "hidden_dim_layer0": args.hidden_dim_layer0,
        "hidden_dim_layer_out_Z": args.hidden_dim_layer_out_z,
        "patience": args.patience,
        "tune_threshold": args.tune_threshold,
        "binarize_source": args.binarize_source,
        "binarize_threshold": args.binarize_threshold,
    }

    target_file = Path(args.target_features).stem
    results = run_scatd_benchmark(
        benchmark_args,
        prepared["x_train_source"],
        prepared["y_train_source"],
        prepared["x_val_source"],
        prepared["y_val_source"],
        prepared["x_test_source"],
        prepared["y_test_source"],
        prepared["x_train_target"],
        prepared["y_train_target"],
        prepared["x_test_target"],
        prepared["y_test_target"],
        prepared["X_target_independent"],
        prepared["y_target_independent"],
        target_file=target_file,
        independent_target_file="independent_placeholder",
        seed=args.seed,
    )

    _print_results(results)


if __name__ == "__main__":
    main()
