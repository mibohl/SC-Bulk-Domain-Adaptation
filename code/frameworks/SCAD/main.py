import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
from pathlib import Path
import sys
from typing import Dict

# Ensure project root is on sys.path for shared utilities.
# Ensure project root on sys.path.
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(FILE_PATH,  "..", ".."))

from data_utils import prepare_source_target_datasets
from training_utils import run_scad_benchmark

DEFAULT_SYMBOL_MAP = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "reference", "symbol_ensembl_map.txt")
DEFAULT_SOURCE_FEATURES = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "processed", "X_Gefitinib_bulk.csv")
DEFAULT_SOURCE_LABELS = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "processed", "y_Gefitinib_bulk.csv")
DEFAULT_TARGET_FEATURES = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "processed", "X_Gefitinib_JHU006.csv")
DEFAULT_TARGET_LABELS = os.path.join(FILE_PATH , "..", "..", "..", "datasets", "processed", "y_Gefitinib_JHU006.csv")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate SCAD on custom source/target datasets."
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
        help="Optional path to symbol↔Ensembl mapping file. Defaults to datasets/processed/symbol_ensembl_map.txt.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed used across splitting and training.")
    parser.add_argument(
        "--binarize-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize labels when stratifying splits.",
    )
    parser.add_argument(
        "--source-val-fraction",
        type=float,
        default=0.2,
        help="Fraction of source data reserved for validation (of the full dataset).",
    )
    parser.add_argument(
        "--source-test-fraction",
        type=float,
        default=0.2,
        help="Fraction of source data reserved for testing.",
    )
    parser.add_argument(
        "--target-val-fraction",
        type=float,
        default=0.0,
        help="Optional fraction of target data reserved for validation.",
    )
    parser.add_argument(
        "--target-test-fraction",
        type=float,
        default=0.2,
        help="Fraction of target data reserved for testing.",
    )
    parser.add_argument(
        "--invert-source-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1 - y for source labels (useful when labels are viabilities).",
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
        help="How to run Weights & Biases logging. 'disabled' turns logging off.",
    )

    # --- Model configuration ---
    parser.add_argument("--drug", default="Vorinostat", help="Identifier used for logging and wandb run naming.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--mbS", type=int, default=32, help="Source mini-batch size.")
    parser.add_argument("--mbT", type=int, default=32, help="Target mini-batch size.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lam1", type=float, default=0.2)
    parser.add_argument("--h_dim", type=int, default=1024)
    parser.add_argument("--predictor_z_dim", type=int, default=256)
    parser.add_argument(
        "--balancing-strategy",
        default="weighted",
        choices=["none", "weighted", "smote"],
        help="Class-balancing strategy applied to the source loader.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument(
        "--plot-umap",
        action="store_true",
        help="If set, enable UMAP plotting callback inside the benchmark runner.",
    )
    parser.add_argument(
        "--tune-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MCC-based decision threshold tuning on the validation set.",
    )
    parser.add_argument(
        "--binarize-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Binarize source labels (>= threshold) before training.",
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
        "mbS": args.mbS,
        "mbT": args.mbT,
        "lr": args.lr,
        "dropout": args.dropout,
        "lam1": args.lam1,
        "h_dim": args.h_dim,
        "predictor_z_dim": args.predictor_z_dim,
        "balancing_strategy": args.balancing_strategy,
        "patience": args.patience,
        "plot_umap": bool(args.plot_umap),
        "tune_threshold": args.tune_threshold,
        "binarize_source": args.binarize_source,
    }

    target_file = Path(args.target_features).stem
    results = run_scad_benchmark(
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
