import os
os.environ["SCIPY_ARRAY_API"] = "1"  # requested env var

import sys
import numpy as np
import warnings
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
import optuna
from types import SimpleNamespace
from optuna.integration.wandb import WeightsAndBiasesCallback



class ScadArgs:
    def __init__(self, drug_name):
        self.drug = drug_name
        self.epochs = 100
        self.data_split = "test"
        self.log_normalized = True
        self.plot_umap = False
        self.mode = "SCAD"
        # Tuned params
        self.lr = 0.0005
        self.mbS = 8
        self.mbT = 8
        self.dropout = 0.5
        self.lam1 = 1
        self.balancing_strategy = "weighted"
        self.h_dim = 1024
        self.predictor_z_dim = 128 
        self.tune_threshold = True
        self.binarize_source = True
        self.patience = 10

class ScDealArgs:
    def __init__(self, drug_name):
        self.drug = drug_name
        self.result_path = "scDEAL_temp/"
        self.model_path = self.result_path
        self.bulk_model = self.result_path
        self.sc_model = self.result_path
        self.epochs_bulk = 50
        self.epochs = 100
        self.patience = 10
        self.batch_size = 32
        self.bulk_h_dims = "512,256"
        self.predictor_h_dims = "64,32"
        self.dimreduce = "DAE"
        self.freeze_pretrain = 0
        self.pretrain = "True"
        self.regularization_weight = 1.0
        self.data_split = "test"
        self.log_normalized = True
        self.mode = "scDEAL"
        # Tuned params
        self.lr_DA = 0.01
        self.dropout_sc = 0.3
        self.bottleneck = 128
        self.mmd_weight = 0.25
        self.balancing_strategy = "weighted"
        self.lr_bulk = 0.01 
        self.lr_sc = 0.01 
        self.dropout_bulk = 0.3  
        self.tune_threshold = True
        self.binarize_source = True


class Ssda4DrugArgs:
    def __init__(self, drug_name):
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
        # Tuned params
        self.lr = 0.001
        self.dropout = 0.3
        self.encoder_h_dims = "512,256"
        self.predictor_h_dims = "64,32"
        self.balancing_strategy = "weighted"
        self.tune_threshold = True
        self.binarize_source = True

class ScAtdArgs:
    def __init__(self, drug_name):
        self.drug = drug_name
        self.epochs = 100 # Max fine-tuning epochs
        self.epochs_classifier = 50 # Fixed classifier pre-training epochs
        self.patience = 10
        self.data_split = "test"
        self.log_normalized = True
        self.mode = "scATD"
        # Tuned params
        self.lr = 1e-3
        self.batch_size = 128
        self.weight_decay = 1e-3
        self.mmd_weight = 1
        self.balancing_strategy = "weighted"
        # VAE architecture - using defaults from main.py
        self.z_dim = 421
        self.hidden_dim_layer0 = 1664
        self.hidden_dim_layer_out_Z = 359
        self.pretrained_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "frameworks", "scATD", "pretrained_models", "checkpoint_fold1_epoch_30.pth"))
        self.tune_threshold = True
        self.binarize_source = True


# Add project root to system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from training_utils import (
    run_scad_benchmark,
    run_ssda4drug_benchmark,
    run_scdeal_benchmark,
    run_scatd_benchmark,
    run_catboost_benchmark,
    run_catboost_fewshot_baseline,
    set_seed,
)
from data_utils import (
    convert_to_ensembl,
    drop_all_nan_and_deduplicate,
    normalize_cpm_log1p_if_counts,
    intersect_genes,
    initialize_symbol_map,
)

# ----------------------- Script Configuration -----------------------
SEED = 42
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "processed"))
SYMBOL_ENSEMBL_MAP = os.path.join(DATA_DIR, "..", "reference", "symbol_ensembl_map.txt")
WANDB_PROJECT = "hyper_tuning_v5"

# This global variable will hold the data for the current sweep run.
CURRENT_DATA = {}
CURRENT_MODEL_CONFIG = {}

set_seed(SEED)


# ----------------------- Data Preparation -----------------------
def prepare_data(drug, target_tag, all_files, data_dir):
    """Loads, preprocesses, and splits the data for a given drug and target."""
    
    # Find source files
    X_source_files = [f for f in all_files if f.startswith("X_") and "bulk" in f]
    y_source_files = [f for f in all_files if f.startswith("y_") and "bulk" in f]
    if not X_source_files or not y_source_files:
        warnings.warn(f"[{drug}] Missing source files. Skipping.")
        return None

    # Load source data
    X_source_raw = pd.read_csv(os.path.join(data_dir, X_source_files[0]), index_col=0)
    y_source_raw = pd.read_csv(os.path.join(data_dir, y_source_files[0]), index_col=0)["viability"] 
    y_source = 1 - y_source_raw # 1 - Cmax viability = Sensitivity rate at Cmax
    y_source_bin = (y_source >= 0.5).astype(int) # binarized labels

    # Process source data
    X_source = convert_to_ensembl(X_source_raw.copy())
    X_source = drop_all_nan_and_deduplicate(X_source)
    X_source = normalize_cpm_log1p_if_counts(X_source, "X_source")

    # Find and load target files
    X_target_files = [f for f in all_files if f.startswith("X_") and drug in f and target_tag in f]
    y_target_files = [f for f in all_files if f.startswith("y_") and drug in f and target_tag in f]
    if not X_target_files or not y_target_files:
        warnings.warn(f'[{drug}] Missing target files for dataset {target_tag}. Skipping.')
        return None
    
    X_target_raw = pd.read_csv(os.path.join(data_dir, X_target_files[0]), index_col=0)
    y_target_raw = pd.read_csv(os.path.join(data_dir, y_target_files[0]), index_col=0)
    y_target_series = y_target_raw.iloc[:, 0]
    
    # Preprocessing
    X_target = convert_to_ensembl(X_target_raw.copy())
    X_target = drop_all_nan_and_deduplicate(X_target)
    X_target = normalize_cpm_log1p_if_counts(X_target, "X_target")


    # intersect genes with source
    (X_source, X_target), common_genes = intersect_genes(X_source, X_target)
 
    # Check if the gene names were processed correctly
    if not common_genes:
        print("No common genes found. Skipping this target pair.")
        return None


    # Data splitting
    # source: train 64%, val 16%, test 20%
    X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
        X_source, y_source, test_size=0.2, random_state=SEED, stratify=y_source_bin
    )
    # for stratification
    y_source_train_bin = y_source_train >= 0.5

    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_train, y_source_train, test_size=0.2, random_state=SEED, stratify=y_source_train_bin
    )

    # target: train 80%, test 20%
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target_series, test_size=0.2, random_state=SEED, stratify=y_target_series
    )

    # DEBUG statements
    # label distribution
    print(f"Source train label counts {np.bincount((y_source_train >= 0.5).astype(int))}, val counts {np.bincount((y_source_val >= 0.5).astype(int))}, test counts {np.bincount((y_source_test >= 0.5).astype(int))}")
    print(f"Target train label counts {y_target_train.value_counts().to_dict()}, test counts {y_target_test.value_counts().to_dict()}")

    # source and target data dimensions
    print(f"Source train data dimensions: {X_source_train.shape}")
    print(f"Target train data dimensions: {X_target_train.shape}")

    # Scaling
    source_scaler = StandardScaler()
    X_source_train = pd.DataFrame(source_scaler.fit_transform(X_source_train), index=X_source_train.index, columns=X_source_train.columns)
    X_source_val = pd.DataFrame(source_scaler.transform(X_source_val), index=X_source_val.index, columns=X_source_val.columns)
    X_source_test = pd.DataFrame(source_scaler.transform(X_source_test), index=X_source_test.index, columns=X_source_test.columns)
    
    expected_cols = list(source_scaler.feature_names_in_)
    X_target_train = pd.DataFrame(source_scaler.transform(X_target_train.reindex(columns=expected_cols)), index=X_target_train.index, columns=expected_cols)
    X_target_test = pd.DataFrame(source_scaler.transform(X_target_test.reindex(columns=expected_cols)), index=X_target_test.index, columns=expected_cols)

    # The benchmark functions expect it, Independent target data logic so we pass None.
    return {
        "x_train_source": X_source_train, "y_train_source": y_source_train,
        "x_val_source": X_source_val, "y_val_source": y_source_val,
        "x_test_source": X_source_test, "y_test_source": y_source_test,
        "x_train_target": X_target_train, "y_train_target": y_target_train,
        "x_test_target": X_target_test, "y_test_target": y_target_test,
        "X_target_independent": None, "y_target_independent": None, # independent target not needed in this hyperparameter tuning script
        "target_file": target_tag, "independent_target_file": "independent_placeholder",
    }



# ----------------------- Main Execution -----------------------
def main():
    """Main function to run the hyperparameter tuning sweeps using Optuna."""
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for various models.")
    parser.add_argument("--drugs", nargs='+', default=["Olaparib"], help="List of drugs to process.")
    parser.add_argument("--n_trials", type=int, default=101, help="Number of Optuna trials.")
    parser.add_argument("--model", required=True, help="Name of the model to tune.")
    cli_args = parser.parse_args()

    initialize_symbol_map(SYMBOL_ENSEMBL_MAP)

    model_name = cli_args.model

    target_file_names = {
        "Cisplatin":   ["GSE138267", "GSE117872_HN120", "GSE117872_HN137"], 
        "Paclitaxel": [ "GSE163836_FCIBC02", "GSE131984"], 
        "Ibrutinib": ["GSE152469_CLL"],
        "Olaparib": ["GSE228382"],
        "Vorinostat": ["SCC47"],
        "Etoposide": ["GSE149383_PC9"],
        "Erlotinib": ["GSE149383_PC9"],
        "Docetaxel": ["GSE140440_PC3", "GSE140440_DU145"],
        "Sorafenib":  ["GSE175716_HCC", "SCC47"], 
        "Gefitinib": [ "GSE162045_PC9", "GSE202234_H1975", "GSE202234_PC9", "JHU006", "GSE112274_PC9"], # 
        "Afatinib": ["GSE228154_LT","SCC47"] 
    }

    for drug in cli_args.drugs:
        print(f'\n{"="*25} Processing drug: {drug} {"="*25}')
        all_files = [f for f in os.listdir(DATA_DIR) if drug in f and f.endswith(".csv")]

        for target_tag in target_file_names.get(drug, []):
            print(f'\n--- Preparing data for target: {target_tag} ---')
            data = prepare_data(drug, target_tag, all_files, DATA_DIR)
            if data is None:
                print(f'Skipping {drug} with target {target_tag} due to data issues.')
                continue
            print(f'\n--- Starting Tuning for {model_name} on {drug}::{target_tag} ---')

            def objective(trial: optuna.Trial) -> float:
                monitor_split = "source_val"
                monitor_metric = "mcc"

                if model_name == "SCAD":
                    args = ScadArgs(drug)
                    args.lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
                    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
                    args.h_dim = trial.suggest_int("h_dim", 512, 1024, step=128)
                    args.predictor_z_dim = trial.suggest_int("predictor_z_dim", 128, 256, step=64)
                    args.mbS = trial.suggest_categorical("mbS", [8, 16, 32, 64])
                    args.mbT = trial.suggest_categorical("mbT", [8, 16, 32, 64])
                    args.lam1 = trial.suggest_float("lam1", 0.1, 10.0, log=True)
                    args.binarize_source = trial.suggest_categorical("binarize_source", [True, False])
                    benchmark_func = run_scad_benchmark
                
                elif model_name == "SSDA4Drug":
                    args = Ssda4DrugArgs(drug)
                    args.lr = trial.suggest_float("lr", 5e-5, 1e-2, log=True)
                    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
                    args.encoder_h_dims = trial.suggest_categorical("encoder_h_dims", ["256,128", "512,256", "1024,512"])
                    args.predictor_h_dims = trial.suggest_categorical("predictor_h_dims", ["64,32", "128,64", "256,128"])
                    args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
                    use_epsilon = trial.suggest_categorical("use_epsilon", [True, False])
                    if use_epsilon:
                        args.epsilon = trial.suggest_float("epsilon", 1e-4, 1e-2, log=True)
                    else:
                        args.epsilon = 0.0
                    args.binarize_source = trial.suggest_categorical("binarize_source", [True, False])
                    benchmark_func = run_ssda4drug_benchmark

                elif model_name == "scDEAL":
                    args = ScDealArgs(drug)
                    args.lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
                    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
                    args.bulk_h_dims = trial.suggest_categorical("bulk_h_dims", ["256,256", "512,256", "512,512"])
                    args.predictor_h_dims = trial.suggest_categorical("predictor_h_dims", ["128,64", "64,32", "32,16", "16,8"])
                    args.bottleneck = trial.suggest_int("bottleneck", 64, 512, step=64)
                    args.mmd_weight = trial.suggest_float("mmd_weight", 0.1, 1.0, log=False)
                    args.binarize_source = trial.suggest_categorical("binarize_source", [True, False])
                    benchmark_func = run_scdeal_benchmark

                elif model_name == "scATD":
                    args = ScAtdArgs(drug)
                    args.lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
                    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
                    args.mmd_weight = trial.suggest_float("mmd_weight", 0.1, 1.0, log=False)
                    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
                    args.binarize_source = trial.suggest_categorical("binarize_source", [True, False])
                    benchmark_func = run_scatd_benchmark


                elif model_name == "CatBoost_fs":
                    args = SimpleNamespace()

                    def benchmark_func(x_train_source, y_train_source, x_val_source, y_val_source, x_test_source, y_test_source, x_train_target, y_train_target, x_test_target, y_test_target, X_target_independent, y_target_independent, target_file, independent_target_file, trial=None, seed=SEED, **kwargs):
                        cur_seed = seed if trial is None else seed + trial.number
                        return run_catboost_fewshot_baseline(
                            x_val_source,
                            y_val_source,
                            x_test_source,
                            y_test_source,
                            x_train_target,
                            y_train_target,
                            x_test_target,
                            y_test_target,
                            seed=cur_seed,
                        )

                    monitor_split = "target_test"
                    monitor_metric = "auc"
                
                elif model_name.startswith("CatBoost"):
                    class CatBoostArgs:
                        def __init__(self, drug_name, model_name):
                            self.drug = drug_name
                            self.data_split = "test"
                            self.mode = model_name
                            self.tune_threshold = True 

                    args = CatBoostArgs(drug, model_name)
                    # The benchmark function will handle the trial suggestions.
                    benchmark_func = run_catboost_benchmark
                    
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                args_dict = vars(args)
                # For CatBoost, we also need to pass the model_type
                if model_name.startswith("CatBoost"):
                    args_dict["model_type"] = model_name

                wandb_config = dict(trial.params)
                if hasattr(args, "binarize_source"):
                    wandb_config.setdefault("binarize_source", args.binarize_source)
                if model_name == "RF":
                    rf_seed_value = 42
                    wandb_config.update(
                        {
                            "rf_n_estimators": 25,
                            "rf_max_depth": 3,
                            "rf_shots_per_class": 3,
                            "rf_seed": rf_seed_value,
                        }
                    )

                with wandb.init(
                    project=WANDB_PROJECT,
                    group=f"{drug}_{target_tag}_{model_name}",
                    job_type="optuna_trial",
                    config=wandb_config,
                    reinit=True,
                ) as run:
                    try:
                        if model_name.startswith("CatBoost"):
                            results = benchmark_func(**args_dict, **data, trial=trial, seed=SEED)
                        else:
                            results = benchmark_func(args_dict, **data, trial=trial, seed=SEED)
                    except ValueError as exc:
                        warning_msg = f"{model_name} failed: {exc}"
                        print(warning_msg)
                        run.log({"error": warning_msg})
                        return -1.0

                    to_log = {
                        "drug": drug,
                        "target": target_tag,
                        "model_name": model_name,
                        **{
                            f"{split}_{metric}": value
                            for split, metrics in results.items()
                            if isinstance(metrics, dict)
                            for metric, value in metrics.items()
                        },
                    }
                    run.log(to_log)

                    return results.get(monitor_split, {}).get(monitor_metric, -1.0)

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), study_name=f"{drug}_{target_tag}_{model_name}")
            # Initial starting point (baseline) parameters
            if model_name == "SCAD":
                initial_params = {
                    "lr": 0.0005,
                    "dropout": 0.5,
                    "h_dim": 1024,
                    "predictor_z_dim": 128,
                    "mbS": 8,
                    "mbT": 8,
                    "lam1": 1,
                    "binarize_source": True
                }
            elif model_name == "SSDA4Drug":
                initial_params = {
                    "lr": 0.001,
                    "dropout": 0.3,
                    "encoder_h_dims": "512,256",
                    "predictor_h_dims": "64,32",
                    "batch_size": 128,
                    "epsilon": 0.0,
                    "binarize_source": True
                }
            elif model_name == "scDEAL":
                initial_params = {
                    "lr": 0.01,
                    "dropout": 0.3,
                    "bulk_h_dims": "512,256",
                    "predictor_h_dims": "64,32",
                    "bottleneck": 128,
                    "mmd_weight": 0.25,
                    "binarize_source": True
                }
            elif model_name == "scATD":
                initial_params = {
                    "lr": 0.001,
                    "mmd_weight": 1,
                    "batch_size": 128,
                    "weight_decay": 0.001,
                    "binarize_source": True,
                }
            elif model_name.startswith("CatBoost"):
                # For CatBoost, we use the default parameter values as initial starting point
                initial_params = {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3, "border_count": 254}
            else:
                initial_params = {}

            if initial_params:
                study.enqueue_trial(initial_params)

            study.optimize(objective, n_trials=cli_args.n_trials)

            print(f"--- Finished study for {model_name} on {drug}::{target_tag} ---")

if __name__ == "__main__":
    main()
