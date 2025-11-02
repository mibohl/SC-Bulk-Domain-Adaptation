# Standard library imports
import os
import random
import sys
import warnings
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional
import json

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from catboost import CatBoostClassifier
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import TYPE_CHECKING
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path for framework imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils import (
    CombinedDataLoader,
    align_dataframe_to_gene_vocab,
    convert_to_symbols,
    create_dataloader,
    create_shot_dataloaders,
    load_scatd_gene_vocab,
)
from frameworks.scDeal.scDEAL_utils import calculateKNNgraphDistanceMatrix, generateLouvainCluster


def _detect_catboost_task_type() -> str:
    """
    Return 'GPU' when CatBoost sees at least one GPU, otherwise 'CPU'.
    Falls back to CPU if detection fails.
    """
    try:
        from catboost.utils import get_gpu_device_count  # type: ignore
    except ImportError:  # pragma: no cover - optional helper
        get_gpu_device_count = None

    if get_gpu_device_count is not None:
        try:
            if get_gpu_device_count() > 0:
                return "GPU"
        except Exception:  # pragma: no cover - safety fallback
            warnings.warn("CatBoost GPU detection failed; defaulting to CPU.")
    return "CPU"


_CATBOOST_TASK_TYPE = _detect_catboost_task_type()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class DelayedEarlyStopping(EarlyStopping):
    """Early stopping that ignores validation metrics for an initial epoch window."""

    def __init__(self, delay_epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_epochs = delay_epochs

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch < self.delay_epochs:
            return
        super().on_validation_end(trainer, pl_module)


def calculate_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculates a dictionary of metrics."""

    # Ensure y_true is a 1D array and binarized
    y_true_binary = (np.asarray(y_true).ravel() >= 0.5).astype(int)

    # Ensure y_pred_proba is a 1D array
    y_pred_proba = np.asarray(y_pred_proba).ravel()

    # Check for NaN values in predictions
    if np.any(np.isnan(y_pred_proba)):
        print(f"Warning: Found {np.sum(np.isnan(y_pred_proba))} NaN values in predictions. Replacing with 0.5.")
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)

    # Check for infinite values
    if np.any(np.isinf(y_pred_proba)):
        print(f"Warning: Found {np.sum(np.isinf(y_pred_proba))} infinite values in predictions. Replacing with 0.5.")
        y_pred_proba = np.nan_to_num(y_pred_proba, posinf=1.0, neginf=0.0)

    y_pred_binary = (y_pred_proba > threshold).astype(int)

    if len(np.unique(y_true_binary)) < 2:
        return {
            "auc": -1,
            "auprc": -1,
            "mcc": -1,
            "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
            "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            "recall_sensitivity": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            "f1_score": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            "specificity": -1,
        }

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calculate AUC with error handling
    try:
        auc_score = float(roc_auc_score(y_true_binary, y_pred_proba))
    except ValueError as e:
        print(f"Warning: Error calculating AUC: {e}. Setting AUC to -1.")
        auc_score = -1.0

    # Calculate AUPRC with error handling
    try:
        auprc_score = float(average_precision_score(y_true_binary, y_pred_proba))
    except ValueError as e:
        print(f"Warning: Error calculating AUPRC: {e}. Setting AUPRC to -1.")
        auprc_score = -1.0

    return {
        "auc": auc_score,
        "auprc": auprc_score,
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true_binary, y_pred_binary)) if len(np.unique(y_true_binary)) > 1 else 0.0,
    }


def roc_auc_score_trainval(y_true, y_pred_proba):
    """Safe roc_auc_score for training/validation where a batch might have only one class."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        return 0.5


def _tune_threshold_DL(get_preds_func, source_val_loader, target_val_loader):
    """Tunes the decision threshold based on MCC score on validation sets."""
    s_val_labels, s_val_probs = get_preds_func(source_val_loader)
    tgt_val_labels, tgt_val_probs = get_preds_func(target_val_loader)
    best_t, best_mcc = 0.5, -1.0

    for t in [i / 20.0 for i in range(1, 20)]:
        s_preds = (s_val_probs >= t).astype(int)
        tgt_preds = (tgt_val_probs >= t).astype(int)

        s_mcc = 0
        if len(np.unique(s_val_labels)) > 1 and len(np.unique(s_preds)) > 1:
            s_mcc = matthews_corrcoef(s_val_labels, s_preds)

        tgt_mcc = 0
        if len(np.unique(tgt_val_labels)) > 1 and len(np.unique(tgt_preds)) > 1:
            tgt_mcc = matthews_corrcoef(tgt_val_labels, tgt_preds)

        mcc = (s_mcc + tgt_mcc) / 2.0
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t

    if wandb.run is not None:
        wandb.log({"decision_threshold": best_t, "tuned_threshold_mcc": best_mcc})

    return best_t


def _tune_threshold_DL_source_only(get_preds_func, val_loader):
    """Tune decision threshold on validation set using MCC.
    Logs 'decision_threshold' and 'decision_threshold_mcc' to wandb.
    """
    val_labels, val_probs = get_preds_func(val_loader)

    # Binarize continuous labels for MCC calculation
    y_bin = (np.asarray(val_labels).ravel() >= 0.5).astype(int)
    probs = np.asarray(val_probs).ravel()

    best_t, best_mcc = 0.5, -1.0
    for t in [i / 20.0 for i in range(1, 20)]:
        preds = (probs >= t).astype(int)
        if len(np.unique(preds)) < 2:
            mcc = -1.0
        else:
            mcc = matthews_corrcoef(y_bin, preds)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t

    if wandb.run is not None:
        wandb.log({"decision_threshold": best_t, "decision_threshold_mcc": best_mcc})

    return best_t


def _tune_threshold_GB(model, X_val, y_val):
    """Tune decision threshold on validation set using MCC over 0.1..0.9.
    Returns (best_t, best_mcc) and logs to wandb if active.
    """
    probs = model.predict_proba(X_val)[:, 1]
    y_bin = (np.asarray(y_val).ravel() >= 0.5).astype(int)
    best_t, best_mcc = 0.5, -1.0
    for t in [i / 20.0 for i in range(1, 20)]:
        preds = (probs >= t).astype(int)
        # avoid degenerate constant predictions
        if len(np.unique(preds)) < 2:
            mcc = -1.0
        else:
            mcc = matthews_corrcoef(y_bin, preds)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    if wandb.run is not None:
        wandb.log({"decision_threshold": best_t, "tuned_threshold_mcc": best_mcc})
    return best_t, best_mcc


def plot_training_history(history, save_path):
    """
    Plots the training and validation loss and metrics over epochs.
    Args:
        history (list): A list of dictionaries, where each dictionary contains
                        metrics for a specific epoch and phase ('train' or 'val').
        save_path (str): The file path to save the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not history:
        print("No history to plot.")
        return

    # Dynamically find all available metrics to plot
    metrics_to_plot = {}
    for item in history:
        for key, value in item.items():
            if isinstance(value, (int, float)) and key not in ["Epoch", "phase"]:
                if key not in metrics_to_plot:
                    metrics_to_plot[key] = {"title": key.replace("_", " ").title(), "train": [], "val": []}

    epochs = sorted(list(set([d["Epoch"] for d in history])))

    for epoch_num in epochs:
        train_data = next((item for item in history if item["Epoch"] == epoch_num and item["phase"] == "train"), {})
        val_data = next((item for item in history if item["Epoch"] == epoch_num and item["phase"] == "val"), {})

        for key in metrics_to_plot:
            metrics_to_plot[key]["train"].append(train_data.get(key))
            metrics_to_plot[key]["val"].append(val_data.get(key))

    plots_available = {
        key: data
        for key, data in metrics_to_plot.items()
        if any(v is not None for v in data["train"]) and any(v is not None for v in data["val"])
    }

    if not plots_available:
        print("No complete data to plot (missing train/val pairs for metrics).")
        return

    num_plots = len(plots_available)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    for ax, (key, data) in zip(axes, plots_available.items()):
        ax.plot(epochs, data["train"], label=f'Train {data["title"]}')
        ax.plot(epochs, data["val"], label=f'Validation {data["title"]}')
        ax.set_title(f'{data["title"]} Over Epochs')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(data["title"])
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _log_benchmark_results(results, history=None, threshold=None):
    """Logs benchmark results to wandb."""
    if wandb.run is None:
        return

    # Log history
    if history:
        if isinstance(history, dict):
            for key in history:
                for i, value in enumerate(history[key]):
                    wandb.log({f"history/{key}": value, "epoch": i})
        elif isinstance(history, list):
            for entry in history:
                if "phase" in entry and "Epoch" in entry:
                    phase = entry["phase"]
                    epoch = entry["Epoch"]
                    for metric_name, value in entry.items():
                        if metric_name not in ["Epoch", "phase"]:
                            wandb.log({f"history/{phase}_{metric_name}": value, "epoch": epoch})

    # Log final metrics
    flat_results = {}
    if threshold is not None:
        flat_results["selected_threshold"] = threshold

    for test_type, metrics in results.items():
        for metric_name, value in metrics.items():
            flat_results[f"{test_type.replace('_test','').replace('_val','_validation')}_{metric_name}"] = value
    
    wandb.log(flat_results)


def convert_numpy_types(obj):
    """
    Recursively converts NumPy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def initialize_experiment_log(result_path: str, mode: str, label: str):
    """Create a fresh experiment log structure and return its path."""
    os.makedirs(result_path, exist_ok=True)
    log_path = os.path.join(result_path, f"{mode.lower()}_{label}.json")
    log_dict = {
        "mode": mode,
        "label": label,
        "folds": [],
    }
    return log_path, log_dict


def log_fold_results(experiment_log: dict, fold_index: int, source_metrics: dict, target_metrics: dict):
    """Append source/target metrics for a fold to the running experiment log."""
    experiment_log.setdefault("folds", []).append(
        {
            "fold": fold_index,
            "source": convert_numpy_types(source_metrics),
            "target": convert_numpy_types(target_metrics),
        }
    )


def finalize_and_save_log(log_path: str, experiment_log: dict, args) -> None:
    """Persist the experiment log alongside argument metadata."""
    payload = dict(experiment_log)
    if hasattr(args, "__dict__"):
        payload["args"] = convert_numpy_types(vars(args))
    else:
        payload["args"] = convert_numpy_types(args)

    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(convert_numpy_types(payload), fp, indent=2)


def evaluate_sklearn_model(model, X_source_test, y_source_test, X_target_test, y_target_test, X_target_independent, y_target_independent, threshold=0.5):
    """
    Evaluates the Random Forest model on source, target, and independent target datasets.
    Returns a dictionary of evaluation metrics.
    """
    # Predict probabilities
    source_prob = model.predict_proba(X_source_test)[:, 1]
    target_prob = model.predict_proba(X_target_test)[:, 1]
    independent_target_prob = model.predict_proba(X_target_independent)[:, 1]

    # Get true labels
    y_source_vec = _y_to_vector(y_source_test)
    y_target_vec = _y_to_vector(y_target_test)
    y_indep_vec = _y_to_vector(y_target_independent)

    # Calculate metrics
    source_metrics = calculate_all_metrics(y_source_vec, source_prob, threshold=threshold)
    target_metrics = calculate_all_metrics(y_target_vec, target_prob, threshold=threshold)
    independent_target_metrics = calculate_all_metrics(y_indep_vec, independent_target_prob, threshold=threshold)

    results = {
        "source_test": source_metrics,
        "target_test": target_metrics,
        "independent_target_test": independent_target_metrics,
    }
    return results


def train_AE_model(
    net,
    data_loaders={},
    optimizer=None,
    loss_function=None,
    n_epochs=100,
    scheduler=None,
    load=False,
    save_path="model.pkl",
):

    if load:
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
            return net, 0

    best_model_wts = deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            n_iters = len(data_loaders[phase])

            for batchidx, (x, _) in enumerate(data_loaders[phase]):
                x.requires_grad_(True)
                output = net(x)
                loss = loss_function(output, x)
                optimizer.zero_grad()
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / n_iters
            if phase == "train":
                scheduler.step(epoch_loss)

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = deepcopy(net.state_dict())

    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)

    return net, None


def train_DAE_model(
    net,
    task,
    data_loaders={},
    optimizer=None,
    loss_function=None,
    n_epochs=100,
    scheduler=None,
    load=False,
    save_path="model.pkl",
):

    if load:
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
            return net, 0
        else:
            print("Failed to load existing file, proceed to the trainning process.")

    best_model_wts = deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            n_iters = len(data_loaders[phase])

            for batchidx, (x, _) in enumerate(data_loaders[phase]):
                z = x.clone()
                y = np.random.binomial(1, 0.2, (z.shape[0], z.shape[1]))
                z[np.array(y, dtype=bool),] = 0
                x.requires_grad_(True)
                output = net(z)
                loss = loss_function(output, x)
                optimizer.zero_grad()
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / n_iters
            if phase == "train":
                scheduler.step(epoch_loss)

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = deepcopy(net.state_dict())

    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)

    return net, None


def train_predictor_model(
    net,
    data_loaders,
    optimizer,
    loss_function,
    n_epochs,
    scheduler,
    load=False,
    save_path="/cluster/scratch/mibohl/master_thesis/code/replicate_and_benchmark/RF_benchmark/scDEAL_temp/model.pkl",
    task="",
):

    if load:
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
            return net, 0
        else:
            print("Failed to load existing file, proceed to the trainning process.")

    best_model_wts = deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            n_iters = len(data_loaders[phase])

            for batchidx, (x, y) in enumerate(data_loaders[phase]):
                x.requires_grad_(True)
                output = net(x)
                loss = loss_function(output, y.view(-1, 1).float())
                optimizer.zero_grad()
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / n_iters
            if phase == "train":
                scheduler.step(epoch_loss)

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = deepcopy(net.state_dict())

    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)

    return net, None


def train_DaNN_model(
    net,
    source_loader,
    target_loader,
    optimizer,
    loss_function,
    n_epochs,
    scheduler,
    dist_loss,
    alpha=0.25,
    beta=1,
    GAMMA=1000,
    epoch_tail=0.90,
    load=False,
    save_path="/cluster/scratch/mibohl/master_thesis/code/replicate_and_benchmark/RF_benchmark/scDEAL_temp",
    best_model_cache="memory",
    top_models=5,
    k=10,
    device="cuda",
):

    if load:
        if os.path.exists(save_path):
            try:
                net.load_state_dict(torch.load(save_path))
                return net, 0, 0, 0
            except Exception as e:
                print(f"Failed to load existing file: {e}, proceed to the trainning process.")

        else:
            print("Failed to load existing file, proceed to the trainning process.")

    best_model_wts = deepcopy(net.state_dict())
    best_loss = np.inf
    history = []

    for epoch in range(n_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_mmd = 0.0
            running_sc = 0.0
            running_auc = []
            running_aupr = []
            running_loss_c = []
            source_iter = iter(source_loader[phase])
            target_iter = iter(target_loader[phase])
            n_iters = max(len(source_loader[phase]), len(target_loader[phase]))

            for i in range(n_iters):
                try:
                    x_src, y_src = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader[phase])
                    x_src, y_src = next(source_iter)
                try:
                    x_tar, y_tar = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader[phase])
                    x_tar, y_tar = next(target_iter)

                x_tar.requires_grad_(True)
                x_src.requires_grad_(True)

                min_size = min(x_src.shape[0], x_tar.shape[0])

                if x_src.shape[0] != x_tar.shape[0]:
                    x_src = x_src[:min_size,]
                    y_src = y_src[:min_size,]
                    x_tar = x_tar[:min_size,]
                    y_tar = y_tar[:min_size,]

                y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar)
                encoderrep = net.target_model.encoder(x_tar)

                if encoderrep.shape[0] < k:
                    continue
                else:
                    edgeList = calculateKNNgraphDistanceMatrix(
                        encoderrep.cpu().detach().numpy(),
                        distanceType="euclidean",
                        k=10,
                    )
                    listResult, size = generateLouvainCluster(edgeList)
                    loss_s = 0
                    for i in range(size):
                        s = cosine_similarity(x_tar[np.asarray(listResult) == i, :].cpu().detach().numpy())
                        s = 1 - s
                        loss_s += np.sum(np.triu(s, 1)) / ((s.shape[0] * s.shape[0]) * 2 - s.shape[0])
                    if device == "cuda":
                        loss_s = torch.tensor(loss_s).cuda()
                    else:
                        loss_s = torch.tensor(loss_s).cpu()
                    loss_s.requires_grad_(True)
                    loss_c = loss_function(y_pre, y_src.view(-1, 1).float())
                    loss_mmd = dist_loss(x_src_mmd, x_tar_mmd)
                    loss = loss_c + alpha * loss_mmd + beta * loss_s
                    optimizer.zero_grad()
                    if phase == "train":
                        loss.backward(retain_graph=True)
                        optimizer.step()

                    running_loss += loss.item()
                    running_mmd += loss_mmd.item()
                    running_sc += loss_s.item()
                    running_loss_c.append(loss_c.item())
                    y_true = y_src.cpu().detach().numpy()
                    y_pred = y_pre.cpu().detach().numpy()
                    try:
                        auc = roc_auc_score(y_true, y_pred)
                    except Exception:
                        auc = float("nan")
                    try:
                        aupr = average_precision_score(y_true, y_pred)
                    except Exception:
                        aupr = float("nan")
                    running_auc.append(auc)
                    running_aupr.append(aupr)

            epoch_loss = running_loss / n_iters
            epoch_mmd = running_mmd / n_iters
            epoch_sc = running_sc / n_iters
            epoch_auc = np.nanmean(running_auc) if running_auc else float("nan")
            epoch_aupr = np.nanmean(running_aupr) if running_aupr else float("nan")
            epoch_loss_c = np.nanmean(running_loss_c) if running_loss_c else float("nan")
            if phase == "train":
                scheduler.step(epoch_loss)
            history.append(
                {
                    "Epoch": epoch + 1,
                    "phase": phase,
                    "loss": epoch_loss,
                    "loss_c": epoch_loss_c,
                    "loss_mmd": epoch_mmd,
                    "loss_sc": epoch_sc,
                    "auc": epoch_auc,
                    "aupr": epoch_aupr,
                }
            )
            if (phase == "val") and (epoch_loss < best_loss) and (epoch > (n_epochs * (1 - epoch_tail))):
                best_loss = epoch_loss
                if best_model_cache == "memory":
                    best_model_wts = deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path + "_bestcahce.pkl")

    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)
    else:
        net.load_state_dict((torch.load(save_path + "_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, history, None, None, None


def run_scdeal_benchmark(
    args,
    x_train_source,
    y_train_source,
    x_val_source,
    y_val_source,
    x_test_source,
    y_test_source,
    x_train_target,
    y_train_target,
    x_test_target,
    y_test_target,
    X_target_independent,
    y_target_independent,
    target_file,
    independent_target_file,
    seed=42,
    trial=None,
    **kwargs,
):
    """
    Runs the scDEAL benchmark with the given arguments and data using PyTorch Lightning.
    """
    from frameworks.scDeal import modules as scdeal_modules
    from frameworks.scDeal.lightning_module import (
        AELightningModule,
        DaNNLightningModule,
        PredictorLightningModule,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)

    pl.seed_everything(seed, workers=True)

    if args.get('binarize_source', True):
        y_train_source = (y_train_source >= 0.5).astype(int)
        y_val_source = (y_val_source >= 0.5).astype(int)
        y_test_source = (y_test_source >= 0.5).astype(int)

    # --- Data Loaders ---
    source_train_loader = create_dataloader(
        x_train_source,
        y_train_source,
        args['batch_size'],
        shuffle=True,
        drop_last=False,
        balancing_strategy=args.get('balancing_strategy'),
        seed=seed,
    )
    source_val_loader = create_dataloader(x_val_source, y_val_source, args['batch_size'], shuffle=False)
    target_train_loader = create_dataloader(
        x_train_target,
        y_train_target,
        args['batch_size'],
        shuffle=True,
        drop_last=False,
        balancing_strategy=args.get('balancing_strategy'),
        seed=seed,
    )
    x_val_target = kwargs.get("x_val_target")
    y_val_target = kwargs.get("y_val_target")
    if x_val_target is None or y_val_target is None:
        warnings.warn(
            "Target validation split not provided; reusing target training data for validation.",
            RuntimeWarning,
        )
        target_val_loader = create_dataloader(
            x_train_target,
            y_train_target,
            args['batch_size'],
            shuffle=False,
            drop_last=False,
        )
    else:
        target_val_loader = create_dataloader(x_val_target, y_val_target, args['batch_size'], shuffle=False)
    
    dataloader_source = {"train": source_train_loader, "val": source_val_loader}
    dataloader_target = {"train": target_train_loader, "val": target_val_loader}

    input_dim = x_train_source.shape[1]
    bottleneck_dim = args['bottleneck']
    bulk_encoder_hdims = list(map(int, args['bulk_h_dims'].split(",")))
    sc_encoder_hdims = [512, 256]  # sc encoder stays at default width
    dropout_rate = args['dropout']
    predictor_hdims = list(map(int, args['predictor_h_dims'].split(",")))

    # --- Logger ---
    run_name = f"scDEAL_{args['drug']}_{target_file}"
    if trial:
        run_name += f"_trial_{trial.number}"
    
    wandb_logger = WandbLogger(
        name=run_name, 
        config=args
    )

    lr_bulk = args.get('lr_bulk', args['lr'])
    lr_sc = args.get('lr_sc', args['lr'])
    epochs_sc = args.get('epochs_sc', args['epochs'])

    # === Stage 1: Pretrain Encoder on Bulk Data ===
    print("--- Stage 1: Pre-training bulk encoder ---")
    bulk_encoder_model = scdeal_modules.AEBase(input_dim, bottleneck_dim, bulk_encoder_hdims, dropout_rate)
    ae_module = AELightningModule(bulk_encoder_model, lr_bulk)
    trainer_ae = pl.Trainer(max_epochs=args['epochs_bulk'], accelerator=device, devices=1, enable_checkpointing=False, logger=False, enable_progress_bar=False)
    trainer_ae.fit(ae_module, dataloader_source['train'], dataloader_source['val'])
    
    # === Stage 2: Train Bulk Predictor ===
    print("--- Stage 2: Training bulk predictor ---")
    predictor_model = scdeal_modules.PretrainedPredictor(
        input_dim=input_dim, latent_dim=bottleneck_dim, h_dims=bulk_encoder_hdims,
        hidden_dims_predictor=predictor_hdims, output_dim=2,
        pretrained_weights=None, freezed=bool(args['freeze_pretrain']),
        drop_out=dropout_rate, drop_out_predictor=dropout_rate
    )
    predictor_model.load_state_dict(ae_module.model.state_dict(), strict=False)
    
    predictor_module = PredictorLightningModule(predictor_model, lr_bulk)
    trainer_pred = pl.Trainer(max_epochs=args['epochs_bulk'], accelerator=device, devices=1, enable_checkpointing=False, logger=False, enable_progress_bar=False)
    trainer_pred.fit(predictor_module, dataloader_source['train'], dataloader_source['val'])

    # === Stage 3: Pretrain Encoder on Single-Cell Data ===
    print("--- Stage 3: Pre-training single-cell encoder ---")
    sc_encoder_model = scdeal_modules.AEBase(input_dim, bottleneck_dim, sc_encoder_hdims, dropout_rate)
    sc_ae_module = AELightningModule(sc_encoder_model, lr_sc, noise_prob=0.2)
    trainer_sc_ae = pl.Trainer(
        max_epochs=epochs_sc,
        accelerator=device,
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer_sc_ae.fit(sc_ae_module, dataloader_target['train'], dataloader_target['val'])

    # === Stage 4: Train DaNN ===
    print("--- Stage 4: Training DaNN model ---")
    dann_model = scdeal_modules.DaNN(source_model=predictor_module.model, target_model=sc_ae_module.model).to(device)
    dann_module = DaNNLightningModule(
        model=dann_model,
        learning_rate=args['lr'],
        alpha=args['mmd_weight'],
        beta=args.get('regularization_weight', 1.0),
    )
    
    train_loader_dann = CombinedDataLoader(
        dataloader_source['train'],
        dataloader_target['train'],
        length_mode="max",
        resample_target=True,
    )
    val_loader_dann = dataloader_source['val']

    checkpoint_cb = ModelCheckpoint(monitor="val_loss_c", mode="min", save_top_k=1, filename="dann-best")
    earlystop_cb  = EarlyStopping(monitor="val_loss_c", patience=args.get('patience', 10))

    trainer_dann = pl.Trainer(
        max_epochs=args['epochs'],
        accelerator=device,
        devices=1,
        callbacks=[earlystop_cb, checkpoint_cb],
        logger=wandb_logger,
        enable_progress_bar=False,
    )

    trainer_dann.fit(dann_module, train_loader_dann, val_loader_dann)

    # Restore best weights
    best_ckpt_path = checkpoint_cb.best_model_path
    if best_ckpt_path and os.path.isfile(best_ckpt_path):
        final_model = DaNNLightningModule.load_from_checkpoint(
            best_ckpt_path,
            model=dann_model,
            learning_rate=args['lr'],
            alpha=args['mmd_weight'],
            beta=args['regularization_weight'],
        ).model.source_model.to(device).eval()
    else:
        warnings.warn("Could not find best model checkpoint. Using last model state for evaluation.")
        final_model = dann_module.model.source_model.to(device).eval()


    def get_preds(loader):
        labels = []
        probs = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = final_model(x)
                prob = logits[:, 1]
                probs.append(prob.cpu())
                labels.append(y.view(-1).cpu())
        
        labels = torch.cat(labels).view(-1).cpu().numpy()
        probs = torch.cat(probs).view(-1).cpu().numpy()
        return labels, probs

    # Tune threshold on source validation set
    tuned_threshold = _tune_threshold_DL_source_only(get_preds, val_loader_dann)

    # Evaluate with tuned threshold
    source_val_labels, source_val_probs = get_preds(source_val_loader)
    source_val_metrics = calculate_all_metrics(source_val_labels, source_val_probs, threshold=tuned_threshold)

    source_test_loader = create_dataloader(x_test_source, y_test_source, args['batch_size'], shuffle=False)
    source_test_labels, source_test_probs = get_preds(source_test_loader)
    source_test_metrics = calculate_all_metrics(source_test_labels, source_test_probs, threshold=tuned_threshold)

    target_test_loader = create_dataloader(x_test_target, y_test_target, args['batch_size'], shuffle=False)
    target_test_labels, target_test_probs = get_preds(target_test_loader)
    target_metrics = calculate_all_metrics(target_test_labels, target_test_probs, threshold=tuned_threshold)

    independent_target_metrics = {}
    if X_target_independent is not None and y_target_independent is not None:
        independent_target_loader = create_dataloader(X_target_independent, y_target_independent, args['batch_size'], shuffle=False)
        ind_target_labels, ind_target_probs = get_preds(independent_target_loader)
        independent_target_metrics = calculate_all_metrics(ind_target_labels, ind_target_probs, threshold=tuned_threshold)

    results = {
        "source_val": source_val_metrics,
        "source_test": source_test_metrics, 
        "target_test": target_metrics, 
        "independent_target_test": independent_target_metrics,
        "selected_threshold": tuned_threshold
    }
    return results

import os
import warnings

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def run_scatd_benchmark(
    args,
    x_train_source,
    y_train_source,
    x_val_source,
    y_val_source,
    x_test_source,
    y_test_source,
    x_train_target,
    y_train_target,
    x_test_target,
    y_test_target,
    X_target_independent,
    y_target_independent,
    target_file,
    independent_target_file,
    seed=42,
    trial=None,
):
    """Run the scATD benchmark with the given arguments and data using PyTorch Lightning."""
    # Local imports prevent circular dependency with scATDLightningModule
    from frameworks.scATD.lightning_module import scATDLightningModule
    from frameworks.scATD.lightning_datamodule import CombinedDataLoader as scATDCombinedLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pl.seed_everything(seed, workers=True)

    binarize_threshold = float(args.get("binarize_threshold", 0.5))

    if args.get("binarize_source", True):
        y_train_source = (y_train_source >= binarize_threshold).astype(int)
        y_val_source = (y_val_source >= binarize_threshold).astype(int)
        y_test_source = (y_test_source >= binarize_threshold).astype(int)

    print("\nConverting Ensembl IDs to Gene Symbols for scATD...")
    x_train_source = convert_to_symbols(x_train_source)
    x_val_source = convert_to_symbols(x_val_source)
    x_test_source = convert_to_symbols(x_test_source)
    x_train_target = convert_to_symbols(x_train_target)
    x_test_target = convert_to_symbols(x_test_target)
    if X_target_independent is not None:
        X_target_independent = convert_to_symbols(X_target_independent)

    print("\nAligning data to scATD gene vocabulary (19,264 genes)...")
    gene_vocab = load_scatd_gene_vocab()

    x_train_source = align_dataframe_to_gene_vocab(x_train_source, gene_vocab, "x_train_source")
    x_val_source = align_dataframe_to_gene_vocab(x_val_source, gene_vocab, "x_val_source")
    x_test_source = align_dataframe_to_gene_vocab(x_test_source, gene_vocab, "x_test_source")
    x_train_target = align_dataframe_to_gene_vocab(x_train_target, gene_vocab, "x_train_target")
    x_test_target = align_dataframe_to_gene_vocab(x_test_target, gene_vocab, "x_test_target")

    if X_target_independent is not None:
        X_target_independent = align_dataframe_to_gene_vocab(
            X_target_independent, gene_vocab, "X_target_independent"
        )

    # === Create DataLoaders ===
    source_train_loader = create_dataloader(
        x_train_source,
        y_train_source,
        args["batch_size"],
        shuffle=True,
        drop_last=False,
        balancing_strategy=args.get("balancing_strategy"),
        seed=seed,
    )
    target_train_loader = create_dataloader(
        x_train_target,
        y_train_target,
        args["batch_size"],
        shuffle=True,
        drop_last=False,
        seed=seed,
    )

    train_loader = scATDCombinedLoader(source_train_loader, target_train_loader)

    source_val_loader = create_dataloader(x_val_source, y_val_source, args["batch_size"], shuffle=False)
    source_test_loader = create_dataloader(x_test_source, y_test_source, args["batch_size"], shuffle=False)
    target_test_loader = create_dataloader(x_test_target, y_test_target, args["batch_size"], shuffle=False)

    # === Initialize Model ===
    input_size = len(gene_vocab)

    dist_vae_config = {
        "input_dim": input_size,
        "z_dim": args.get("z_dim", 421),
        "hidden_dim_layer0": args.get("hidden_dim_layer0", 1664),
        "hidden_dim_layer_out_Z": args.get("hidden_dim_layer_out_Z", 359),
        "Encoder_layer_dims": [
            input_size,
            args.get("hidden_dim_layer0", 1664),
            args.get("hidden_dim_layer_out_Z", 359),
        ],
        "Decoder_layer_dims": [args.get("z_dim", 421), args.get("hidden_dim_layer0", 1664)],
    }

    model = scATDLightningModule(
        lr=args["lr"],
        mmd_weight=args["mmd_weight"],
        epochs=args["epochs"],  # For scheduler T_max
        epochs_classifier=args["epochs_classifier"],
        pretrained_model_path=args.get("pretrained_model_path", None),
        weight_decay=args.get("weight_decay", 1e-3),
        class_weights=None,  # Not using class weights here, using sampler
        target_pool=x_train_target.values,
        **dist_vae_config,
    )

    # --- Logger & Callbacks ---
    run_name = f"scATD_{args['drug']}_{target_file}"
    if trial:
        run_name += f"_trial_{trial.number}"

    wandb_logger = WandbLogger(name=run_name, config=args)
    if wandb.run is not None:
        wandb.log({"binarize_threshold": binarize_threshold, "binarize_treshold": binarize_threshold})

    early_stop_callback = DelayedEarlyStopping(
        delay_epochs=args.get("epochs_classifier", 0),
        monitor="val_loss",
        patience=args.get("patience", 10),
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
    )
    callbacks = [early_stop_callback, checkpoint_callback]

    # === Train Model (Single Trainer) ===
    total_epochs = args["epochs_classifier"] + args["epochs"]
    trainer = pl.Trainer(
        max_epochs=total_epochs,
        accelerator=device,
        devices=1 if device == "cuda" else "auto",
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=True,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=source_val_loader)

    # === Evaluate Model ===
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        warnings.warn("Could not find best model checkpoint. Using last model state for evaluation.")
        best_model = model
    else:
        best_model = scATDLightningModule.load_from_checkpoint(best_model_path)

    best_model.to(device)
    best_model.eval()

    def get_preds(loader):
        labels = []
        probs = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                pred = best_model.model(x)
                softmax_preds = torch.softmax(pred, dim=1)[:, 1]
                probs.append(softmax_preds.cpu())
                labels.append(y)

        labels = torch.cat(labels).numpy().ravel()
        probs = torch.cat(probs).cpu().numpy().ravel()
        return labels, probs

    tuned_threshold = _tune_threshold_DL_source_only(get_preds, source_val_loader)

    source_val_labels, source_val_probs = get_preds(source_val_loader)
    source_val_metrics = calculate_all_metrics(
        source_val_labels, source_val_probs, threshold=tuned_threshold
    )

    source_test_labels, source_test_probs = get_preds(source_test_loader)
    source_test_metrics = calculate_all_metrics(
        source_test_labels, source_test_probs, threshold=tuned_threshold
    )

    target_test_labels, target_test_probs = get_preds(target_test_loader)
    target_test_metrics = calculate_all_metrics(
        target_test_labels, target_test_probs, threshold=tuned_threshold
    )

    independent_target_metrics = {}
    if X_target_independent is not None and y_target_independent is not None:
        independent_target_loader = create_dataloader(
            X_target_independent, y_target_independent, args["batch_size"], shuffle=False
        )
        ind_target_labels, ind_target_probs = get_preds(independent_target_loader)
        independent_target_metrics = calculate_all_metrics(
            ind_target_labels, ind_target_probs, threshold=tuned_threshold
        )

    results = {
        "source_val": source_val_metrics,
        "source_test": source_test_metrics,
        "target_test": target_test_metrics,
        "independent_target_test": independent_target_metrics,
        "selected_threshold": tuned_threshold,
    }

    return results

def _y_to_vector(y):
    if isinstance(y, pd.DataFrame):
        if "response" in y.columns:
            return y["response"].values.ravel()
        else:
            return y.iloc[:, 0].values.ravel()
    elif isinstance(y, pd.Series):
        return y.values.ravel()
    else:
        arr = np.asarray(y)
        return arr.ravel()
    

def run_catboost_benchmark(
    model_type,
    x_train_source,
    y_train_source,
    x_val_source,
    y_val_source,
    x_test_source,
    y_test_source,
    x_train_target,
    y_train_target,
    x_test_target,
    y_test_target,
    X_target_independent=None,
    y_target_independent=None,
    seed=42,
    trial=None,
    **kwargs,
):
    """
    Runs the CatBoost benchmark for different training scenarios.
    """

    catboost_params = kwargs.get("catboost_params") or {}

    if trial is not None:
        params = {
            # boosting rounds is not tuned since we have early stopping
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }
    else:
        params = {
            "depth": int(catboost_params.get("depth", 6)),
            "learning_rate": float(catboost_params.get("learning_rate", 0.03)),
            "l2_leaf_reg": float(catboost_params.get("l2_leaf_reg", 3.0)),
            "border_count": int(catboost_params.get("border_count", 254)),
        }

    params.update(
        {
            "verbose": 0,
            "random_seed": seed,
            "loss_function": "Logloss",
            "eval_metric": "MCC",
        }
    )
    params.setdefault("task_type", _CATBOOST_TASK_TYPE)
 
    if model_type == "CatBoost_source_only":
        X_train, y_train = x_train_source, y_train_source
        X_val, y_val = x_val_source, y_val_source
    elif model_type == "CatBoost_target_only":
        X_train, y_train = x_train_target, y_train_target
        # Since target validation set is removed, create a validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=(y_train >= 0.5).astype(int))
    elif model_type == "CatBoost_combined":
        X_train = pd.concat([x_train_source, x_train_target])
        y_train = pd.concat([y_train_source, y_train_target])
        # Using source validation set for consistency.
        X_val, y_val = x_val_source, y_val_source
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Binarize labels for CatBoost and calculate class weights
    y_train_bin = (y_train >= 0.5).astype(int)
    
    neg_count = (y_train_bin == 0).sum()
    pos_count = (y_train_bin == 1).sum()
    
    if pos_count > 0:
        params['scale_pos_weight'] = neg_count / pos_count

    if trial is None:
        for extra_key, extra_value in catboost_params.items():
            if extra_key not in params:
                params[extra_key] = extra_value

    if wandb.run:
        wandb.config.update(params)

    model = CatBoostClassifier(**params)
    
    # Binarize validation labels
    y_val_bin = (y_val >= 0.5).astype(int)

    model.fit(X_train, y_train_bin, eval_set=(X_val, y_val_bin), early_stopping_rounds=50, verbose=0)
    tuned_threshold, _ = _tune_threshold_GB(model, X_val, y_val)

    # Evaluation
    source_val_metrics = calculate_all_metrics(_y_to_vector(y_val_source >= 0.5), model.predict_proba(x_val_source)[:, 1], threshold=tuned_threshold)
    source_test_metrics = calculate_all_metrics(_y_to_vector(y_test_source >= 0.5), model.predict_proba(x_test_source)[:, 1], threshold=tuned_threshold)
    target_test_metrics = calculate_all_metrics(_y_to_vector(y_test_target >= 0.5), model.predict_proba(x_test_target)[:, 1], threshold=tuned_threshold)

    independent_metrics: Dict[str, float] = {}
    if X_target_independent is not None and y_target_independent is not None:
        independent_metrics = calculate_all_metrics(
            _y_to_vector(y_target_independent >= 0.5),
            model.predict_proba(X_target_independent)[:, 1],
            threshold=tuned_threshold,
        )

    results = {
        "source_val": source_val_metrics,
        "source_test": source_test_metrics,
        "target_test": target_test_metrics,
        "independent_target_test": independent_metrics,
        "selected_threshold": tuned_threshold
    }
    

    return results





def run_ssda4drug_benchmark(args, x_train_source, y_train_source, x_val_source, y_val_source, x_test_source, y_test_source, x_train_target, y_train_target, x_test_target, y_test_target, X_target_independent, y_target_independent, target_file, independent_target_file, trial=None, seed = 42):
    """
    Runs the SSDA4Drug benchmark with the given arguments and data using PyTorch Lightning.
    """
    from pytorch_lightning.utilities.combined_loader import CombinedLoader
    from frameworks.SSDA4Drug.lightning_module import SSDA4DrugLightningModule
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)

    pl.seed_everything(seed, workers=True)

    if args.get('binarize_source', True):
        y_train_source = (y_train_source >= 0.5).astype(int)
        y_val_source = (y_val_source >= 0.5).astype(int)
        y_test_source = (y_test_source >= 0.5).astype(int)

    # --- Data Handling ---
    source_train_loader = create_dataloader(x_train_source, y_train_source, args['batch_size'], shuffle=True, drop_last=True, balancing_strategy=args.get('balancing_strategy'),seed=seed)
    
    dataloader_labeled_target, dataloader_unlabeled_target = create_shot_dataloaders(
        x_train_target, y_train_target, pd.DataFrame(), pd.DataFrame(), args['batch_size'], args['n_shot'],
        seed, balancing_strategy=args.get('balancing_strategy')
    )
    
    train_loaders = {
        "source": source_train_loader,
        "labeled_target": dataloader_labeled_target["train"],
        "unlabeled_target": dataloader_unlabeled_target["train"]
    }

    source_val_loader = create_dataloader(x_val_source, y_val_source, args['batch_size'], shuffle=False)
    source_test_loader = create_dataloader(x_test_source, y_test_source, args['batch_size'], shuffle=False)
    target_test_loader = create_dataloader(x_test_target, y_test_target, args['batch_size'], shuffle=False)

    # --- Logger & Callbacks ---
    run_name = f"SSDA4Drug_{args['drug']}_{target_file}"
    if trial:
        run_name += f"_trial_{trial.number}"

    wandb_logger = WandbLogger(
        name=run_name, 
        config=args
    )
    
    patience = args.get('patience', 10)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=patience, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename='best-checkpoint')
    callbacks = [early_stop_callback, checkpoint_callback]

    # --- Model ---
    input_size = x_train_source.shape[1]
    encoder_h_dims = list(map(int, args['encoder_h_dims'].split(",")))
    predictor_h_dims = list(map(int, args['predictor_h_dims'].split(",")))

    from frameworks.SSDA4Drug import modules as ssda_modules

    latent_dim = int(args.get("latent_dim", 128))
    predictor_hidden_dims = list(map(int, predictor_h_dims))
    predictor_hidden_dim = predictor_hidden_dims[-1] if predictor_hidden_dims else 32

    encoder_choice = args.get("encoder", "DAE").lower()
    if encoder_choice == "mlp":
        feature_extractor = ssda_modules.MLP(
            input_dim=input_size,
            latent_dim=latent_dim,
            h_dims=encoder_h_dims,
            drop_out=args['dropout'],
        )
        uses_dae = False
    else:
        feature_extractor = ssda_modules.DAE(
            input_dim=latent_dim,
            fc_dim=max(latent_dim * 2, 256),
            AE_input_dim=input_size,
            AE_h_dims=encoder_h_dims,
            pretrained_weights=None,
            drop=args['dropout'],
        )
        uses_dae = True

    predictor = ssda_modules.Predictor(
        input_dim=latent_dim,
        output_dim=predictor_hidden_dim,
        drop_out=args['dropout'],
    )
    head = ssda_modules.Predictor_adentropy(
        num_class=2,
        inc=predictor_hidden_dim,
        temp=0.05,
    )

    weight_decay = args.get("weight_decay", 0.0)
    lambda_sup_t = args.get("lambda_sup_t", 1.0)
    lambda_ent_enc = args.get("lambda_ent_enc", 0.5)
    lambda_ent_pred = args.get("lambda_ent_pred", 0.5)
    lambda_fgm = args.get("lambda_fgm", 0.0)
    grl_eta = args.get("grl_eta", 0.1)
    fgm_eps = args.get("epsilon", 0.0)
    lambda_rec_source = args.get("lambda_rec_source", 1.0 if uses_dae else 0.0)
    lambda_rec_target = args.get("lambda_rec_target", 1.0 if uses_dae else 0.0)

    class_weights = None
    if args.get("balancing_strategy") == "class_weights":
        unique, counts = np.unique(y_train_source, return_counts=True)
        weight_map = {cls: len(y_train_source) / (len(unique) * count) for cls, count in zip(unique, counts)}
        class_weights = torch.tensor([weight_map[cls] for cls in sorted(weight_map.keys())], dtype=torch.float32)

    model = SSDA4DrugLightningModule(
        feature_extractor=feature_extractor,
        predictor=predictor,
        predictor_adentropy=head,
        uses_dae=uses_dae,
        num_classes=2,
        lr_f=args['lr'],
        lr_c=args['lr'],
        weight_decay=weight_decay,
        lambda_sup_t=lambda_sup_t,
        lambda_ent_enc=lambda_ent_enc,
        lambda_ent_pred=lambda_ent_pred,
        lambda_fgm=lambda_fgm,
        grl_eta=grl_eta,
        fgm_eps=fgm_eps,
        lambda_rec_source=lambda_rec_source,
        lambda_rec_target=lambda_rec_target,
        class_weights=class_weights,
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args['epochs'],
        accelerator=device,
        devices=1 if device == 'cuda' else 'auto',
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=True,
        enable_progress_bar=False,
    )

    # --- Training & Testing ---
    trainer.fit(
        model,
        train_dataloaders=CombinedLoader(train_loaders, mode="max_size_cycle"),
        val_dataloaders=source_val_loader,
    )

    if checkpoint_callback.best_model_path:
        map_location = "cuda:0" if device == "cuda" else "cpu"
        state = torch.load(checkpoint_callback.best_model_path, map_location=map_location)
        model.load_state_dict(state["state_dict"])

    best_model = model.to(device)
    best_model.eval()

    def get_preds(loader):
        labels = []
        probs = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                pred = best_model(x)
                softmax_preds = nn.Softmax(dim=1)(pred)[:, 1]
                probs.append(softmax_preds.cpu())
                labels.append(y)
        
        labels = torch.cat(labels).numpy().ravel()
        probs = torch.cat(probs).cpu().numpy().ravel()
        return labels, probs

    tuned_threshold = _tune_threshold_DL_source_only(get_preds, source_val_loader)

    source_val_labels, source_val_probs = get_preds(source_val_loader)
    source_val_metrics = calculate_all_metrics(source_val_labels, source_val_probs, threshold=tuned_threshold)

    source_test_labels, source_test_probs = get_preds(source_test_loader)
    source_test_metrics = calculate_all_metrics(source_test_labels, source_test_probs, threshold=tuned_threshold)

    target_test_labels, target_test_probs = get_preds(target_test_loader)
    target_test_metrics = calculate_all_metrics(target_test_labels, target_test_probs, threshold=tuned_threshold)

    independent_target_metrics = {}
    if X_target_independent is not None and y_target_independent is not None:
        independent_target_loader = create_dataloader(X_target_independent, y_target_independent, args['batch_size'], shuffle=False)
        ind_target_labels, ind_target_probs = get_preds(independent_target_loader)
        independent_target_metrics = calculate_all_metrics(ind_target_labels, ind_target_probs, threshold=tuned_threshold)

    results = {
        "source_val": source_val_metrics,
        "source_test": source_test_metrics,
        "target_test": target_test_metrics,
        "independent_target_test": independent_target_metrics,
        "selected_threshold": tuned_threshold
    }
    return results


def run_catboost_fewshot_baseline(
    x_val_source: pd.DataFrame,
    y_val_source: pd.Series,
    x_test_source: pd.DataFrame,
    y_test_source: pd.Series,
    x_train_target: pd.DataFrame,
    y_train_target: pd.Series,
    x_test_target: pd.DataFrame,
    y_test_target: pd.Series,
    X_target_independent: Optional[pd.DataFrame] = None,
    y_target_independent: Optional[pd.Series] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Train a shallow RandomForest on 3 positive and 3 negative target samples.
    Returns metrics on the few-shot subset and the target test set.
    """
    from catboost import CatBoostClassifier
    rng = np.random.default_rng(seed)

    if isinstance(y_train_target, pd.DataFrame):
        y_train_series = y_train_target.iloc[:, 0]
    elif isinstance(y_train_target, pd.Series):
        y_train_series = y_train_target.copy()
    else:
        y_train_series = pd.Series(y_train_target, index=x_train_target.index)

    if isinstance(y_test_target, pd.DataFrame):
        y_test_series = y_test_target.iloc[:, 0]
    elif isinstance(y_test_target, pd.Series):
        y_test_series = y_test_target.copy()
    else:
        y_test_series = pd.Series(y_test_target, index=x_test_target.index)

    if isinstance(y_target_independent, pd.DataFrame):
        y_independent_series = y_target_independent.iloc[:, 0]
    elif isinstance(y_target_independent, pd.Series):
        y_independent_series = y_target_independent.copy()
    else:
        y_independent_series = None

    y_train_bin = (y_train_series >= 0.5).astype(int)
    y_test_bin = (y_test_series >= 0.5).astype(int)

    class0_idx = y_train_bin[y_train_bin == 0].index.tolist()
    class1_idx = y_train_bin[y_train_bin == 1].index.tolist()

    if len(class0_idx) < 3 or len(class1_idx) < 3:
        raise ValueError("Not enough samples per class to build the 3-shot RF baseline.")

    fewshot_idx = rng.choice(class0_idx, 3, replace=False).tolist()
    fewshot_idx += rng.choice(class1_idx, 3, replace=False).tolist()

    x_fewshot = x_train_target.loc[fewshot_idx]
    y_fewshot = y_train_bin.loc[fewshot_idx]

    fs = CatBoostClassifier(
        iterations=30,
        depth=3,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        task_type=_CATBOOST_TASK_TYPE,
    )
    fs.fit(x_fewshot, y_fewshot)

    tuned_threshold = 0.5 # alternetive:  tuned_threshold , = _tune_threshold_GB(fs, x_val_source, y_val_source)

    # Evaluation
    source_val_metrics = calculate_all_metrics(_y_to_vector(y_val_source >= 0.5), fs.predict_proba(x_val_source)[:, 1], threshold=tuned_threshold)
    source_test_metrics = calculate_all_metrics(_y_to_vector(y_test_source >= 0.5), fs.predict_proba(x_test_source)[:, 1], threshold=tuned_threshold)
    target_test_metrics = calculate_all_metrics(_y_to_vector(y_test_target >= 0.5), fs.predict_proba(x_test_target)[:, 1], threshold=tuned_threshold)

    independent_metrics: Dict[str, float] = {}
    if X_target_independent is not None and y_independent_series is not None:
        independent_metrics = calculate_all_metrics(
            _y_to_vector(y_independent_series >= 0.5),
            fs.predict_proba(X_target_independent)[:, 1],
            threshold=tuned_threshold,
        )

    results = {
        "source_val": source_val_metrics,
        "source_test": source_test_metrics,
        "target_test": target_test_metrics,
        "independent_target_test": independent_metrics,
        "selected_threshold": tuned_threshold
    }
    

    return results


def run_scad_benchmark(args, x_train_source, y_train_source, x_val_source, y_val_source, x_test_source, y_test_source, x_train_target, y_train_target, x_test_target, y_test_target, X_target_independent, y_target_independent, target_file, independent_target_file, seed=42, trial=None):
    """
    Runs the SCAD benchmark with the given arguments and data using PyTorch Lightning.
    """
    from frameworks.SCAD.callbacks import UMAPPlotCallback
    from frameworks.SCAD.lightning_module import SCADLightningModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)

    pl.seed_everything(seed, workers=True)

    if args.get('binarize_source', True):
        y_train_source = (y_train_source >= 0.5).astype(int)
        y_val_source = (y_val_source >= 0.5).astype(int)
        y_test_source = (y_test_source >= 0.5).astype(int)


    # === Create DataLoaders ===
    balancing_strategy = args.get('balancing_strategy', None)
    class_weights = None
    if balancing_strategy == "class_weights":
        if isinstance(y_train_source, pd.DataFrame):
            y_array = y_train_source.iloc[:, 0].values
        else:
            y_array = y_train_source.values
        classes = sorted(np.unique(y_array))
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_array)

    source_balancing = balancing_strategy if balancing_strategy == "smote" else None
    source_train_loader = create_dataloader(
        x_train_source,
        y_train_source,
        args['mbS'],
        shuffle=True,
        drop_last=False,
        balancing_strategy=source_balancing,
        seed=seed,
    )
    target_train_loader = create_dataloader(
        x_train_target,
        y_train_target,
        args['mbT'],
        shuffle=True,
        drop_last=False,
        seed=seed,
    )
    train_loader = CombinedDataLoader(
        source_train_loader,
        target_train_loader,
        length_mode="target",
        min_batch_size=5,
        resample_target=False,
    )


    eval_batch = max(len(x_val_source), len(x_test_source), len(x_test_target))
    source_val_loader = create_dataloader(x_val_source, y_val_source, eval_batch, shuffle=False)

    source_test_loader = create_dataloader(x_test_source, y_test_source, eval_batch, shuffle=False)
    target_test_loader = create_dataloader(x_test_target, y_test_target, eval_batch, shuffle=False)

    # === Initialize Model ===
    input_size = x_train_source.shape[1]
    model = SCADLightningModule(
        input_dim=input_size,
        h_dim=args['h_dim'],
        predictor_z_dim=args['predictor_z_dim'],
        dropout=args['dropout'],
        lr=args['lr'],
        lam1=args['lam1'],
    )

    # --- Logger & Callbacks ---
    run_name = f"SCAD_{args['drug']}_{target_file}"
    if trial:
        run_name += f"_trial_{trial.number}"

    wandb_logger = WandbLogger(
        name=run_name, 
        config=args
    )

  
    callbacks = []
    patience = args.get("patience", None)

    if patience is not None:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=int(patience),
            mode="min",
            verbose=False,
            check_on_train_epoch_end=False,
            check_finite=True,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,        # keep only the best
            filename="best-checkpoint",
        )
        callbacks += [early_stop_callback, checkpoint_callback]
    
    if args.get('plot_umap', False):
        umap_path = os.path.join("results/SCAD/figures", args['drug'])
        callbacks.append(UMAPPlotCallback(umap_save_path=umap_path))

    # === Train Model ===
    trainer = pl.Trainer(
        max_epochs=args['epochs'],
        accelerator=device,
        devices=1 if device == 'cuda' else 'auto',
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=True,
        enable_progress_bar=False
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=source_val_loader)

    # === Evaluate Model ===
    if patience is not None:
        ckpt_path = checkpoint_callback.best_model_path
    else:
        # If no early stopping, there's no "best" model, so we'd need to define
        # what to do. For now, let's assume we want to use the last state.
        # This requires `save_last=True` in a ModelCheckpoint if used.
        # For simplicity, we'll just use the model in memory.
        ckpt_path = None

    if ckpt_path and os.path.isfile(ckpt_path):
        best_model = SCADLightningModule.load_from_checkpoint(ckpt_path)
    else:
        warnings.warn(
            f"No checkpoint available at {ckpt_path!r}; using in-memory model weights for evaluation."
        )
        best_model = model

    best_model.to("cuda" if torch.cuda.is_available() else "cpu")
    best_model.eval()

    def get_preds(loader):
        labels = []
        probs = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                features = best_model.encoder(x)
                pred = best_model.predictor(features)
                probs.append(pred.cpu())
                labels.append(y)
        
        labels = torch.cat(labels).numpy().ravel()
        probs = torch.cat(probs).cpu().numpy().ravel()
        return labels, probs

    # Tune threshold on source validation set
    tuned_threshold = _tune_threshold_DL_source_only(get_preds, source_val_loader)

    # Evaluate with tuned threshold
    source_val_labels, source_val_probs = get_preds(source_val_loader)
    source_val_metrics = calculate_all_metrics(source_val_labels, source_val_probs, threshold=tuned_threshold)

    source_test_labels, source_test_probs = get_preds(source_test_loader)
    source_test_metrics = calculate_all_metrics(source_test_labels, source_test_probs, threshold=tuned_threshold)

    target_test_labels, target_test_probs = get_preds(target_test_loader)
    target_test_metrics = calculate_all_metrics(target_test_labels, target_test_probs, threshold=tuned_threshold)

    independent_target_metrics = {}
    if X_target_independent is not None and y_target_independent is not None:
        independent_batch = len(X_target_independent)
        independent_target_loader = create_dataloader(X_target_independent, y_target_independent, max(independent_batch, 1), shuffle=False)
        ind_target_labels, ind_target_probs = get_preds(independent_target_loader)
        independent_target_metrics = calculate_all_metrics(ind_target_labels, ind_target_probs, threshold=tuned_threshold)

    results = {
        "source_val": source_val_metrics,
        "source_test": source_test_metrics, 
        "target_test": target_test_metrics, 
        "independent_target_test": independent_target_metrics,
        "selected_threshold": tuned_threshold
    }
    
    return results
