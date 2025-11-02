import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score, roc_auc_score
from copy import deepcopy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from . import modules as scdeal_m
from .scDEAL_utils import calculateKNNgraphDistanceMatrix, generateLouvainCluster
from .DaNN.mmd import mmd_loss
from training_utils import calculate_all_metrics, roc_auc_score_trainval


class scDEALLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        bottleneck_dim,
        bulk_h_dims,
        predictor_h_dims,
        dropout,
        lr,
        freeze_pretrain,
        mmd_weight,
        regularization_weight,
        k=10,
        epochs_bulk=50,
        epochs_sc=50,
        scheduler_patience=5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # --- Instantiate all sub-models ---
        self.source_encoder = scdeal_m.AEBase(input_dim, bottleneck_dim, bulk_h_dims, dropout)
        self.source_predictor = scdeal_m.PretrainedPredictor(
            input_dim=input_dim, latent_dim=bottleneck_dim, h_dims=bulk_h_dims,
            hidden_dims_predictor=predictor_h_dims, output_dim=2,
            pretrained_weights=None, freezed=bool(freeze_pretrain),
            drop_out=dropout, drop_out_predictor=dropout
        )
        self.target_encoder = scdeal_m.AEBase(input_dim, bottleneck_dim, bulk_h_dims, dropout)
        self.dann_model = scdeal_m.DaNN(source_model=self.source_predictor, target_model=self.target_encoder)

        self.loss_c_fn = nn.CrossEntropyLoss()
        self._best_val_loss = float("inf")
        self._best_state = None
        self._scheduler = None
        self._train_loss_total = 0.0
        self._train_loss_batches = 0

    def setup(self, stage: str):
        if stage == 'fit':
            print("--- Running scDEAL pre-training setup ---")
            dm = self.trainer.datamodule
            
            pretrainer = pl.Trainer(
                max_epochs=self.hparams.epochs_bulk,
                accelerator="auto",
                devices="auto",
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False,
                deterministic=True
            )

            if hasattr(dm, "source_autoencoder_train_dataloader"):
                source_train_loader_ae = dm.source_autoencoder_train_dataloader()
                source_val_loader_ae = dm.source_autoencoder_val_dataloader()
            else:
                source_train_loader_ae = dm.source_train_dataloader()
                source_val_loader_ae = dm.source_val_dataloader()

            if hasattr(dm, "source_classifier_train_dataloader"):
                source_train_loader_cls = dm.source_classifier_train_dataloader()
                source_val_loader_cls = dm.source_classifier_val_dataloader()
            else:
                source_train_loader_cls = dm.source_train_dataloader()
                source_val_loader_cls = dm.source_val_dataloader()

            if hasattr(dm, "target_autoencoder_train_dataloader"):
                target_train_loader_ae = dm.target_autoencoder_train_dataloader()
                target_val_loader_ae = dm.target_autoencoder_val_dataloader()
            else:
                target_train_loader_ae = dm.target_train_dataloader()
                target_val_loader_ae = dm.target_val_dataloader()

            print("--- Stage 1: Pre-training bulk encoder ---")
            source_ae_module = AELightningModule(
                self.source_encoder,
                self.hparams.lr,
                noise_prob=0.0,
                scheduler_patience=self.hparams.scheduler_patience,
            )
            pretrainer.fit(source_ae_module, source_train_loader_ae, source_val_loader_ae)
            if getattr(source_ae_module, "_best_state", None) is not None:
                self.source_encoder.load_state_dict(source_ae_module._best_state)
            
            print("--- Stage 2: Training bulk predictor ---")
            self.source_predictor.load_state_dict(self.source_encoder.state_dict(), strict=False)
            predictor_module = PredictorLightningModule(
                self.source_predictor,
                self.hparams.lr,
                scheduler_patience=self.hparams.scheduler_patience,
            )
            pretrainer.fit(predictor_module, source_train_loader_cls, source_val_loader_cls)
            if getattr(predictor_module, "_best_state", None) is not None:
                self.source_predictor.load_state_dict(predictor_module._best_state)

            print("--- Stage 3: Pre-training single-cell encoder ---")
            sc_pretrainer = pl.Trainer(
                max_epochs=self.hparams.epochs_sc,
                accelerator="auto",
                devices="auto",
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False,
                deterministic=True
            )
            sc_ae_module = AELightningModule(
                self.target_encoder,
                self.hparams.lr,
                noise_prob=0.2,
                scheduler_patience=self.hparams.scheduler_patience,
            )
            sc_pretrainer.fit(sc_ae_module, target_train_loader_ae, target_val_loader_ae)
            if getattr(sc_ae_module, "_best_state", None) is not None:
                self.target_encoder.load_state_dict(sc_ae_module._best_state)
            
            print("--- scDEAL pre-training finished ---")
            for param in self.dann_model.source_model.parameters():
                param.requires_grad_(False)

    def on_fit_start(self):
        self._best_val_loss = float("inf")
        self._best_state = deepcopy(self.dann_model.state_dict())

    def on_train_epoch_start(self):
        self._train_loss_total = 0.0
        self._train_loss_batches = 0

    def on_train_epoch_end(self):
        if self._scheduler is None or self._train_loss_batches == 0:
            return
        avg_loss = self._train_loss_total / self._train_loss_batches
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device, dtype=torch.float32)
        self._scheduler.step(avg_loss)
        self.log("train_epoch_loss", avg_loss_tensor, on_epoch=True, prog_bar=False, logger=True)

    def on_validation_epoch_end(self):
        if self.trainer is None or getattr(self.trainer, "sanity_checking", False):
            return
        val_loss_tensor = self.trainer.callback_metrics.get("val_loss")
        if val_loss_tensor is None:
            return
        val_loss = float(val_loss_tensor.detach().cpu())
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_state = deepcopy(self.dann_model.state_dict())

    def on_train_end(self):
        if self._best_state is not None:
            self.dann_model.load_state_dict(self._best_state)

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        source_batch, target_batch = batch
        x_src, y_src = source_batch
        x_tar, y_tar = target_batch


        # Drop batches with < 2 classes to prevent metric calculation errors.

        if torch.unique(y_src).size(0) < 2:
            return None

        min_size = min(x_src.shape[0], x_tar.shape[0])
        x_src, y_src = x_src[:min_size], y_src[:min_size]
        x_tar, y_tar = x_tar[:min_size], y_tar[:min_size]

        y_pre, x_src_mmd, x_tar_mmd = self.dann_model(x_src, x_tar)

        y_src_labels = y_src.view(-1).long()
        loss_c = self.loss_c_fn(y_pre, y_src_labels)
        loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)

        loss_s_value = 0.0
        if x_tar.shape[0] >= self.hparams.k:
            # 1. Get latent embeddings for clustering
            tgt_embeddings = self.dann_model.target_model.encoder(x_tar)
            
            # 2. Perform KNN and Louvain on latent embeddings (NumPy)
            edgeList = calculateKNNgraphDistanceMatrix(
                tgt_embeddings.cpu().detach().numpy(), distanceType="euclidean", k=self.hparams.k
            )
            listResult, size = generateLouvainCluster(edgeList)

            if size > 0:
                # 3. Calculate penalty based on raw features (NumPy)
                cluster_indices = np.asarray(listResult)
                for i in range(size):
                    members = cluster_indices == i
                    if np.sum(members) > 1:
                        # Original implementation bug: uses raw x_tar for similarity
                        s = cosine_similarity(x_tar[members, :].cpu().detach().numpy())
                        s = 1 - s
                        # Original implementation bug: incorrect denominator
                        denom = (2 * s.shape[0] * s.shape[0] - s.shape[0])
                        if denom > 0:
                            loss_s_value += np.sum(np.triu(s, 1)) / denom
        
        # 4. Convert to tensor, replicating the gradient bug
        loss_s = torch.tensor(loss_s_value, device=self.device, dtype=x_tar.dtype)
        loss_s.requires_grad_(True)
        loss = loss_c + self.hparams.mmd_weight * loss_mmd + self.hparams.regularization_weight * loss_s

        opt.zero_grad()
        self.manual_backward(loss, retain_graph=True) # retain_graph is needed here due to loss_s logic
        opt.step()

        y_prob = y_pre[:, 1].detach().cpu().numpy()
        y_true = y_src_labels.cpu().numpy()
        auc = roc_auc_score_trainval(y_true, y_prob)
        aupr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

        self._train_loss_total += float(loss.detach().cpu())
        self._train_loss_batches += 1

        self.log_dict({
            'train_loss': loss, 'train_loss_c': loss_c, 'train_loss_mmd': loss_mmd,
            'train_loss_sc': loss_s.detach(), 'train_auc': auc, 'train_aupr': aupr
        }, on_step=False, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        source_batch, _ = batch
        x_src, y_src = source_batch

        # Drop batches with < 2 classes to prevent metric calculation errors.
        if torch.unique(y_src).size(0) < 2:
            return None

        # Ignore target batch during validation to match original logic
        if x_src.shape[0] == 0:
            return

        y_pre = self.dann_model.source_model(x_src)
        
        y_src_labels = y_src.view(-1).long()
        loss_c = self.loss_c_fn(y_pre, y_src_labels)
        loss = loss_c
        
        y_prob = y_pre[:, 1].detach().cpu().numpy()
        y_true = y_src_labels.cpu().numpy()
        auc = roc_auc_score_trainval(y_true, y_prob)
        
        self.log_dict({'val_loss': loss, 'val_auc': auc}, on_epoch=True, prog_bar=True, logger=True)

    def on_test_start(self):
        self.test_step_outputs = {}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        predictions = self.dann_model.source_model(x)
        probs = predictions[:, 1]
        output = {'preds': probs, 'labels': y.view(-1)}

        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = []
        self.test_step_outputs[dataloader_idx].append(output)

        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        for dataloader_idx in sorted(outputs.keys()):
            dl_outputs = outputs[dataloader_idx]
            prefix = "source_test" if dataloader_idx == 0 else "target_test"
            
            if not dl_outputs:
                continue

            all_preds = torch.cat([o['preds'] for o in dl_outputs]).view(-1).cpu().numpy()
            all_labels = torch.cat([o['labels'] for o in dl_outputs]).view(-1).cpu().numpy()

            metrics = calculate_all_metrics(all_labels, all_preds)
            
            for key, value in metrics.items():
                self.log(f"{prefix}_{key}", value, logger=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.dann_model.target_model.parameters(), lr=self.hparams.lr)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.hparams.scheduler_patience,
        )
        return optimizer

# Helper modules for pre-training
class _BestWeightMixin:
    """Utility mixin to keep track of the best validation loss weights."""

    def _init_best_tracking(self):
        self._best_loss = float("inf")
        self._best_state = deepcopy(self.model.state_dict())

    def on_validation_epoch_end(self):
        if self.trainer is None:
            return
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        loss_value = float(val_loss.detach().cpu())
        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._best_state = deepcopy(self.model.state_dict())

    def on_fit_end(self):
        if getattr(self, "_best_state", None) is not None:
            self.model.load_state_dict(self._best_state)


class AELightningModule(_BestWeightMixin, pl.LightningModule):
    def __init__(self, model, learning_rate, noise_prob: float = 0.0, scheduler_patience: int = 5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.noise_prob = noise_prob
        self.scheduler_patience = scheduler_patience
        self._init_best_tracking()

    def forward(self, x):
        return self.model(x)

    def _apply_noise(self, x):
        if self.noise_prob <= 0.0 or not self.training:
            return x
        mask = torch.rand_like(x).lt(self.noise_prob)
        return x.masked_fill(mask, 0.0)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self.model(self._apply_noise(x))
        loss = self.loss_fn(reconstructed, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self.model(x)
        loss = self.loss_fn(reconstructed, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.scheduler_patience,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class PredictorLightningModule(_BestWeightMixin, pl.LightningModule):
    def __init__(self, model, learning_rate, scheduler_patience: int = 5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler_patience = scheduler_patience
        self._init_best_tracking()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y.view(-1).long())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y.view(-1).long())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.scheduler_patience,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

class DaNNLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate, alpha, beta, k=10, scheduler_patience: int = 5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.alpha = alpha # mmd_weight
        self.beta = beta # regularization_weight
        self.k = k
        self.scheduler_patience = scheduler_patience
        self.loss_c_fn = nn.CrossEntropyLoss()
        self.automatic_optimization = False
        self._scheduler = None
        self._train_loss_total = 0.0
        self._train_loss_batches = 0

    def on_train_epoch_start(self):
        self._train_loss_total = 0.0
        self._train_loss_batches = 0

    def on_train_epoch_end(self):
        if self._scheduler is None or self._train_loss_batches == 0:
            return
        avg_loss = self._train_loss_total / self._train_loss_batches
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device, dtype=torch.float32)
        self._scheduler.step(avg_loss)
        self.log("train_epoch_loss", avg_loss_tensor, on_epoch=True, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        source_batch, target_batch = batch
        x_src, y_src = source_batch
        x_tar, y_tar = target_batch

        min_size = min(x_src.shape[0], x_tar.shape[0])
        x_src, y_src = x_src[:min_size], y_src[:min_size]
        x_tar, y_tar = x_tar[:min_size], y_tar[:min_size]

        y_pre, x_src_mmd, x_tar_mmd = self.model(x_src, x_tar)
        
        encoder_rep = self.model.target_model.encoder(x_tar)

        if encoder_rep.shape[0] < self.k:
            return None

        y_src_labels = y_src.view(-1).long()
        loss_c = self.loss_c_fn(y_pre, y_src_labels)
        loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)

        edgeList = calculateKNNgraphDistanceMatrix(encoder_rep.cpu().detach().numpy(), distanceType="euclidean", k=self.k)
        listResult, size = generateLouvainCluster(edgeList)
        
        loss_s = 0
        if size > 0:
            for i in range(size):
                cluster_indices = np.asarray(listResult) == i
                if np.sum(cluster_indices) > 1:
                    s = cosine_similarity(x_tar[cluster_indices, :].cpu().detach().numpy())
                    s = 1 - s
                    loss_s += np.sum(np.triu(s, 1)) / (2 * s.shape[0] * s.shape[0] - s.shape[0])

        loss_s = torch.tensor(loss_s, device=self.device, dtype=torch.float32)
        loss_s.requires_grad_(True)
        loss = loss_c + self.alpha * loss_mmd + self.beta * loss_s

        opt.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        opt.step()

        y_prob = y_pre[:, 1].detach().cpu().numpy()
        y_true = y_src_labels.cpu().numpy()
        auc = roc_auc_score_trainval(y_true, y_prob)
        aupr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

        self._train_loss_total += float(loss.detach().cpu())
        self._train_loss_batches += 1

        self.log_dict({
            'train_loss': loss, 'train_loss_c': loss_c, 'train_loss_mmd': loss_mmd,
            'train_loss_sc': loss_s.detach(), 'train_auc': auc, 'train_aupr': aupr
        }, on_step=False, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_src, y_src = batch

        y_pre = self.model.source_model(x_src)
        
        y_src_labels = y_src.view(-1).long()
        loss_c = self.loss_c_fn(y_pre, y_src_labels)
        loss = loss_c
        
        y_prob = y_pre[:, 1].detach().cpu().numpy()
        y_true = y_src_labels.cpu().numpy()
        auc = roc_auc_score_trainval(y_true, y_prob)
        
        self.log_dict({'val_loss': loss, 'val_loss_c': loss_c, 'val_auc': auc}, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.target_model.parameters(), lr=self.learning_rate)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.scheduler_patience,
        )
        return optimizer
