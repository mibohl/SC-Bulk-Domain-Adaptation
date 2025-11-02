"""
Lightning module for scATD-sf-dist.

This module implements the two-stage training process:
1. Pre-training (Distillation): In the `setup` hook, it loads the weights of a
   pre-trained Dist-VAE model. This step replaces the need for live distillation.
2. Fine-tuning (Domain Adaptation): The `training_step` performs domain adaptation
   using a classification loss on source data and an MMD loss between source and
   target latent representations.
"""
import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import sys
import os
from pytorch_lightning.loggers.logger import DummyLogger

# Add project root to system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from frameworks.scATD import modules as scATD_m
from training_utils import calculate_all_metrics

class scATDLightningModule(pl.LightningModule):
    def __init__(self, input_dim, z_dim, hidden_dim_layer0, hidden_dim_layer_out_Z, 
                 Encoder_layer_dims, Decoder_layer_dims, lr, mmd_weight, 
                 pretrained_model_path, epochs, epochs_classifier, weight_decay=1e-3, post_training=True, post_training_epoch_num=10,
                 class_weights=None, target_pool=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.vae = scATD_m.ContinuousResidualVAE(
            input_dim=input_dim,
            hidden_dim_layer0=hidden_dim_layer0,
            Encoder_layer_dims=Encoder_layer_dims,
            Decoder_layer_dims=Decoder_layer_dims,
            hidden_dim_layer_out_Z=hidden_dim_layer_out_Z,
            z_dim=z_dim,
        )
        self.model = scATD_m.VAEclassification(
            model_pretraining=self.vae, z_dim=z_dim, class_num=2
        )
        if class_weights:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None
        self.automatic_optimization = False
        self.training_phase = "classifier"
        self._initial_target_pool = target_pool
        self.target_pool_tensor = None
        self.scheduler = None

    def _log_to_logger(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        logger = getattr(trainer, "logger", None)
        if logger is None or isinstance(logger, DummyLogger):
            return False
        return True

    def setup(self, stage: str):
        if stage == 'fit':
            print("--- Loading pre-trained Dist-VAE model weights ---")
            if not os.path.exists(self.hparams.pretrained_model_path):
                raise FileNotFoundError(f"Pre-trained model not found at: {self.hparams.pretrained_model_path}")
            
            pretrained_state = torch.load(self.hparams.pretrained_model_path, map_location=torch.device('cpu'), weights_only=True)
            
            self.vae.load_adapted_state_dict(pretrained_state, strict=False)
            print("Pre-trained weights loaded successfully.")

            self._freeze_encoder()
            print("Encoder layers frozen for the initial phase of the fine-tuning.")

    def _freeze_encoder(self):
        for module in (self.model.Encoder_resblocks, self.model.fc21, self.model.fc22):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def _unfreeze_encoder(self):
        for module in (self.model.Encoder_resblocks, self.model.fc21, self.model.fc22):
            module.train()
            for param in module.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        source_batch, target_batch = batch
        opt = self.optimizers()

        if self.training_phase == "classifier":
            x_src, y_src = source_batch
            y_pred = self.model(x_src)
            loss = self._classification_loss(y_pred, y_src)

            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

            y_pred_proba = torch.softmax(y_pred, dim=1)[:, 1].cpu().detach().numpy()
            y_true = y_src.cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
            aupr = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5

            self.log_dict({
                'clf_train_loss': loss,
                'clf_train_auc': auc,
                'clf_train_aupr': aupr
            }, on_step=False, on_epoch=True, prog_bar=True, logger=self._log_to_logger())
            return loss

        # --- Fine-tuning phase ---
        if target_batch is None:
            raise RuntimeError("Target batch missing during fine-tuning phase.")

        x_src, y_src = source_batch
        x_tar = self._sample_target_batch(x_src, target_batch[0])

        # --- Classification Loss on Source Data ---
        y_pred = self.model(x_src)
        loss_c = self._classification_loss(y_pred, y_src)

        # --- MMD Loss for Domain Alignment ---
        source_z = self.model.get_embeddings(x_src)
        target_z = self.model.get_embeddings(x_tar)
        loss_mmd = scATD_m.mmd_loss(source_z, target_z)

        # --- Total Loss ---
        loss = loss_c + self.hparams.mmd_weight * loss_mmd

        # --- Optimization ---
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # --- Logging ---
        y_pred_proba = torch.softmax(y_pred, dim=1)[:, 1].cpu().detach().numpy()
        y_true = y_src.cpu().detach().numpy()
        
        auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        aupr = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5

        self.log_dict({
            'train_loss': loss,
            'train_loss_c': loss_c,
            'train_loss_mmd': loss_mmd,
            'train_auc': auc,
            'train_aupr': aupr
        }, on_step=False, on_epoch=True, prog_bar=True, logger=self._log_to_logger())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model.forward_deterministic(x)
        loss = self._classification_loss(y_pred, y)
        y_pred_proba = torch.softmax(y_pred, dim=1)[:, 1].cpu().numpy()
        y_true = y.cpu().numpy()
        auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        aupr = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        self.log_dict({
            'val_loss': loss,
            'val_auc': auc,
            'val_aupr': aupr
        }, on_epoch=True, prog_bar=True, logger=self._log_to_logger())

    def on_train_epoch_start(self):
        if self.current_epoch == self.hparams.epochs_classifier:
            self.training_phase = "finetune"
            self._unfreeze_encoder()
            print(f"--- Unfreezing encoder at epoch {self.current_epoch} for fine-tuning ---")
            
            # Manually instantiate the scheduler now that we have the optimizer
            if self.scheduler is None:
                optimizer = self.optimizers()
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, self.hparams.epochs // 4),
                    eta_min=self.hparams.lr * 1e-2,
                )

    def on_test_start(self):
        self.test_step_outputs = {}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        predictions = self.model.forward_deterministic(x)
        softmax_preds = torch.softmax(predictions, dim=1)[:, 1]
        output = {'preds': softmax_preds, 'labels': y}
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
            all_preds = torch.cat([o['preds'] for o in dl_outputs]).cpu().numpy()
            all_labels = torch.cat([o['labels'] for o in dl_outputs]).cpu().numpy()
            metrics = calculate_all_metrics(all_labels, all_preds)
            for key, value in metrics.items():
                self.log(f"{prefix}_{key}", value, logger=self._log_to_logger())
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def _classification_loss(self, logits, labels):
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        return F.cross_entropy(logits, labels.long(), weight=weight)

    def on_train_epoch_end(self):
        # Manual scheduler step
        if self.scheduler is not None and self.training_phase == "finetune":
            self.scheduler.step()

    def _sample_target_batch(self, x_src, current_batch=None):
        batch_size = x_src.size(0)
        if self.target_pool_tensor is None:
            if self._initial_target_pool is not None:
                self.target_pool_tensor = torch.tensor(self._initial_target_pool, dtype=torch.float32)
            else:
                datamodule = getattr(self.trainer, "datamodule", None)
                if datamodule is not None and hasattr(datamodule, "x_train_target"):
                    target_df = datamodule.x_train_target
                    self.target_pool_tensor = torch.tensor(target_df.values, dtype=torch.float32)
                elif current_batch is not None:
                    self.target_pool_tensor = current_batch.detach().cpu().float()
                else:
                    raise RuntimeError("Target training data not available for sampling.")
        if self.target_pool_tensor.device != x_src.device:
            self.target_pool_tensor = self.target_pool_tensor.to(x_src.device)
        pool = self.target_pool_tensor
        if pool.size(0) == 0:
            raise RuntimeError("Target pool is empty.")
        
        # Sample with replacement to ensure batch size always matches source
        indices = torch.randint(0, pool.size(0), (batch_size,), device=pool.device)
        target_batch = pool.index_select(0, indices).to(dtype=x_src.dtype)
        return target_batch
