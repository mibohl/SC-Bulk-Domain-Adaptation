import pytorch_lightning as pl
import torch
import torch.nn as nn
import itertools
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning.utilities import rank_zero_warn

# Add project root to system path for module imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    # Prefer package-relative import when running via `python -m`.
    from . import modules as m  # type: ignore
except ImportError:
    # Fallback for direct script execution.
    import modules as m  # type: ignore
from training_utils import calculate_all_metrics


def roc_auc_score_trainval(y_true, y_pred_proba):
    """Safe roc_auc_score for training/validation where a batch might have only one class."""
    if len(np.unique(y_true)) < 2:
        return 0.5  # A neutral value
    return roc_auc_score(y_true, y_pred_proba)


class SCADLightningModule(pl.LightningModule):
    def __init__(self, input_dim, h_dim, predictor_z_dim, dropout, lr, lam1):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = m.FX(h_dim=h_dim, input_dim=input_dim, dropout_rate=dropout)
        self.predictor = m.MTLP(h_dim=h_dim, z_dim=predictor_z_dim, dropout_rate=dropout)
        self.discriminator = m.Discriminator(h_dim=h_dim, dropout_rate=dropout)

        self.loss_c = nn.BCELoss()

    def forward(self, x):
        return self.encoder(x)

    def _common_step(self, batch, batch_idx, phase):
        source_data, target_data = batch
        xs, ys = source_data
        xt, _ = target_data

        ys = ys.view(-1, 1)

        F_xs = self.encoder(xs)
        F_xt = self.encoder(xt)

        # Guard against non-finite activations propagating through the network.
        dirty_encoder = (not torch.isfinite(F_xs).all()) or (not torch.isfinite(F_xt).all())
        if dirty_encoder:
            self.log(
                f"{phase}_encoder_non_finite",
                1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            rank_zero_warn(
                f"[SCAD] Non-finite encoder activations detected during {phase}; "
                "values will be sanitized to keep training running. Please inspect upstream preprocessing."
            )
            F_xs = torch.nan_to_num(F_xs, nan=0.0, posinf=0.0, neginf=0.0)
            F_xt = torch.nan_to_num(F_xt, nan=0.0, posinf=0.0, neginf=0.0)

        yhat_xs_raw = self.predictor(F_xs)
        # BCE expects probabilities in (0,1); clamp after cleaning NaNs/Infs.
        yhat_xs = torch.nan_to_num(yhat_xs_raw, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-9, 1 - 1e-9)
        loss1 = self.loss_c(yhat_xs, ys.float())

        # Domain adversarial training
        labels_source = torch.ones(F_xs.size(0), 1, device=self.device)
        labels_target = torch.zeros(F_xt.size(0), 1, device=self.device)
        domain_labels = torch.cat([labels_source, labels_target], 0)
        features_combined = torch.cat([F_xs, F_xt], 0)

        yhat_DG_raw = self.discriminator(features_combined)
        yhat_DG = torch.nan_to_num(yhat_DG_raw, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-9, 1 - 1e-9)
        DG_loss = self.loss_c(yhat_DG, domain_labels)

        loss2 = self.hparams.lam1 * DG_loss
        loss = loss1 + loss2

        # Metrics
        ys_cpu = ys.cpu()
        y_pre_cpu = yhat_xs.cpu()

        if torch.isnan(y_pre_cpu).any():
            auc = 0.5
            aupr = ys_cpu.float().mean().item()
        else:
            ys_binary = (ys_cpu.detach().numpy() >= 0.5).astype(int)
            auc = roc_auc_score_trainval(ys_binary, y_pre_cpu.detach().numpy())
            aupr = average_precision_score(ys_binary, y_pre_cpu.detach().numpy())

        self.log(f"{phase}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{phase}_loss_c", loss1, on_step=False, on_epoch=True, logger=True)
        self.log(f"{phase}_loss_dann", loss2, on_step=False, on_epoch=True, logger=True)
        self.log(f"{phase}_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{phase}_aupr", aupr, on_step=False, on_epoch=True, logger=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        ys = ys.view(-1, 1)

        F_xs = self.encoder(xs)
        if not torch.isfinite(F_xs).all():
            self.log(
                "val_encoder_non_finite",
                1.0,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            rank_zero_warn(
                "[SCAD] Non-finite encoder activations detected during validation; "
                "values will be sanitized."
            )
            F_xs = torch.nan_to_num(F_xs, nan=0.0, posinf=0.0, neginf=0.0)

        yhat_xs_raw = self.predictor(F_xs)
        if not torch.isfinite(yhat_xs_raw).all():
            rank_zero_warn(
                "[SCAD] Predictor produced non-finite outputs during validation; "
                "outputs were clamped after sanitizing."
            )
        yhat_xs = torch.nan_to_num(yhat_xs_raw, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-9, 1 - 1e-9)
        loss = self.loss_c(yhat_xs, ys.float())

        # Metrics
        ys_cpu = ys.cpu()
        y_pre_cpu = yhat_xs.cpu()

        if torch.isnan(y_pre_cpu).any():
            auc = 0.5
            aupr = ys_cpu.float().mean().item()
        else:
            ys_binary = (ys_cpu.detach().numpy() >= 0.5).astype(int)
            auc = roc_auc_score_trainval(ys_binary, y_pre_cpu.detach().numpy())
            aupr = average_precision_score(ys_binary, y_pre_cpu.detach().numpy())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_aupr", aupr, on_step=False, on_epoch=True, logger=True)
        return {"val_loss": loss.detach()}

    def on_test_start(self):
        self.test_step_outputs = {}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        features = self.encoder(x)
        if not torch.isfinite(features).all():
            rank_zero_warn(
                "[SCAD] Non-finite encoder activations detected during testing; "
                "values will be sanitized."
            )
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        predictions_raw = self.predictor(features)
        if not torch.isfinite(predictions_raw).all():
            rank_zero_warn(
                "[SCAD] Predictor produced non-finite outputs during testing; "
                "outputs were clamped after sanitizing."
            )
        predictions = torch.nan_to_num(predictions_raw, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-9, 1 - 1e-9)
        output = {'preds': predictions, 'labels': y}
        
        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = []
        self.test_step_outputs[dataloader_idx].append(output)
        
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        for dataloader_idx in sorted(outputs.keys()):
            dl_outputs = outputs[dataloader_idx]
            if dataloader_idx == 0:
                prefix = "source_test"
            elif dataloader_idx == 1:
                prefix = "target_test"
            else:
                prefix = "independent_target_test"
            
            if not dl_outputs:
                continue

            all_preds = torch.cat([o['preds'] for o in dl_outputs]).cpu().numpy()
            all_labels = torch.cat([o['labels'] for o in dl_outputs]).cpu().numpy()

            metrics = calculate_all_metrics(all_labels, all_preds)
            
            # Log metrics with the dataloader prefix
            for key, value in metrics.items():
                self.log(f"{prefix}_{key}", value, logger=True)
        
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(
            itertools.chain(self.encoder.parameters(), self.predictor.parameters(), self.discriminator.parameters()),
            lr=self.hparams.lr
        )
        return optimizer
