import pytorch_lightning as pl
import torch
from torch import nn, optim
import itertools

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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SSDA4DrugLightningModule(pl.LightningModule):
    """
    Semi-Supervised Domain Adaptation for Drug sensitivity (SSDA4Drug)
    -------------------------------------------------------------------
    This LightningModule mirrors the original training logic (source CE,
    few-shot target CE, minimax entropy on unlabeled target) and adds a
    *working* Fast Gradient Method (FGM) implemented at the **feature level**.

    Key ideas:
      - Feature extractor F and entropy head C (Predictor_adentropy-style).
      - Minimax entropy on unlabeled target:
          * predictor branch: maximize entropy (via GRL / reverse=True),
            encoder is detached – updates C only.
          * encoder branch: minimize entropy (reverse=False), updates F.
      - FGM on unlabeled target features: compute a tiny adversarial delta
        in the feature space and minimize entropy at z_adv = z + delta,
        which improves local smoothness of F around target samples.

    Expected train batch structure from a CombinedLoader(dict):
        {
          "source": (x_s, y_s),
          "labeled_target": (x_tl, y_tl),      # optional if n_shot == 0
          "unlabeled_target": (x_tu, _dummy),  # _dummy can be None/zeros
        }

    Validation/test step can use a single dataloader that yields (x, y) on the
    target domain (or multiple – adapt as needed in your DataModule).

    Parameters
    ----------
    feature_extractor : nn.Module
        Encoder F: maps gene-expression features to latent features.
    predictor_adentropy : nn.Module
        Classifier C with a signature like `forward(z, reverse=False, eta=...)`
        (as in the original Predictor_adentropy with GRL inside).
    num_classes : int
        Number of classes (default: 2).
    lr_f : float
        Learning rate for feature extractor.
    lr_c : float
        Learning rate for predictor head.
    weight_decay : float
        Weight decay.
    lambda_sup_t : float
        Weight for labeled target CE term.
    lambda_ent_enc : float
        Weight for encoder-entropy minimization term on unlabeled target.
    lambda_ent_pred : float
        Weight for predictor-entropy maximization term on unlabeled target.
    lambda_fgm : float
        Weight for the FGM (adversarial) entropy term on unlabeled target.
    ent_temp : float
        Temperature applied inside the predictor (should match C's temp).
    grl_eta : float
        Coefficient for gradient reversal when calling predictor with reverse=True.
    fgm_eps : float
        L2 step size for the feature-level adversarial perturbation.
    class_weights : Optional[torch.Tensor]
        Optional class weighting for CE (on device at runtime).
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        predictor: nn.Module,
        predictor_adentropy: nn.Module,
        uses_dae: bool = False,
        num_classes: int = 2,
        lr_f: float = 1e-3,
        lr_c: float = 1e-3,
        weight_decay: float = 0.0,
        lambda_sup_t: float = 1.0,
        lambda_ent_enc: float = 0.5,
        lambda_ent_pred: float = 0.5,
        lambda_fgm: float = 0.2,
        ent_temp: float = 0.05,
        grl_eta: float = 0.1,
        fgm_eps: float = 1e-2,
        class_weights: Optional[torch.Tensor] = None,
        lambda_rec_source: float = 1.0,
        lambda_rec_target: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "predictor", "predictor_adentropy", "class_weights"])  # noqa: E501

        self.F = feature_extractor
        self.P = predictor
        self.C = predictor_adentropy
        self.num_classes = num_classes
        self.uses_dae = uses_dae

        self.lr_f = lr_f
        self.lr_c = lr_c
        self.weight_decay = weight_decay

        self.lambda_sup_t = lambda_sup_t
        self.lambda_ent_enc = lambda_ent_enc
        self.lambda_ent_pred = lambda_ent_pred
        self.lambda_fgm = lambda_fgm
        self.ent_temp = ent_temp
        self.grl_eta = grl_eta
        self.fgm_eps = fgm_eps
        self.lambda_rec_source = lambda_rec_source
        self.lambda_rec_target = lambda_rec_target

        self.register_buffer("_class_weights", class_weights if class_weights is not None else torch.tensor([]))
        self.ce = nn.CrossEntropyLoss(weight=None if self._class_weights.numel() == 0 else self._class_weights)
        self.reconstruction_criterion = nn.MSELoss()

    # ----------------------- helpers -----------------------
    def _forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.uses_dae:
            features, recon = self.F(x.clone())
            return features, recon
        features = self.F(x)
        return features, None

    # ----------------------- utility losses -----------------------
    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Mean entropy of softmax(logits)."""
        p = F.softmax(logits, dim=1)
        ent = -torch.sum(p * torch.log(p + eps), dim=1)
        return ent.mean()

    @staticmethod
    def _l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.linalg.norm(v, ord=2, dim=1, keepdim=True)
        return v / (norm + eps)

    # ----------------------- forward -----------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self._forward_encoder(x)
        logits = self.C(self.P(features), reverse=False)
        return logits

    # ----------------------- training step -----------------------
    def training_step(self, batch: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]], batch_idx: int):
        # unpack
        x_s, y_s = batch.get("source", (None, None))
        x_tl, y_tl = batch.get("labeled_target", (None, None))
        x_tu, _ = batch.get("unlabeled_target", (None, None))

        total_loss = 0.0

        # 1) Source supervised CE
        if x_s is not None and y_s is not None:
            z_s, recon_s = self._forward_encoder(x_s)
            h_s = self.P(z_s)
            logits_s = self.C(h_s, reverse=False)
            loss_s = self.ce(logits_s, y_s.long())
            total_loss = total_loss + loss_s
            with torch.no_grad():
                acc_s = (logits_s.argmax(1) == y_s).float().mean()
            self.log("train/loss_source", loss_s, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/acc_source", acc_s, on_step=True, on_epoch=True, prog_bar=True)
            if recon_s is not None and self.lambda_rec_source > 0:
                loss_rec_s = self.reconstruction_criterion(recon_s, x_s) * self.lambda_rec_source
                total_loss = total_loss + loss_rec_s
                self.log("train/loss_rec_source", loss_rec_s, on_step=True, on_epoch=True, prog_bar=False)

        # 2) Few-shot target supervised CE (optional)
        if x_tl is not None and y_tl is not None and self.lambda_sup_t > 0:
            z_tl, recon_tl = self._forward_encoder(x_tl)
            h_tl = self.P(z_tl)
            logits_tl = self.C(h_tl, reverse=False)
            loss_tl = self.ce(logits_tl, y_tl.long()) * self.lambda_sup_t
            total_loss = total_loss + loss_tl
            with torch.no_grad():
                acc_tl = (logits_tl.argmax(1) == y_tl).float().mean()
            self.log("train/loss_tgt_sup", loss_tl, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/acc_tgt_sup", acc_tl, on_step=True, on_epoch=True, prog_bar=False)
            if recon_tl is not None and self.lambda_rec_target > 0:
                loss_rec_tl = self.reconstruction_criterion(recon_tl, x_tl) * self.lambda_rec_target
                total_loss = total_loss + loss_rec_tl
                self.log("train/loss_rec_target", loss_rec_tl, on_step=True, on_epoch=True, prog_bar=False)

        # 3) Unlabeled target: minimax entropy + FGM
        if x_tu is not None:
            # 3a) Predictor branch (maximize entropy): detach encoder to update C only
            with torch.no_grad():
                z_tu_det, _ = self._forward_encoder(x_tu)
                h_tu_det = self.P(z_tu_det)
            logits_pred_rev = self.C(h_tu_det, reverse=True, eta=self.grl_eta)
            loss_pred_rev = self._entropy_from_logits(logits_pred_rev) * self.lambda_ent_pred
            total_loss = total_loss + loss_pred_rev
            self.log("train/loss_pred_rev_ent", loss_pred_rev, on_step=True, on_epoch=True, prog_bar=False)

            # 3b) Encoder branch (minimize entropy): normal forward (updates F)
            z_tu, _ = self._forward_encoder(x_tu)
            h_tu = self.P(z_tu)
            logits_enc = self.C(h_tu, reverse=False)
            loss_enc = self._entropy_from_logits(logits_enc) * self.lambda_ent_enc
            total_loss = total_loss + loss_enc
            self.log("train/loss_enc_ent", loss_enc, on_step=True, on_epoch=True, prog_bar=False)

            # 3c) Feature-level FGM on unlabeled target (VAT-lite)
            if self.lambda_fgm > 0 and self.fgm_eps > 0:
                # compute gradient of entropy wrt features (stop graph to avoid higher-order grads)
                h_for_delta = h_tu.detach().requires_grad_(True)
                logits_for_delta = self.C(h_for_delta, reverse=False)
                ent_for_delta = self._entropy_from_logits(logits_for_delta)
                g = torch.autograd.grad(ent_for_delta, h_for_delta, retain_graph=False, create_graph=False)[0]
                if g is not None:
                    delta = self.fgm_eps * self._l2_normalize(g)
                    h_adv = h_tu + delta.detach()  # keep gradient path via encoder+predictor
                    logits_adv = self.C(h_adv, reverse=False)
                    loss_fgm = self._entropy_from_logits(logits_adv) * self.lambda_fgm
                    total_loss = total_loss + loss_fgm
                    self.log("train/loss_fgm_ent", loss_fgm, on_step=True, on_epoch=True, prog_bar=False)

        self.log("train/loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    # ----------------------- validation step -----------------------
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        z, _ = self._forward_encoder(x)
        h = self.P(z)
        logits = self.C(h, reverse=False)
        loss = self.ce(logits, y.long())
        preds = logits.argmax(1)
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    # ----------------------- test step (optional) -----------------------
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        z, _ = self._forward_encoder(x)
        h = self.P(z)
        logits = self.C(h, reverse=False)
        loss = self.ce(logits, y.long())
        preds = logits.argmax(1)
        acc = (preds == y).float().mean()
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)
        return {"test_loss": loss, "test_acc": acc}

    # ----------------------- optimizer config -----------------------
    def configure_optimizers(self):
        # separate lrs for encoder and predictor, as in many DA setups
        params = [
            {"params": itertools.chain(self.F.parameters(), self.P.parameters()), "lr": self.lr_f, "weight_decay": self.weight_decay},
            {"params": self.C.parameters(), "lr": self.lr_c, "weight_decay": self.weight_decay},
        ]
        opt = torch.optim.Adam(params)
        return opt
