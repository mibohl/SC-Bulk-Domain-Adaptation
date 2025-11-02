import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import Callback

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_utils import create_dataloader

class UMAPPlotCallback(Callback):
    def __init__(self, umap_save_path, every_n_epochs=5):
        super().__init__()
        self.umap_save_path = umap_save_path
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0 and epoch != 0:
            return
        
        print(f"Generating UMAP plot for epoch {epoch + 1}...")
        
        # Get dataloader from datamodule
        source_val_dataloader = trainer.datamodule.val_dataloader()
        target_val_loader = create_dataloader(trainer.datamodule.x_val_target, trainer.datamodule.y_val_target, trainer.datamodule.hparams.mbT, shuffle=False)
        
        # Get latent representations
        source_latents, target_latents, source_labels, target_labels = self._get_latent_representations(pl_module, source_val_dataloader, target_val_loader)

        if source_latents is None:
            print(f"Warning: Skipping UMAP plot for epoch {epoch + 1}. No data.")
            return

        try:
            reducer = umap.UMAP(random_state=42)
            embedding = reducer.fit_transform(np.concatenate([source_latents, target_latents]))

            plt.figure(figsize=(10, 8))

            # Plot source data
            source_mask_0 = source_labels == 0
            source_mask_1 = source_labels == 1
            if np.any(source_mask_0):
                plt.scatter(embedding[:len(source_latents)][source_mask_0, 0], embedding[:len(source_latents)][source_mask_0, 1], label="Source 0", c="blue", alpha=0.5)
            if np.any(source_mask_1):
                plt.scatter(embedding[:len(source_latents)][source_mask_1, 0], embedding[:len(source_latents)][source_mask_1, 1], label="Source 1", c="cyan", alpha=0.5)

            # Plot target data
            target_mask_0 = target_labels == 0
            target_mask_1 = target_labels == 1
            if np.any(target_mask_0):
                plt.scatter(embedding[len(source_latents):][target_mask_0, 0], embedding[len(source_latents):][target_mask_0, 1], label="Target 0", c="red", alpha=0.5)
            if np.any(target_mask_1):
                plt.scatter(embedding[len(source_latents):][target_mask_1, 0], embedding[len(source_latents):][target_mask_1, 1], label="Target 1", c="orange", alpha=0.5)

            # Plot unlabeled target data
            target_mask_unlabeled = target_labels == -1
            if np.any(target_mask_unlabeled):
                plt.scatter(embedding[len(source_latents):][target_mask_unlabeled, 0], embedding[len(source_latents):][target_mask_unlabeled, 1], label="Target Unlabeled", c="gray", alpha=0.5)

            plt.title(f"UMAP of Latent Space - Epoch {epoch + 1}")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.legend()

            os.makedirs(self.umap_save_path, exist_ok=True)
            save_path = os.path.join(self.umap_save_path, f"umap_epoch_{epoch + 1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"UMAP plot saved to {save_path}")

        except Exception as e:
            print(f"Error generating UMAP plot for epoch {epoch + 1}: {e}")
            plt.close()

    def _get_latent_representations(self, pl_module, source_dataloader, target_dataloader):
        pl_module.eval()
        source_latents, target_latents = [], []
        source_labels, target_labels = [], []
        
        with torch.no_grad():
            for xs, ys in source_dataloader:
                xs = xs.to(pl_module.device)
                F_xs = pl_module.encoder(xs)
                source_latents.append(F_xs.cpu().numpy())
                source_labels.append(ys.cpu().numpy())
            
            for xt, yt in target_dataloader:
                xt = xt.to(pl_module.device)
                F_xt = pl_module.encoder(xt)
                target_latents.append(F_xt.cpu().numpy())
                target_labels.append(yt.cpu().numpy())
        
        if not source_latents:
            return None, None, None, None

        return (
            np.concatenate(source_latents),
            np.concatenate(target_latents),
            np.concatenate(source_labels),
            np.concatenate(target_labels)
        )
