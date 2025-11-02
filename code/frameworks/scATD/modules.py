"""
This file contains the core neural network modules for the scATD framework.
- Swish: A custom activation function.
- ContinuousResidualVAE: The VAE with residual blocks used as the main encoder-decoder.
- VAEclassification: A wrapper that combines the VAE encoder with a classifier head.
- mmd_loss: The Maximum Mean Discrepancy loss function for domain alignment.
"""

import torch
from torch import nn
import torch.nn.init as init
from torch.nn.utils import spectral_norm

class Swish(nn.Module):
    """Swish activation function with optional trainable beta parameter."""
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))
        else:
            self.beta = initial_beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class ContinuousResidualVAE(nn.Module):
    """
    A Variational Autoencoder with continuous residual blocks.
    This architecture is used as the backbone for both pre-training and fine-tuning.
    """
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            # Match Dist-VAE checkpoint: encoder blocks use plain Linear layers
            self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.swish = Swish(trainable_beta=True)
            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')

            if in_dim != out_dim:
                self.downsample = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = self.swish(self.bn(self.fc(x)))
            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, input_dim, hidden_dim_layer0, Encoder_layer_dims, Decoder_layer_dims, hidden_dim_layer_out_Z, z_dim, loss_type='MSE', reduction='mean'):
        super().__init__()
        self.Encoder_resblocks = nn.ModuleList()
        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))

        self.fc21 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim), n_power_iterations=5)
        init.xavier_normal_(self.fc21.weight)
        self.fc22 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim), n_power_iterations=5)
        init.xavier_normal_(self.fc22.weight)

        self.Decoder_resblocks = nn.ModuleList()
        for i in range(len(Decoder_layer_dims) - 1):
            self.Decoder_resblocks.append(self.ResBlock(Decoder_layer_dims[i], Decoder_layer_dims[i + 1]))

        self.fc4 = spectral_norm(nn.Linear(hidden_dim_layer0, input_dim), n_power_iterations=5)
        init.xavier_normal_(self.fc4.weight)

        self.loss_type = loss_type
        self.reduction = reduction
        if reduction not in ['mean', 'sum']:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'sum'.")

    def encode(self, x):
        h = x
        for block in self.Encoder_resblocks:
            h = block(h)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = z
        for block in self.Decoder_resblocks:
            h = block(h)
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def load_adapted_state_dict(self, pretrained_state: dict, strict: bool = False):
        """Adapt and load a pretrained state dict, handling input dimension mismatches."""
        current_state = self.state_dict()
        model_device = next(self.parameters()).device
        adapted_state: dict = {}

        for key, value in pretrained_state.items():
            if key not in current_state:
                continue

            # Ensure the tensor from the loaded state is on the correct device
            value = value.to(model_device)

            current_value = current_state[key]
            if value.shape == current_value.shape:
                adapted_state[key] = value
                continue

            def maybe_pad(tensor: torch.Tensor, dim: int, target: int) -> torch.Tensor:
                if tensor.size(dim) >= target:
                    slices = [slice(None)] * tensor.ndim
                    slices[dim] = slice(0, target)
                    return tensor[tuple(slices)]
                pad_shape = list(tensor.shape)
                pad_shape[dim] = target - tensor.size(dim)
                padding = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
                return torch.cat([tensor, padding], dim=dim)
            
            # Logic to adapt weights based on key and shape
            if "Encoder_resblocks.0.downsample.weight" in key and value.ndim == 2:
                adapted_state[key] = maybe_pad(value, 1, current_value.shape[1])
            elif "Encoder_resblocks.0.downsample.weight_v" in key and value.ndim == 1:
                adapted_state[key] = maybe_pad(value, 0, current_value.shape[0])
            elif "fc4.weight" in key and value.ndim == 2:
                adapted_state[key] = maybe_pad(value, 0, current_value.shape[0])
            elif ("fc4.weight_u" in key or "fc4.bias" in key or "fc4.weight_orig" in key) and value.ndim == 1:
                adapted_state[key] = maybe_pad(value, 0, current_value.shape[0])
            else:
                tensor = value
                for dim in range(value.ndim):
                    if value.shape[dim] != current_value.shape[dim]:
                        tensor = maybe_pad(tensor, dim, current_value.shape[dim])
                adapted_state[key] = tensor

        self.load_state_dict(adapted_state, strict=strict)

class VAEclassification(nn.Module):
    """
    Classifier model that uses the encoder from a pre-trained ContinuousResidualVAE.
    """
    def __init__(self, model_pretraining, z_dim, class_num):
        super(VAEclassification, self).__init__()
        self.Encoder_resblocks = model_pretraining.Encoder_resblocks
        self.fc21 = model_pretraining.fc21
        self.fc22 = model_pretraining.fc22
        self.fc3 = spectral_norm(nn.Linear(z_dim, z_dim // 2), n_power_iterations=5)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        self.bn3 = nn.BatchNorm1d(z_dim // 2)
        self.swish = Swish(trainable_beta=True)
        self.fc4 = spectral_norm(nn.Linear(z_dim // 2, class_num), n_power_iterations=5)
        init.xavier_normal_(self.fc4.weight)

    def encode(self, x):
        h = x
        for block in self.Encoder_resblocks:
            h = block(h)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classifier(self, x):
        h = self.fc3(x)
        h = self.bn3(h)
        h = self.swish(h)
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.classifier(z)

    def forward_deterministic(self, x):
        """Run the classifier on the posterior mean to avoid stochastic sampling."""
        mu, _ = self.encode(x.view(-1, x.shape[1]))
        return self.classifier(mu)

    def get_embeddings(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return z

def compute_gamma(x, y):
    """
    Computes the gamma parameter for the RBF kernel based on the median
    pairwise distance of the combined data. This matches the original
    author's implementation.
    """
    combined = torch.cat([x, y], dim=0)
    pairwise_dists = torch.cdist(combined, combined, p=2)
    # Get the upper triangular part of the distance matrix to avoid duplicates
    upper_tri = pairwise_dists[torch.triu(torch.ones_like(pairwise_dists), diagonal=1) == 1]
    median_dist = torch.median(upper_tri)
    
    # Prevent division by zero
    if median_dist < 1e-9:
        return 1.0
    
    return 1.0 / (2 * (median_dist ** 2))


def mmd_loss(x, y):
    """
    Computes the Maximum Mean Discrepancy (MMD) loss with a single RBF kernel.
    This implementation is aligned with the original author's code.
    """
    gamma = compute_gamma(x, y)
    
    # Compute kernel matrices
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = torch.diag(xx).unsqueeze(0).expand_as(xx)
    ry = torch.diag(yy).unsqueeze(0).expand_as(yy)

    K = torch.exp(-gamma * (rx.t() + rx - 2 * xx))
    L = torch.exp(-gamma * (ry.t() + ry - 2 * yy))
    P = torch.exp(-gamma * (rx.t() + ry - 2 * zz))

    # Compute biased MMD2
    beta = 1.0 / (x.size(0) * x.size(0))
    gamma_val = 1.0 / (y.size(0) * y.size(0))
    delta = 2.0 / (x.size(0) * y.size(0))
    
    return beta * torch.sum(K) + gamma_val * torch.sum(L) - delta * torch.sum(P)
