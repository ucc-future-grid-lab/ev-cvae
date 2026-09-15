"""CVAE Module."""

# Author: Graeme Kelly, Emilio J. Palacios-Garcia
# SPDX-License-Identifier: MIT

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CVAE(nn.Module):
    """CVAE with Gaussian NLL reconstruction.

    Notes:
     - Decoder predicts mean and log-variance per feature.
     - Gaussian NLL is computed analytically (no sampling from p(x|z,c) during training).
     - We clamp log-variance for numeric stability to prevent exploding/vanishing variance.
     - Condition cis concatenated to inputs of both encoder and decoder.
     - At generation time we can use the mean (deterministic) or sample with the predicted variance.
    """

    def __init__(
        self,
        feature_size: int,
        latent_size: int,
        class_size: int,
        hidden_size: int,
        logvar_min: float = -6.0,
        logvar_max: float = 1.5,
    ):
        """Create CVAE model.

        Args:
            feature_size: size of continuous features to reconstruct (excluding condition)
            latent_size: size of latent vector z
            class_size: size of condition vector
            hidden_size: size of hidden layers in encoder and decoder
            logvar_min: minimum log-variance for decoder output (clamped)
            logvar_max: maximum log-variance for decoder output (clamped)
        """
        super().__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # Encoder q(z|x, c)
        self.enc_fc1 = nn.Linear(feature_size + class_size, hidden_size)
        self.enc_fc2 = nn.Linear(hidden_size, hidden_size)
        self.enc_mu = nn.Linear(hidden_size, latent_size)
        self.enc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder p(x|z, c) — outputs per-feature mean and log-variance
        self.dec_fc1 = nn.Linear(latent_size + class_size, hidden_size)
        self.dec_fc2 = nn.Linear(hidden_size, hidden_size)
        self.dec_mu = nn.Linear(hidden_size, feature_size)
        self.dec_logvar = nn.Linear(hidden_size, feature_size)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder q(z|x,c) that outputs mean and log-variance of latent z.

        Args:
            x: [B, F] input features
            c: [B, C] condition vector

        Returns:
            mu: [B, latent_size] mean of latent z
            logvar: [B, latent_size] log-variance of latent z
        """
        h = F.relu(self.enc_fc1(torch.cat([x, c], dim=-1)))
        h = F.relu(self.enc_fc2(h))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, T: float = 1.0
    ) -> torch.Tensor:
        """Latent reparameterization trick for sampling z ~ q(z|x,c) with optional temperature T.

        Args:
            mu: [B, latent_size] mean of latent z
            logvar: [B, latent_size] log-variance of latent z
            T: temperature for sampling; T=0 gives deterministic mean, T>0 adds noise

        Returns:
            z: [B, latent_size] sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (std * eps * T)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoder p(x|z,c) that outputs mean and log-variance of reconstructed x.

        Args:
            z: [B, latent_size] latent vector
            c: [B, C] condition vector

        Returns:
            x_mu: [B, F] mean of reconstructed x
            x_logvar: [B, F] log-variance of reconstructed x
        """
        h = F.relu(self.dec_fc1(torch.cat([z, c], dim=-1)))
        h = F.relu(self.dec_fc2(h))
        x_mu = self.dec_mu(h)
        x_logvar = self.dec_logvar(h)
        # clamp for stability;
        # prevents the model from setting huge variance that makes outputs look random
        x_logvar = torch.clamp(x_logvar, min=self.logvar_min, max=self.logvar_max)
        return x_mu, x_logvar

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, T: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the CVAE: encode, reparameterize, decode.

        Args:
            x: [B, F] input features
            c: [B, C] condition vector
            T: temperature for reparameterization

        Returns:
            x_mu: [B, F] mean of reconstructed x
            x_logvar: [B, F] log-variance of reconstructed x
            z_mu: [B, latent_size] mean of latent z
            z_logvar: [B, latent_size] log-variance of latent z
        """
        z_mu, z_logvar = self.encode(x, c)
        z = self.reparameterize(z_mu, z_logvar, T=T)
        x_mu, x_logvar = self.decode(z, c)
        return x_mu, x_logvar, z_mu, z_logvar

    @staticmethod
    def gaussian_nll(
        x_mu: torch.Tensor, x_logvar: torch.Tensor, x: torch.Tensor, reduce: str = "sum"
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood per feature.

        Args:
            x_mu: [B, F] mean of reconstructed x
            x_logvar: [B, F] log-variance of reconstructed x
            x: [B, F] original input features
            reduce: "sum", "mean", or "none" to control output shape

        Returns:
            - scalar if reduce in {"sum", "mean"}
            - [B, F] if reduce == "none" (or anything else)
        """
        LOG_2PI = math.log(2.0 * math.pi)

        nll = 0.5 * ((x - x_mu) ** 2 * torch.exp(-x_logvar) + x_logvar + LOG_2PI)

        if reduce == "sum":
            return nll.sum()
        elif reduce == "mean":
            return nll.mean()
        else:
            return nll

    @staticmethod
    def kld_standard_normal(
        z_mu: torch.Tensor, z_logvar: torch.Tensor, reduce: str = "sum"
    ) -> torch.Tensor:
        """Latent KL divergence between q(z|x,c) and standard normal N(0,I).

        Args:
            z_mu: [B, latent_size] mean of latent z
            z_logvar: [B, latent_size] log-variance of latent z
            reduce: "sum", "mean", or "none" to control output shape

        Returns:
            - scalar if reduce in {"sum", "mean"}
            - [B, latent_size] if reduce == "none" (or anything else)
        """
        kld = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        if reduce == "sum":
            return kld.sum()
        elif reduce == "mean":
            return kld.mean()
        else:
            return kld

    def cvae_loss(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        beta: float = 0.075,
        reduce: str = "sum",
        T: float = 1.0,
        feature_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """CVAE loss with optional per-feature weighting.

        Args:
            x: [B, F] input features
            c: [B, C] condition vector
            beta: weight for KL divergence term
            reduce: "sum", "mean", or "none" to control output shape
            T: temperature for reparameterization
            feature_weights: [F] optional weights for per-feature NLL in the same order
                as input feature x. If None, all features are equally weighted.

        Returns:
            loss: scalar if reduce in {"sum", "mean"}, else [B, F]
            metrics: dict with keys "recon" and "kld" containing detached values
        """
        x_mu, x_logvar, z_mu, z_logvar = self.forward(x, c, T=T)

        # [B, F] per-sample, per-feature NLL
        nll_bf = self.gaussian_nll(x_mu, x_logvar, x, reduce="none")

        if feature_weights is not None:
            # reshape to [1, F] for broadcasting over batch
            w = feature_weights.view(1, -1)
            nll_bf = nll_bf * w

        # now reduce according to `reduce`
        if reduce == "sum":
            recon = nll_bf.sum()
        elif reduce == "mean":
            recon = nll_bf.mean()
        else:
            recon = nll_bf  # (B, F)

        kld = self.kld_standard_normal(z_mu, z_logvar, reduce=reduce)
        return recon + beta * kld, {"recon": recon.detach(), "kld": kld.detach()}

    def per_feature_nll_over_loader(
        self, data_loader: torch.utils.data.DataLoader, device: torch.device
    ) -> torch.Tensor:
        """Average Gaussian NLL per feature over an entire dataloader.

        Args:
            data_loader: PyTorch DataLoader yielding (x, c) batches
            device: torch.device to run computations on

        Returns:
            feature_nll: [F] average NLL per feature over the entire dataset
        """
        self.eval()
        feature_sums = torch.zeros(self.feature_size, device=device)
        n_samples = 0

        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device)
                x_mu, x_logvar, z_mu, z_logvar = self(xb, yb)  # forward
                # [B, F] NLL per sample & feature
                nll_bf = self.gaussian_nll(x_mu, x_logvar, xb, reduce="none")
                feature_sums += nll_bf.sum(dim=0)  # sum over batch, keep features
                n_samples += xb.size(0)

        return (feature_sums / n_samples).cpu()  # [F]

    def generate_samples(
        self,
        n_samples: int,
        c_samples: np.ndarray,
        device: torch.device,
        T: float = 1.0,
    ) -> np.ndarray:
        """Generate samples from the CVAE given condition vectors.

        Args:
            n_samples: number of samples to generate
            c_samples: [n_samples, C] condition vectors for generation
            device: torch.device to run computations on
            T: temperature for sampling; T=0 gives deterministic mean, T>0 adds noise

        Returns:
            x_gen_scaled: [n_samples, F] generated samples in original feature scale
        """
        self.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_size, device=device)
            c = torch.from_numpy(c_samples).to(device)
            x_mu, x_logvar = self.decode(z, c)
            x_gen_scaled = self.reparameterize(x_mu, x_logvar, T)

            # x_std = torch.exp(0.5 * x_logvar)
            # x_gen_scaled = (x_mu + T * x_std * torch.randn_like(x_std)).cpu().numpy()
            return x_gen_scaled.cpu().numpy()
