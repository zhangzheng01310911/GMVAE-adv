from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


EPS = 1e-8


def normalize_adjacency(adjacency: Tensor) -> Tensor:
    """Return D^-1/2 (A + I) D^-1/2 for a dense adjacency matrix."""
    eye = torch.eye(adjacency.size(-1), device=adjacency.device, dtype=adjacency.dtype)
    a = adjacency + eye
    degree = a.sum(dim=-1).clamp_min(EPS)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


class GraphLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        return self.projection(torch.einsum("ij,bjf->bif", adjacency, x))


class TemporalGraphEncoder(nn.Module):
    """GRU temporal encoder followed by graph-conditioned Gaussian heads."""

    def __init__(self, input_channels: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden_dim, batch_first=True)
        self.shared = GraphLinear(hidden_dim, hidden_dim)
        self.mu = GraphLinear(hidden_dim, latent_dim)
        self.logvar = GraphLinear(hidden_dim, latent_dim)

    def forward(self, x: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # x: [batch, time, nodes, channels]
        batch, time, nodes, channels = x.shape
        sequences = x.permute(0, 2, 1, 3).reshape(batch * nodes, time, channels)
        _, hidden = self.gru(sequences)
        h = hidden[-1].reshape(batch, nodes, -1)
        h_graph = torch.relu(self.shared(h, adjacency))
        mu = self.mu(h_graph, adjacency)
        # Unit upper variance plus the EM variance floor keeps the alpha>1
        # Gaussian Renyi power integral finite for the default alpha=1.75.
        logvar = self.logvar(h_graph, adjacency).clamp(-10.0, 0.0)
        return h, mu, logvar


class ForecastDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, horizon: int, output_channels: int) -> None:
        super().__init__()
        self.graph = GraphLinear(latent_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, horizon * output_channels)
        self.horizon = horizon
        self.output_channels = output_channels
        # Combined with residual forecasting in train.py, zero initialization
        # makes the untrained model exactly equal to the persistence baseline.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        reconstructed_hidden = torch.relu(self.graph(z, adjacency))
        forecast = self.output(reconstructed_hidden)
        batch, nodes, _ = forecast.shape
        forecast = forecast.reshape(batch, nodes, self.horizon, self.output_channels)
        return forecast.permute(0, 2, 1, 3), reconstructed_hidden


@dataclass
class GMVAEOutput:
    prediction: Tensor
    hidden: Tensor
    reconstructed_hidden: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor
    responsibilities: Tensor


class EMGMVAE(nn.Module):
    """Gaussian-posterior VAE with a global diagonal GMM prior.

    The encoder produces q(z_i|x,A)=N(mu_i, diag(sigma_i^2)). The global
    mixture prior p(z)=sum_k pi_k N(m_k, diag(s_k^2)) is estimated once after
    warm-up by :func:`gmvae.em.fit_gmm_em` and then fixed. During fine-tuning,
    an analytic auxiliary-joint Renyi divergence regularizes the posterior.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        latent_dim: int,
        horizon: int,
        output_channels: int,
        num_components: int,
    ) -> None:
        super().__init__()
        self.encoder = TemporalGraphEncoder(input_channels, hidden_dim, latent_dim)
        self.decoder = ForecastDecoder(latent_dim, hidden_dim, horizon, output_channels)
        self.register_buffer("mixture_logits", torch.zeros(num_components))
        self.register_buffer("mixture_means", torch.zeros(num_components, latent_dim))
        self.register_buffer("mixture_logvars", torch.zeros(num_components, latent_dim))

    @property
    def mixture_weights(self) -> Tensor:
        return torch.softmax(self.mixture_logits, dim=0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def responsibilities(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Return analytic responsibilities from E_q[log pi_k p_k(z)]."""
        q_var = logvar.exp()[..., None, :]
        p_var = self.mixture_logvars.exp()
        squared_offset = (mu[..., None, :] - self.mixture_means).square()
        expected_log_component = -0.5 * (
            torch.log(2.0 * torch.pi * p_var)
            + (squared_offset + q_var) / p_var
        ).sum(dim=-1)
        logits = expected_log_component + torch.log(self.mixture_weights.clamp_min(EPS))
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def set_mixture(self, weights: Tensor, means: Tensor, variances: Tensor) -> None:
        if weights.shape != self.mixture_logits.shape:
            raise ValueError("mixture weight shape mismatch")
        if means.shape != self.mixture_means.shape or variances.shape != means.shape:
            raise ValueError("mixture parameter shape mismatch")
        self.mixture_logits.copy_(torch.log(weights.clamp_min(EPS)))
        self.mixture_means.copy_(means)
        self.mixture_logvars.copy_(torch.log(variances.clamp_min(1e-5)))

    def forward(self, x: Tensor, adjacency: Tensor) -> GMVAEOutput:
        normalized = normalize_adjacency(adjacency)
        hidden, mu, logvar = self.encoder(x, normalized)
        z = self.reparameterize(mu, logvar)
        prediction, reconstructed_hidden = self.decoder(z, normalized)
        resp = self.responsibilities(mu, logvar)
        return GMVAEOutput(
            prediction=prediction,
            hidden=hidden,
            reconstructed_hidden=reconstructed_hidden,
            mu=mu,
            logvar=logvar,
            z=z,
            responsibilities=resp,
        )
