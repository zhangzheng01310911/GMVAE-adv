from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def fit_gmm_em(
    embeddings: Tensor,
    num_components: int,
    iterations: int = 20,
    variance_floor: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fit a diagonal-covariance GMM with deterministic quantile initialization."""
    x = embeddings.reshape(-1, embeddings.size(-1))
    if x.size(0) < num_components:
        raise ValueError("number of embeddings must be >= number of components")

    order = torch.argsort(x[:, 0])
    indices = torch.linspace(0, x.size(0) - 1, num_components, device=x.device).long()
    means = x[order[indices]].clone()
    variances = x.var(dim=0, unbiased=False).clamp_min(variance_floor).repeat(num_components, 1)
    weights = torch.full((num_components,), 1.0 / num_components, device=x.device)

    for _ in range(iterations):
        diff = x[:, None, :] - means[None, :, :]
        log_prob = -0.5 * (
            (diff.square() / variances[None, :, :]).sum(-1)
            + torch.log(variances).sum(-1)[None, :]
            + x.size(-1) * torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
        )
        responsibilities = torch.softmax(
            log_prob + torch.log(weights.clamp_min(1e-8))[None, :], dim=1
        )
        counts = responsibilities.sum(0).clamp_min(1e-6)
        weights = counts / counts.sum()
        means = responsibilities.T @ x / counts[:, None]
        centered = x[:, None, :] - means[None, :, :]
        variances = (
            responsibilities[:, :, None] * centered.square()
        ).sum(0) / counts[:, None]
        variances.clamp_min_(variance_floor)

    return weights, means, variances, responsibilities
