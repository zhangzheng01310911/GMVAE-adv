from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def gaussian_renyi_divergence_diag(
    q_mu: Tensor,
    q_logvar: Tensor,
    p_mu: Tensor,
    p_logvar: Tensor,
    alpha: float,
) -> Tensor:
    """D_alpha(q||p) for diagonal Gaussians, summed across latent dimensions.

    The expression is evaluated through the Gaussian power integral and is
    valid only when alpha * Sigma_q^-1 + (1-alpha) * Sigma_p^-1 is positive.
    Alpha=1 is intentionally rejected; use KL divergence for that limit.
    """
    if alpha <= 0 or abs(alpha - 1.0) < 1e-6:
        raise ValueError("alpha must be positive and different from 1")
    q_var, p_var = q_logvar.exp(), p_logvar.exp()
    precision = alpha / q_var + (1.0 - alpha) / p_var
    if torch.any(precision <= 0):
        raise ValueError("invalid alpha/covariance combination for finite Renyi divergence")
    natural = alpha * q_mu / q_var + (1.0 - alpha) * p_mu / p_var
    constant = alpha * q_mu.square() / q_var + (1.0 - alpha) * p_mu.square() / p_var
    log_integral = -0.5 * (
        alpha * q_logvar
        + (1.0 - alpha) * p_logvar
        + torch.log(precision)
        + constant
        - natural.square() / precision
    )
    return log_integral.sum(-1) / (alpha - 1.0)


def posterior_gmm_responsibilities(
    q_mu: Tensor,
    q_logvar: Tensor,
    prior_weights: Tensor,
    prior_means: Tensor,
    prior_logvars: Tensor,
) -> Tensor:
    """Analytic q(c=k|x,A) from E_q[log pi_k p_k(z)]."""
    q_var = q_logvar.exp()[..., None, :]
    p_var = prior_logvars.exp()
    squared_offset = (q_mu[..., None, :] - prior_means).square()
    expected_log_component = -0.5 * (
        torch.log(2.0 * torch.pi * p_var)
        + (squared_offset + q_var) / p_var
    ).sum(dim=-1)
    logits = expected_log_component + torch.log(prior_weights.clamp_min(1e-8))
    return torch.softmax(logits, dim=-1)


def auxiliary_joint_renyi(
    q_mu: Tensor,
    q_logvar: Tensor,
    responsibilities: Tensor,
    prior_weights: Tensor,
    prior_means: Tensor,
    prior_logvars: Tensor,
    alpha: float,
) -> Tensor:
    """Exact auxiliary-joint Renyi divergence, averaged over samples/nodes.

    This is analytic: no latent samples or Monte Carlo density-ratio estimate
    are used. It upper-bounds the marginal-GMM Renyi divergence by DPI.
    """
    component_divergence = gaussian_renyi_divergence_diag(
        q_mu[..., None, :],
        q_logvar[..., None, :],
        prior_means,
        prior_logvars,
        alpha,
    )
    log_terms = (
        alpha * torch.log(responsibilities.clamp_min(1e-8))
        + (1.0 - alpha) * torch.log(prior_weights.clamp_min(1e-8))
        + (alpha - 1.0) * component_divergence
    )
    return (torch.logsumexp(log_terms, dim=-1) / (alpha - 1.0)).mean()


def total_loss(
    output,
    target: Tensor,
    prior_weights: Tensor,
    prior_means: Tensor,
    prior_logvars: Tensor,
    alpha: float,
    task: str = "regression",
    forecast_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    renyi_weight: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    if task == "regression":
        forecast = F.l1_loss(output.prediction, target)
    elif task == "classification":
        forecast = F.binary_cross_entropy_with_logits(output.prediction, target)
    else:
        raise ValueError(f"unsupported task: {task}")
    reconstruction = F.mse_loss(output.reconstructed_hidden, output.hidden)
    if renyi_weight == 0.0:
        renyi = output.mu.new_zeros(())
    else:
        responsibilities = posterior_gmm_responsibilities(
            output.mu,
            output.logvar,
            prior_weights,
            prior_means,
            prior_logvars,
        )
        renyi = auxiliary_joint_renyi(
            output.mu,
            output.logvar,
            responsibilities,
            prior_weights,
            prior_means,
            prior_logvars,
            alpha,
        )
    loss = forecast_weight * forecast + reconstruction_weight * reconstruction + renyi_weight * renyi
    return loss, {"forecast": forecast, "reconstruction": reconstruction, "renyi": renyi}
