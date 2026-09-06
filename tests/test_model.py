import torch

from gmvae.em import fit_gmm_em
from gmvae.losses import (
    auxiliary_joint_renyi,
    gaussian_renyi_divergence_diag,
    posterior_gmm_responsibilities,
    total_loss,
)
from gmvae.model import EMGMVAE


def test_forward_loss_and_em_are_finite():
    torch.manual_seed(7)
    model = EMGMVAE(3, 8, 4, horizon=2, output_channels=3, num_components=3)
    x = torch.randn(2, 5, 6, 3)
    adjacency = torch.eye(6)
    target = torch.randn(2, 2, 6, 3)
    output = model(x, adjacency)
    weights, means, variances, em_responsibilities = fit_gmm_em(
        output.mu.detach(), 3, iterations=3
    )
    model.set_mixture(weights, means, variances)
    output = model(x, adjacency)
    loss, parts = total_loss(
        output,
        target,
        model.mixture_weights,
        model.mixture_means,
        model.mixture_logvars,
        alpha=1.75,
    )
    assert output.prediction.shape == target.shape
    assert output.responsibilities.shape[-1] == 3
    assert em_responsibilities.shape[-1] == 3
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_renyi_is_zero_for_identical_gaussians():
    mu = torch.zeros(2, 3)
    logvar = torch.zeros_like(mu)
    result = gaussian_renyi_divergence_diag(mu, logvar, mu, logvar, alpha=1.75)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


def test_auxiliary_renyi_is_zero_for_matching_single_gaussian():
    mu = torch.zeros(2, 2)
    logvar = torch.zeros_like(mu)
    weights = torch.ones(1)
    means = torch.zeros(1, 2)
    prior_logvars = torch.zeros(1, 2)
    responsibilities = posterior_gmm_responsibilities(
        mu, logvar, weights, means, prior_logvars
    )
    result = auxiliary_joint_renyi(
        mu, logvar, responsibilities, weights, means, prior_logvars, 1.75
    )
    assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)


def test_responsibilities_are_probabilities():
    mu = torch.randn(2, 5, 3)
    logvar = torch.full_like(mu, -0.5)
    weights = torch.tensor([0.25, 0.75])
    means = torch.zeros(2, 3)
    prior_logvars = torch.zeros(2, 3)
    responsibilities = posterior_gmm_responsibilities(
        mu, logvar, weights, means, prior_logvars
    )
    assert responsibilities.shape == (2, 5, 2)
    assert torch.allclose(
        responsibilities.sum(dim=-1),
        torch.ones_like(responsibilities[..., 0]),
        atol=1e-6,
    )
