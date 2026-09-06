# GMVAE Traffic Forecasting — Auxiliary-Joint Rényi Version

This folder implements the model described in the latest revised Method.

## Model contract

For each node and input window, the GRU–GCN encoder produces

```text
q(z_i | X, A) = Normal(mu_i^q, diag((sigma_i^q)^2)).
```

Training uses the reparameterization

```text
z_i = mu_i^q + sigma_i^q * epsilon_i,  epsilon_i ~ Normal(0, I).
```

The decoder predicts a residual relative to persistence and reconstructs the
temporal hidden state. The implemented reconstruction term is the positive MSE

```text
L_rec = mean((H_hat - H)^2).
```

The global prior is

```text
p(z_i) = sum_k pi_k^p Normal(mu_k^p, diag((sigma_k^p)^2)).
```

After a warm-up stage, EM is run **once** on posterior means from the training
split. The resulting weights, means, and diagonal variances are fixed during
subsequent gradient-based fine-tuning. Validation and test data are never used
to estimate the prior.

The Rényi term is the analytic auxiliary-joint divergence

```text
R_i = logsumexp_k[
    alpha log r_ik
    + (1-alpha) log pi_k^p
    + (alpha-1) D_alpha(q_i || p_k)
] / (alpha-1).
```

Here `r_ik` is computed analytically from the posterior and fixed prior. The
component Gaussian divergence is closed form. This implementation does not use
a Monte Carlo density-ratio estimator for the Rényi term. By the data-processing
inequality, `R_i` upper-bounds the Rényi divergence to the marginal GMM prior.

The training loss is

```text
L = L_task + lambda_rec L_rec + beta(epoch) R_alpha,
```

where regression uses MAE and congestion classification uses binary
cross-entropy. The stochastic decoded losses use one reparameterized latent
sample per forward pass; the Rényi regularizer itself is analytic.

## Installation and tests

```bash
cd GMVAE-revision-aux-renyi
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Synthetic integration check

```bash
python scripts/make_synthetic_data.py
PYTHONPATH=src python scripts/train.py \
  --data work/synthetic_traffic.npz \
  --output work/synthetic_run \
  --epochs 8 --input-length 12 --horizon 3 \
  --em-warmup 2 --em-variance-floor 0.5
```

The synthetic run verifies software integration only; it is not a manuscript
experiment.

## Data contract

Input NPZ files contain:

```text
values:    float32 [time, nodes, channels]
adjacency: float32 [nodes, nodes]
timestamps (optional)
```

Splits are chronological 70/20/10. Missing values and normalization statistics
are fitted using training data only.

## Numerical condition

For `alpha > 1`, the closed-form Gaussian Rényi divergence is finite only when

```text
alpha / q_variance + (1-alpha) / p_variance > 0
```

in every latent dimension. The encoder caps posterior variance at one and EM
uses a configurable prior-variance floor (default `0.5`), which satisfies this
condition for the default `alpha=1.75`. The implementation raises an explicit
error rather than silently clipping an infinite divergence.
