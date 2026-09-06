#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gmvae.data import WindowDataset, chronological_split, fit_zscore, impute_and_normalize, load_npz
from gmvae.em import fit_gmm_em
from gmvae.losses import total_loss
from gmvae.metrics import regression_metrics
from gmvae.model import EMGMVAE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def collect_mu(model, loader, adjacency, device, max_points):
    model.eval()
    chunks = []
    collected = 0
    with torch.no_grad():
        for x, _ in loader:
            flat = model(x.to(device), adjacency).mu.detach().reshape(-1, model.mixture_means.size(1)).cpu()
            remaining = max_points - collected
            chunks.append(flat[:remaining])
            collected += min(len(flat), remaining)
            if collected >= max_points:
                break
    return torch.cat(chunks).to(device)


def add_persistence_residual(output, x, target_channel):
    last_value = x[:, -1, :, target_channel:target_channel + 1]
    output.prediction = output.prediction + last_value[:, None, :, :]
    return output


def evaluate(model, loader, adjacency, device, target_channel):
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for x, y in loader:
            x_device = x.to(device)
            output = add_persistence_residual(model(x_device, adjacency), x_device, target_channel)
            truth.append(y.numpy())
            predicted.append(output.prediction.cpu().numpy())
    return np.concatenate(truth), np.concatenate(predicted)


def inverse_channel(values, mean, std, channel):
    channel_mean = float(mean.reshape(-1)[channel])
    channel_std = float(std.reshape(-1)[channel])
    return values * channel_std + channel_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the revised EM-GMVAE traffic forecaster")
    parser.add_argument("--data", required=True, help="NPZ containing values and adjacency")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--em-warmup", type=int, default=5)
    parser.add_argument("--em-max-points", type=int, default=100000)
    parser.add_argument("--em-variance-floor", type=float, default=0.5)
    parser.add_argument("--target-channel", type=int, default=0, help="0=speed, 1=volume, 2=occupancy")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--forecast-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--renyi-weight", type=float, default=1e-5)
    parser.add_argument("--renyi-ramp-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    if args.alpha > 1.0:
        required_floor = (args.alpha - 1.0) / args.alpha
        if args.em_variance_floor <= required_floor:
            raise ValueError(
                "for alpha > 1, --em-variance-floor must exceed "
                f"(alpha-1)/alpha={required_floor:.6g} because posterior variance is capped at 1"
            )

    seed_everything(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_npz(args.data)
    train_raw, val_raw, test_raw = chronological_split(arrays.values)
    mean, std = fit_zscore(train_raw)
    train = impute_and_normalize(train_raw, mean, std)
    val = impute_and_normalize(val_raw, mean, std)
    test = impute_and_normalize(test_raw, mean, std)
    loaders = [
        DataLoader(
            WindowDataset(part, args.input_length, args.horizon, args.target_channel),
            batch_size=args.batch_size,
            shuffle=(index == 0),
        )
        for index, part in enumerate((train, val, test))
    ]
    if any(len(loader.dataset) == 0 for loader in loaders):
        raise ValueError("each chronological split must contain at least input_length+horizon samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency = torch.as_tensor(arrays.adjacency, dtype=torch.float32, device=device)
    channels = arrays.values.shape[-1]
    if not 0 <= args.target_channel < channels:
        raise ValueError(f"target channel {args.target_channel} outside 0..{channels - 1}")
    model = EMGMVAE(channels, args.hidden, args.latent, args.horizon, 1, args.components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_val_mae = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        # Estimate the global GMM exactly once, then keep it fixed.
        if epoch == args.em_warmup:
            embeddings = collect_mu(model, loaders[0], adjacency, device, args.em_max_points)
            mixture = fit_gmm_em(
                embeddings,
                args.components,
                iterations=20,
                variance_floor=args.em_variance_floor,
            )
            model.set_mixture(*mixture[:3])
        model.train()
        totals = []
        for x, y in loaders[0]:
            optimizer.zero_grad()
            x_device = x.to(device)
            result = add_persistence_residual(model(x_device, adjacency), x_device, args.target_channel)
            if epoch < args.em_warmup:
                effective_renyi_weight = 0.0
            else:
                ramp = min(1.0, (epoch - args.em_warmup + 1) / max(args.renyi_ramp_epochs, 1))
                effective_renyi_weight = args.renyi_weight * ramp
            loss, parts = total_loss(
                result,
                y.to(device),
                model.mixture_weights,
                model.mixture_means,
                model.mixture_logvars,
                alpha=args.alpha,
                forecast_weight=args.forecast_weight,
                reconstruction_weight=args.reconstruction_weight,
                renyi_weight=effective_renyi_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}; try a smaller --renyi-weight"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals.append([loss.item(), *(value.item() for value in parts.values())])
        averages = np.mean(totals, axis=0)
        val_y_norm, val_p_norm = evaluate(model, loaders[1], adjacency, device, args.target_channel)
        val_y = inverse_channel(val_y_norm, mean, std, args.target_channel)
        val_p = inverse_channel(val_p_norm, mean, std, args.target_channel)
        val_metrics = regression_metrics(val_y, val_p)
        row = {
            "epoch": epoch,
            "loss": float(averages[0]),
            "train_forecast": float(averages[1]),
            "train_reconstruction": float(averages[2]),
            "train_renyi": float(averages[3]),
            "effective_renyi_weight": effective_renyi_weight,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation MAE={best_val_mae:.6f}")
            break

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    test_y_norm, test_p_norm = evaluate(model, loaders[2], adjacency, device, args.target_channel)
    test_y = inverse_channel(test_y_norm, mean, std, args.target_channel)
    test_p = inverse_channel(test_p_norm, mean, std, args.target_channel)
    metrics = regression_metrics(test_y, test_p)
    persistence_norm = np.stack([
        loaders[2].dataset[index][0][-1, :, args.target_channel].numpy()
        for index in range(len(loaders[2].dataset))
    ])[:, None, :, None]
    persistence_norm = np.repeat(persistence_norm, args.horizon, axis=1)
    persistence = inverse_channel(persistence_norm, mean, std, args.target_channel)
    baseline_metrics = regression_metrics(test_y, persistence)
    metrics["persistence_mae"] = baseline_metrics["mae"]
    metrics["persistence_rmse"] = baseline_metrics["rmse"]
    metrics["mae_improvement_over_persistence_percent"] = (
        100.0 * (baseline_metrics["mae"] - metrics["mae"]) / max(baseline_metrics["mae"], 1e-8)
    )
    metrics["best_val_mae"] = best_val_mae
    metrics["epochs_completed"] = len(history)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": vars(args)}, output_dir / "model.pt")
    np.savez_compressed(output_dir / "predictions.npz", target=test_y, prediction=test_p, persistence=persistence)
    with (output_dir / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    with (output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
