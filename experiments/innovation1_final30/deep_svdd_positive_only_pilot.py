"""Positive-only Deep SVDD pilot for the current innovation1_final30 protocol.

Goal:
- Keep the existing CycleGAN frozen.
- Train the one-class head ONLY with the target user's genuine auth_train samples.
- Choose the operating radius using ONLY target genuine dev samples.
- Use final genuine/impostor only once for evaluation.
- Compare against a positive-only prototype-distance baseline under the same protocol.

This is an exploratory pilot on the existing preprocessed acc.npy. It does not fix
the known full-data preprocessing/statistics issue; final paper results should be
rerun from raw data with train-only preprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from modelb import create_multi_source_model
from target_auth_protocol_150_final30 import USER_NAMES, build_target_protocol, save_manifest, sha256_file

CHECKPOINT_ROOT = "/root/autodl-fs/cyclegan_forward_aligned_20260928"
DEFAULT_CHECKPOINTS = {
    1: f"{CHECKPOINT_ROOT}/user1/cyclegan/model/checkpoint_epoch_50.pth",
    4: f"{CHECKPOINT_ROOT}/user4/cyclegan/model/checkpoint_epoch_50.pth",
    5: f"{CHECKPOINT_ROOT}/user5/cyclegan/model/checkpoint_epoch_50.pth",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_features(model, x: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    out = {"shallow": [], "bottleneck": [], "absolute_cycle": [], "combined": []}
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(
            x[start:start + batch_size], dtype=torch.float32, device=device
        ).unsqueeze(1)
        shallow = model.shallow_agg(model.shallow_encoder(batch))
        bottleneck = model.bottleneck_agg(model.bottleneck_encoder(batch))
        residual, scalar = model._compute_cycle_diff(batch)
        cycle_emb = model.diff_agg(model.diff_encoder(residual))
        absolute_cycle = torch.cat((cycle_emb, scalar), dim=1)
        combined = torch.cat((bottleneck, absolute_cycle), dim=1)
        for key, value in (
            ("shallow", shallow),
            ("bottleneck", bottleneck),
            ("absolute_cycle", absolute_cycle),
            ("combined", combined),
        ):
            out[key].append(value.detach().cpu().numpy().astype(np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def fit_target_standardizer(x: np.ndarray):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((x - mean) / std).astype(np.float32)


class DeepSVDDHead(nn.Module):
    """Small bias-free mapping used for a one-class Deep SVDD pilot."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 64):
        super().__init__()
        hidden_dim = min(hidden_dim, max(32, in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def initialize_center(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
    )
    parts = []
    model.eval()
    for (batch,) in loader:
        parts.append(model(batch.to(device)).cpu())
    c = torch.cat(parts, dim=0).mean(dim=0).to(device)
    eps = 1e-3
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps
    return c.detach()


def train_deep_svdd(
    x_train: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    hidden_dim: int,
    out_dim: int,
):
    set_seed(seed)
    model = DeepSVDDHead(x_train.shape[1], hidden_dim, out_dim).to(device)
    center = initialize_center(model, x_train, device, batch_size)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            z = model(batch)
            dist2 = ((z - center) ** 2).sum(dim=1)
            loss = dist2.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        history.append(total / max(count, 1))
    return model, center, history


@torch.no_grad()
def svdd_distance(model, center, x: np.ndarray, device: torch.device, batch_size: int):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
    )
    out = []
    model.eval()
    for (batch,) in loader:
        z = model(batch.to(device))
        out.append(((z - center) ** 2).sum(dim=1).cpu().numpy())
    return np.concatenate(out)


def prototype_distance(x_train: np.ndarray, x: np.ndarray):
    proto = x_train.mean(axis=0, keepdims=True)
    return np.sum((x - proto) ** 2, axis=1)


def genuine_only_radius(dev_dist: np.ndarray, target_frr: float):
    """Set radius from genuine dev only; larger distance means more anomalous."""
    q = min(max(1.0 - target_frr, 0.0), 1.0)
    return float(np.quantile(dev_dist, q, method="higher"))


def descriptive_eer(labels: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def evaluate_distances(pos_dist, neg_dist, radius):
    # score is larger for genuine, so negate anomaly distance.
    y = np.r_[np.ones(len(pos_dist)), np.zeros(len(neg_dist))]
    score = -np.r_[pos_dist, neg_dist]
    far = float((neg_dist <= radius).mean())
    frr = float((pos_dist > radius).mean())
    return {
        "radius": float(radius),
        "final_auc": float(roc_auc_score(y, score)),
        "final_eer_descriptive": descriptive_eer(y, score),
        "final_far": far,
        "final_frr": frr,
        "final_hter": float((far + frr) / 2.0),
        "final_genuine_distance_mean_std": [
            float(np.mean(pos_dist)), float(np.std(pos_dist))
        ],
        "final_impostor_distance_mean_std": [
            float(np.mean(neg_dist)), float(np.std(neg_dist))
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_user", type=int, choices=(1, 4, 5), default=1)
    parser.add_argument(
        "--feature",
        choices=("shallow", "bottleneck", "absolute_cycle", "combined"),
        default="combined",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--final_genuine", type=int, choices=(15, 30), default=30)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--auth_seed", type=int, default=42)
    parser.add_argument("--svdd_seed", type=int, default=42)
    parser.add_argument("--target_frr", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument(
        "--output",
        default="/root/autodl-fs/deep_svdd_positive_only_pilot",
    )
    args = parser.parse_args()

    set_seed(args.svdd_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_name = USER_NAMES[args.target_user - 1]
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.target_user]

    protocol = build_target_protocol(
        args.target_user,
        split_seed=args.split_seed,
        auth_seed=args.auth_seed,
        target_final_n=args.final_genuine,
    )
    labels = protocol.labels
    target_label = args.target_user - 1

    model, _ = create_multi_source_model(
        checkpoint, device=device, freeze_generators=True, proj_dim=128
    )
    groups = (
        "auth_train",
        "target_dev_pool",
        "final_genuine",
        "final_impostor",
    )
    features = {}
    for group in groups:
        extracted = extract_features(
            model, protocol.arrays[group], device, args.batch_size
        )
        features[group] = extracted[args.feature]
    del model

    # Critical positive-only condition: the one-class head and standardizer see
    # only the target user's genuine training samples.
    target_train_mask = labels["auth_train"] == target_label
    x_train_raw = features["auth_train"][target_train_mask]
    x_dev_raw = features["target_dev_pool"]
    x_pos_raw = features["final_genuine"]
    x_neg_raw = features["final_impostor"]

    mean, std = fit_target_standardizer(x_train_raw)
    x_train = standardize(x_train_raw, mean, std)
    x_dev = standardize(x_dev_raw, mean, std)
    x_pos = standardize(x_pos_raw, mean, std)
    x_neg = standardize(x_neg_raw, mean, std)

    # Baseline: positive-only prototype distance.
    proto_dev = prototype_distance(x_train, x_dev)
    proto_radius = genuine_only_radius(proto_dev, args.target_frr)
    proto_result = evaluate_distances(
        prototype_distance(x_train, x_pos),
        prototype_distance(x_train, x_neg),
        proto_radius,
    )

    # Deep SVDD: positive-only training and genuine-only radius calibration.
    svdd, center, history = train_deep_svdd(
        x_train,
        device=device,
        seed=args.svdd_seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    )
    svdd_dev = svdd_distance(svdd, center, x_dev, device, args.batch_size)
    svdd_radius = genuine_only_radius(svdd_dev, args.target_frr)
    svdd_result = evaluate_distances(
        svdd_distance(svdd, center, x_pos, device, args.batch_size),
        svdd_distance(svdd, center, x_neg, device, args.batch_size),
        svdd_radius,
    )

    result = {
        "experiment": "positive_only_deep_svdd_pilot",
        "target": target_name,
        "target_user": args.target_user,
        "feature": args.feature,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "protocol": protocol.manifest["protocol"],
        "counts": {
            "target_positive_train": int(len(x_train)),
            "target_positive_dev": int(len(x_dev)),
            "final_genuine": int(len(x_pos)),
            "final_impostor": int(len(x_neg)),
        },
        "positive_only_constraints": {
            "head_training_uses_impostors": False,
            "standardizer_uses_impostors": False,
            "radius_calibration_uses_impostors": False,
            "final_impostors_used_for_evaluation_only": True,
        },
        "target_frr_for_radius_calibration": args.target_frr,
        "prototype_baseline": proto_result,
        "deep_svdd": {
            **svdd_result,
            "train_loss_first": float(history[0]),
            "train_loss_last": float(history[-1]),
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "out_dim": args.out_dim,
            "seed": args.svdd_seed,
        },
        "warning": (
            "Exploratory only: current acc.npy is already preprocessed with "
            "per-user full-data statistics. Final paper results require raw-data "
            "rerun with train-only preprocessing and preferably word/session grouping."
        ),
    }

    out = Path(args.output) / f"user{args.target_user}_{args.feature}_seed{args.svdd_seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_manifest(protocol, str(out / "protocol_manifest.json"))
    (out / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": svdd.state_dict(),
            "center": center.detach().cpu(),
            "feature_mean": torch.from_numpy(mean),
            "feature_std": torch.from_numpy(std),
            "args": vars(args),
        },
        out / "deep_svdd.pt",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
