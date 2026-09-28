"""Train a target-specific authentication head with a leakage-free final test."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from data import MultiUserSpectrogramDataset, SpectrogramAugment
from modelb import SupConLoss, create_multi_source_model
from target_auth_protocol_150 import build_target_protocol, save_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_user", type=int, required=True, choices=(1, 3, 4, 5))
    parser.add_argument("--cyclegan_ckpt", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--cyclegan_seed", type=int, default=42)
    parser.add_argument("--auth_seed", type=int, default=42)
    parser.add_argument("--class_feature_copies", type=int, default=2)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--no_shallow", action="store_true")
    parser.add_argument("--no_bottleneck", action="store_true")
    parser.add_argument("--no_cycle_diff", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def separation(model, loader, device):
    model.eval()
    embeddings, labels = [], []
    for specs, labs in loader:
        embeddings.append(model(specs.to(device), return_embedding=True).cpu())
        labels.append(labs)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    similarity = embeddings @ embeddings.T
    equal = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool)
    intra = similarity[equal & off_diagonal].mean().item()
    inter = similarity[(~equal) & off_diagonal].mean().item()
    return {"intra": intra, "inter": inter, "sep": intra - inter}


@torch.no_grad()
def fusion_inputs(model, specs):
    """Run the frozen CycleGAN branches once and return fusion-layer inputs."""
    features = []
    if model.use_shallow:
        shallow_map = model.shallow_encoder(specs)
        features.append(model.shallow_agg(shallow_map))
    if model.use_bottleneck:
        bottleneck_map = model.bottleneck_encoder(specs)
        features.append(model.bottleneck_agg(bottleneck_map))
    if model.use_cycle_diff:
        diff_map, diff_stats = model._compute_cycle_diff(specs)
        encoded_diff = model.diff_encoder(diff_map)
        features.append(model.diff_agg(encoded_diff))
        features.append(diff_stats)
    return torch.cat(features, dim=1)


@torch.no_grad()
def precompute_features(model, specs, labels, device, batch_size, augment=None):
    dataset = MultiUserSpectrogramDataset(specs, labels, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cached, cached_labels = [], []
    model.eval()
    for batch, labs in loader:
        cached.append(fusion_inputs(model, batch.to(device)).cpu())
        cached_labels.append(labs.cpu())
    return torch.cat(cached), torch.cat(cached_labels)


def head_projection(model, cached_features):
    return model.projector(model.fusion(cached_features))


@torch.no_grad()
def cached_separation(model, loader, device):
    model.eval()
    embeddings, labels = [], []
    for features, labs in loader:
        fused = model.fusion(features.to(device))
        embeddings.append(torch.nn.functional.normalize(fused, p=2, dim=1).cpu())
        labels.append(labs)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    similarity = embeddings @ embeddings.T
    equal = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool)
    intra = similarity[equal & off_diagonal].mean().item()
    inter = similarity[(~equal) & off_diagonal].mean().item()
    return {"intra": intra, "inter": inter, "sep": intra - inter}


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protocol = build_target_protocol(
        args.target_user,
        split_seed=args.cyclegan_seed,
        auth_seed=args.auth_seed,
    )
    manifest_path = os.path.join(args.save_dir, "split_manifest.json")
    save_manifest(protocol, manifest_path)
    print(json.dumps(protocol.manifest["counts"], ensure_ascii=False))

    model, info = create_multi_source_model(
        args.cyclegan_ckpt,
        use_shallow=not args.no_shallow,
        use_bottleneck=not args.no_bottleneck,
        use_cycle_diff=not args.no_cycle_diff,
        freeze_generators=True,
        proj_dim=args.proj_dim,
        device=device,
    )
    print(f"device={device} target={args.target_user} sources={info['sources']}")

    print("precomputing frozen CycleGAN features with class-balanced augmentation", flush=True)
    train_feature_parts, train_label_parts = [], []
    raw_train_specs = protocol.arrays["auth_train"]
    raw_train_labels = protocol.labels["auth_train"]
    class_counts = {int(label): int((raw_train_labels == label).sum()) for label in range(9)}
    desired_per_class = max(class_counts.values()) * args.class_feature_copies
    for label in range(9):
        mask = raw_train_labels == label
        class_specs = raw_train_specs[mask]
        class_labels = raw_train_labels[mask]
        produced_features, produced_labels = [], []
        repeat = 0
        while sum(len(item) for item in produced_features) < desired_per_class:
            cached_features, cached_labels = precompute_features(
                model,
                class_specs,
                class_labels,
                device,
                args.feature_batch_size,
                augment=None if repeat == 0 else SpectrogramAugment(),
            )
            produced_features.append(cached_features)
            produced_labels.append(cached_labels)
            repeat += 1
        train_feature_parts.append(torch.cat(produced_features)[:desired_per_class])
        train_label_parts.append(torch.cat(produced_labels)[:desired_per_class])
        print(
            f"cached class={label} raw={class_counts[label]} "
            f"balanced={desired_per_class} passes={repeat}",
            flush=True,
        )
    train_features = torch.cat(train_feature_parts)
    train_labels = torch.cat(train_label_parts)
    val_features, val_labels = precompute_features(
        model,
        protocol.arrays["model_validation"],
        protocol.labels["model_validation"],
        device,
        args.feature_batch_size,
        augment=None,
    )
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(
        f"cached train={len(train_features)} validation={len(val_features)} "
        f"fusion_dim={train_features.shape[1]}",
        flush=True,
    )
    optimizer = Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    criterion = SupConLoss(args.temperature)
    history = {"loss": [], "intra": [], "inter": [], "sep": [], "lr": []}
    best_sep = -float("inf")
    best_path = os.path.join(args.save_dir, "best_auth_model.pth")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            projection = head_projection(model, features)
            loss = criterion(projection, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()
        metrics = cached_separation(model, val_loader, device)
        mean_loss = total_loss / max(batches, 1)
        history["loss"].append(mean_loss)
        history["intra"].append(metrics["intra"])
        history["inter"].append(metrics["inter"])
        history["sep"].append(metrics["sep"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(
            f"[target {args.target_user} ep {epoch + 1:03d}/{args.num_epochs}] "
            f"loss={mean_loss:.5f} intra={metrics['intra']:.5f} "
            f"inter={metrics['inter']:.5f} sep={metrics['sep']:.5f}",
            flush=True,
        )
        if metrics["sep"] > best_sep:
            best_sep = metrics["sep"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "model_type": "multi",
                "model_config": model.get_config(),
                "cyclegan_ckpt": args.cyclegan_ckpt,
                "target_user": args.target_user,
                "protocol": "150_cyclegan_plus_auth_v1",
                "cyclegan_seed": args.cyclegan_seed,
                "auth_seed": args.auth_seed,
                "manifest_path": manifest_path,
                "metrics": metrics,
                "history": history,
                "n_users": 9,
            }, best_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"])
    axes[0].set_title("Training loss")
    axes[1].plot(history["sep"])
    axes[1].set_title("Model-validation separation")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.log_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"completed target={args.target_user} best_sep={best_sep:.6f} model={best_path}")


if __name__ == "__main__":
    main()


