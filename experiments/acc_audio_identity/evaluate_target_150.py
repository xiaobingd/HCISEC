"""Evaluate target-specific heads with calibration-selected frozen thresholds."""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from data import MultiUserSpectrogramDataset
from modelb import create_multi_source_model
from target_auth_protocol_150 import USER_NAMES, build_target_protocol


SEEDS = [42, 123, 456, 789, 2026]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def embed(model, specs, labels, device, batch_size):
    dataset = MultiUserSpectrogramDataset(specs, labels, augment=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch, _ in loader:
        embeddings.append(model(batch.to(device), return_embedding=True).cpu().numpy())
    return np.concatenate(embeddings)


def choose_threshold(genuine, impostor):
    values = np.unique(np.concatenate([genuine, impostor]))
    if len(values) > 1:
        candidates = np.concatenate([
            [values[0] - 1e-6],
            (values[:-1] + values[1:]) / 2.0,
            [values[-1] + 1e-6],
        ])
    else:
        candidates = values
    best = None
    for threshold in candidates:
        far = float(np.mean(impostor >= threshold))
        frr = float(np.mean(genuine < threshold))
        key = (abs(far - frr), (far + frr) / 2.0)
        if best is None or key < best[0]:
            best = (key, float(threshold), far, frr)
    return best[1]


def test_eer(genuine, impostor):
    threshold = choose_threshold(genuine, impostor)
    far = float(np.mean(impostor >= threshold))
    frr = float(np.mean(genuine < threshold))
    return (far + frr) / 2.0


def scores(embeddings, template):
    template = template / (np.linalg.norm(template) + 1e-12)
    normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return normalized @ template


def compact_roc(genuine, impostor, points=101):
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    values = np.concatenate([genuine, impostor])
    fpr, tpr, thresholds = roc_curve(labels, values)
    grid = np.linspace(0.0, 1.0, points)
    interpolated_tpr = np.interp(grid, fpr, tpr)
    return {
        "fpr": grid.tolist(),
        "tpr": interpolated_tpr.tolist(),
        "fnr": (1.0 - interpolated_tpr).tolist(),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}
    for target_user in (1, 3, 4, 5):
        target_dir = os.path.join(args.experiment_root, f"user{target_user}")
        checkpoint_path = os.path.join(target_dir, "auth", "best_auth_model.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        protocol = build_target_protocol(
            target_user,
            split_seed=int(checkpoint["cyclegan_seed"]),
            auth_seed=int(checkpoint["auth_seed"]),
        )
        model, _ = create_multi_source_model(
            checkpoint["cyclegan_ckpt"],
            use_shallow=checkpoint["model_config"].get("use_shallow", True),
            use_bottleneck=checkpoint["model_config"].get("use_bottleneck", True),
            use_cycle_diff=checkpoint["model_config"].get("use_cycle_diff", True),
            freeze_generators=True,
            proj_dim=int(checkpoint["model_config"].get("proj_dim", 128)),
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        target_label = target_user - 1

        dev_emb = embed(model, protocol.arrays["target_dev_pool"],
                        protocol.labels["target_dev_pool"], device, args.batch_size)
        genuine_test_emb = embed(model, protocol.arrays["final_genuine"],
                                 protocol.labels["final_genuine"], device, args.batch_size)
        imp_cal_emb = embed(model, protocol.arrays["impostor_calibration"],
                            protocol.labels["impostor_calibration"], device, args.batch_size)
        imp_test_emb = embed(model, protocol.arrays["final_impostor"],
                             protocol.labels["final_impostor"], device, args.batch_size)
        imp_test_labels = protocol.labels["final_impostor"]

        seed_results = []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(dev_emb))
            split = len(perm) // 2
            enrollment = dev_emb[perm[:split]]
            genuine_calibration = dev_emb[perm[split:]]
            template = enrollment.mean(axis=0)
            gen_cal_scores = scores(genuine_calibration, template)
            imp_cal_scores = scores(imp_cal_emb, template)
            threshold = choose_threshold(gen_cal_scores, imp_cal_scores)
            gen_test_scores = scores(genuine_test_emb, template)
            imp_test_scores = scores(imp_test_emb, template)
            far = float(np.mean(imp_test_scores >= threshold))
            frr = float(np.mean(gen_test_scores < threshold))
            test_labels = np.concatenate([
                np.ones(len(gen_test_scores)), np.zeros(len(imp_test_scores))
            ])
            test_values = np.concatenate([gen_test_scores, imp_test_scores])
            roc_auc = float(roc_auc_score(test_labels, test_values))
            hard = {}
            for label in range(9):
                if label == target_label:
                    continue
                mask = imp_test_labels == label
                hard[USER_NAMES[label]] = {
                    "count": int(np.sum(imp_test_scores[mask] >= threshold)),
                    "n": int(np.sum(mask)),
                    "far": float(np.mean(imp_test_scores[mask] >= threshold)),
                }
            seed_results.append({
                "seed": seed,
                "threshold": float(threshold),
                "far": far,
                "frr": frr,
                "hter": (far + frr) / 2.0,
                "balanced_accuracy": 1.0 - (far + frr) / 2.0,
                "test_eer_descriptive": test_eer(gen_test_scores, imp_test_scores),
                "roc_auc": roc_auc,
                "roc": compact_roc(gen_test_scores, imp_test_scores),
                "hard_impostors": hard,
            })

        aggregate = {}
        for key in ("far", "frr", "hter", "balanced_accuracy", "test_eer_descriptive", "roc_auc"):
            values = np.asarray([row[key] for row in seed_results])
            aggregate[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        all_results[str(target_user)] = {
            "target_name": USER_NAMES[target_label],
            "best_epoch": int(checkpoint["epoch"]),
            "model_validation_sep": float(checkpoint["metrics"]["sep"]),
            "counts": protocol.manifest["counts"],
            "seeds": seed_results,
            "aggregate": aggregate,
        }
        print(
            f"{USER_NAMES[target_label]}: EER={aggregate['test_eer_descriptive']['mean']:.4f} "
            f"HTER={aggregate['hter']['mean']:.4f} BA={aggregate['balanced_accuracy']['mean']:.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    macro = {}
    for key in ("far", "frr", "hter", "balanced_accuracy", "test_eer_descriptive", "roc_auc"):
        values = np.asarray([all_results[str(i)]["aggregate"][key]["mean"] for i in (1, 3, 4, 5)])
        macro[key] = {"mean": float(values.mean()), "std_across_users": float(values.std(ddof=1))}
    output = {"protocol": "150_cyclegan_plus_auth_v1", "users": all_results, "macro": macro}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps(macro, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


