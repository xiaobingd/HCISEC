"""
多源特征融合 1:N 对比学习身份认证 — 训练脚本

特征源：
  1. 原始ACC浅层特征（CycleGAN编码器浅层）
  2. CycleGAN瓶颈特征（AxialAIA后）
  3. Cycle重建差异特征（ACC→Audio→ACC'的重建误差）

使用方法：
    # 多源融合训练（推荐）
    python trainb.py \\
        --cyclegan_ckpt ./models_bo150/checkpoint_epoch_300.pth \\
        --simulate_users 5 --num_epochs 100
        
        # 只用F1，关闭F4/F5（避免稀释）
python trainb.py \
    --model_type v2 \
    --cyclegan_ckpt ./models_bo.1/checkpoint_epoch_300.pth \
    --diff_ckpt ./models_diff/best_diff.pth \
    --no_f4 --no_f5 \
    --num_epochs 100

python trainb.py \
    --model_type v2 \
    --cyclegan_ckpt ./models_bo150y/checkpoint_epoch_300.pth \
    --diff_ckpt ./models_diffbo150y/best_diff.pth \
    --no_f4 --no_f5 \
    --num_epochs 100

    # 仅瓶颈特征训练（消融对比）
    python trainb.py \
        --cyclegan_ckpt ./models_bo150/checkpoint_epoch_300.pth \
        --model_type single --num_epochs 100
        
    /cycle
        python trainb.py \
        --cyclegan_ckpt ./models_bo150y/checkpoint_epoch_300.pth \
        --model_type multi --num_epochs 100

    # 指定启用/禁用特征源
    python train.py \\
        --cyclegan_ckpt ... --no_shallow --no_cycle_diff
"""

import os
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modelb import (create_multi_source_model, create_single_source_model,
                   create_multi_source_model_v2, SupConLoss, SimCLRLoss)
from data import (
    SpectrogramAugment, ContrastiveAugment,
    create_dataloaders, load_single_user_data, prepare_multi_user_simulation,
    load_real_auth_data,
)


def parse_args():
    p = argparse.ArgumentParser(description='多源特征融合对比学习认证训练')

    # CycleGAN
    p.add_argument('--cyclegan_ckpt', type=str, required=True)
    p.add_argument('--freeze_generators', action='store_true', default=True)

    # 模型类型
    p.add_argument('--model_type', type=str, default='multi',
                   choices=['multi', 'single', 'v2'],
                   help='multi=CycleGAN多源, single=仅瓶颈, v2=CycleGAN+Diffusion融合')

    # Diffusion (v2模式)
    p.add_argument('--diff_ckpt', type=str, default=None,
                   help='Diffusion checkpoint路径 (v2模式必需)')
    p.add_argument('--ddim_steps', type=int, default=50)
    p.add_argument('--no_f1', action='store_true', help='禁用Diffusion F1特征')
    p.add_argument('--no_f4', action='store_true', help='禁用Diffusion F4特征')
    p.add_argument('--no_f5', action='store_true', help='禁用Diffusion F5特征')

    # 特征源开关（multi/v2模式）
    p.add_argument('--no_shallow', action='store_true', help='禁用浅层特征')
    p.add_argument('--no_bottleneck', action='store_true', help='禁用瓶颈特征')
    p.add_argument('--no_cycle_diff', action='store_true', help='禁用重建差异特征')

    # 数据
    p.add_argument('--acc_path', type=str,
                   default='./cyclegan_acc_selected.npy')
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--simulate_users', type=int, default=5)

    # 训练
    p.add_argument('--mode', type=str, default='supcon',
                   choices=['supcon', 'simclr'])
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--proj_dim', type=int, default=128)

    # 增强
    p.add_argument('--aug_strength', type=str, default='medium',
                   choices=['light', 'medium', 'strong'])

    # 输出
    p.add_argument('--save_dir', type=str, default='./checkpoints_auth-y-8')
    p.add_argument('--log_dir', type=str, default='./logs_auth-y-8')
    p.add_argument('--save_every', type=int, default=20)

    return p.parse_args()


# ============================================================
# 训练 / 评估
# ============================================================

def train_supcon_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, n = 0.0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        proj = model(specs)
        loss = loss_fn(proj, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def train_simclr_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, n = 0.0, 0
    for v1, v2, _ in loader:
        v1, v2 = v1.to(device), v2.to(device)
        z1, z2 = model(v1), model(v2)
        loss = loss_fn(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    embs, labs = [], []
    for specs, labels in loader:
        emb = model(specs.to(device), return_embedding=True).cpu()
        embs.append(emb); labs.append(labels)
    embs = torch.cat(embs); labs = torch.cat(labs)

    sim = embs @ embs.T
    eq  = labs.unsqueeze(0) == labs.unsqueeze(1)
    eye = ~torch.eye(len(labs), dtype=torch.bool)

    intra = sim[eq & eye].mean().item() if (eq & eye).any() else 0
    inter = sim[~eq & eye].mean().item() if (~eq & eye).any() else 0
    return {'intra': intra, 'inter': inter, 'sep': intra - inter}


# ============================================================
# 可视化
# ============================================================

def plot_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['loss'], 'b-'); axes[0].set_title('Loss'); axes[0].grid(True, alpha=.3)
    axes[1].plot(history['intra'], 'g-', label='Intra')
    axes[1].plot(history['inter'], 'r-', label='Inter')
    axes[1].set_title('Cosine Similarity'); axes[1].legend(); axes[1].grid(True, alpha=.3)
    axes[2].plot(history['sep'], 'm-'); axes[2].set_title('Separability'); axes[2].grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_tsne(model, loader, device, path, n_users):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for s, l in loader:
            embs.append(model(s.to(device), return_embedding=True).cpu())
            labs.append(l)
    embs = torch.cat(embs).numpy(); labs = torch.cat(labs).numpy()
    if len(embs) < 5: return
    perp = min(30, len(embs) - 1)
    coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(embs)
    plt.figure(figsize=(10, 8))
    for u in range(n_users):
        m = labs == u
        plt.scatter(coords[m, 0], coords[m, 1], label=f'User {u}', alpha=.7, s=30)
    plt.title('t-SNE: Multi-Source Identity Embeddings'); plt.legend()
    plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"t-SNE → {path}")


# ============================================================
# main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 1. 创建模型 ----
    print(f"\n加载 CycleGAN: {args.cyclegan_ckpt}")
    if args.model_type == 'v2':
        if args.diff_ckpt is None:
            raise ValueError("v2模式需要 --diff_ckpt 参数")
        print(f"加载 Diffusion: {args.diff_ckpt}")
        model, info = create_multi_source_model_v2(
            cyclegan_ckpt=args.cyclegan_ckpt,
            diff_ckpt=args.diff_ckpt,
            use_shallow=not args.no_shallow,
            use_bottleneck=not args.no_bottleneck,
            use_cycle_diff=not args.no_cycle_diff,
            use_f1=not args.no_f1,
            use_f4=not args.no_f4,
            use_f5=not args.no_f5,
            ddim_steps=args.ddim_steps,
            proj_dim=args.proj_dim,
            device=device,
        )
        print(f"  V2融合模型, 特征源: {info['sources']}")
    elif args.model_type == 'multi':
        model, info = create_multi_source_model(
            args.cyclegan_ckpt,
            use_shallow=not args.no_shallow,
            use_bottleneck=not args.no_bottleneck,
            use_cycle_diff=not args.no_cycle_diff,
            freeze_generators=args.freeze_generators,
            proj_dim=args.proj_dim,
            device=device,
        )
        print(f"  多源融合模型, 特征源: {info['sources']}")
    else:
        model, info = create_single_source_model(
            args.cyclegan_ckpt,
            freeze=True, proj_dim=args.proj_dim, device=device,
        )
        print(f"  单源模型(仅瓶颈), feat_dim={info['feat_dim']}")

    print(f"  CycleGAN epoch: {info.get('epoch', info.get('cyclegan_epoch', '?'))}")
    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数: 总{total:,} | 可训练{train_p:,} | 冻结{total-train_p:,}")

    # ---- 2. 数据 ----
    # 默认加载真实多用户数据：bo(CycleGAN val+test) + mei
    print("\n加载真实多用户数据:")
    specs, labels, user_names = load_real_auth_data()
    n_users = len(user_names)
    print(f"  数据: {len(specs)} 样本, {n_users} 用户 ({user_names})")

    aug = SpectrogramAugment() if args.mode == 'supcon' else None
    c_aug = ContrastiveAugment(args.aug_strength) if args.mode == 'simclr' else None
    train_ld, val_ld, test_ld = create_dataloaders(
        specs, labels, batch_size=args.batch_size,
        augment=aug, contrastive_augment=c_aug)

    # ---- 3. 损失 / 优化器 ----
    loss_fn = SupConLoss(args.temperature) if args.mode == 'supcon' else SimCLRLoss(0.5)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # ---- 4. 训练 ----
    history = {'loss': [], 'intra': [], 'inter': [], 'sep': [], 'lr': []}
    best_sep = -float('inf')

    print(f"\n{'='*60}")
    print(f"开始训练: {args.num_epochs} epochs, mode={args.mode}, "
          f"model={args.model_type}")
    print(f"{'='*60}\n")

    for epoch in range(args.num_epochs):
        if args.mode == 'supcon':
            loss = train_supcon_epoch(model, train_ld, optimizer, loss_fn, device)
        else:
            loss = train_simclr_epoch(model, train_ld, optimizer, loss_fn, device)

        metrics = evaluate(model, val_ld, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['loss'].append(loss)
        history['intra'].append(metrics['intra'])
        history['inter'].append(metrics['inter'])
        history['sep'].append(metrics['sep'])
        history['lr'].append(lr)

        print(f"[Ep {epoch+1:>3d}/{args.num_epochs}] "
              f"Loss={loss:.4f} | "
              f"Intra={metrics['intra']:.4f} Inter={metrics['inter']:.4f} "
              f"Sep={metrics['sep']:.4f} | LR={lr:.2e}")

        if metrics['sep'] > best_sep:
            best_sep = metrics['sep']
            best_path = os.path.join(args.save_dir, 'best_auth_model.pth')
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'model_type': args.model_type,
                'cyclegan_ckpt': args.cyclegan_ckpt,
                'feat_dim': model.feat_dim,
                'proj_dim': args.proj_dim,
                'metrics': metrics,
                'history': history,
                'n_users': n_users,
            }
            if args.model_type in ('multi', 'v2'):
                save_dict['model_config'] = model.get_config()
            if args.model_type == 'v2':
                save_dict['diff_ckpt'] = args.diff_ckpt
                save_dict['ddim_steps'] = args.ddim_steps
            torch.save(save_dict, best_path)
            print(f"  ★ Best Sep={best_sep:.4f} → {best_path}")

        if (epoch + 1) % args.save_every == 0:
            plot_curves(history, os.path.join(args.log_dir, f'curves_ep{epoch+1}.png'))

    # ---- 5. 最终评估 ----
    print(f"\n{'='*60}")
    print(f"训练完成! Best Sep={best_sep:.4f}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_m = evaluate(model, test_ld, device)
    print(f"测试集: Intra={test_m['intra']:.4f} Inter={test_m['inter']:.4f} "
          f"Sep={test_m['sep']:.4f}")

    plot_curves(history, os.path.join(args.log_dir, 'curves_final.png'))
    plot_tsne(model, test_ld, device,
              os.path.join(args.log_dir, 'tsne_final.png'), n_users)

    print(f"\n模型: {args.save_dir}/")
    print(f"日志: {args.log_dir}/")


if __name__ == '__main__':
    main()
