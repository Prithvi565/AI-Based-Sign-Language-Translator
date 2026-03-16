import os
import json
import time
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import (MobileNet_V2_Weights,
                                ResNet50_Weights,
                                EfficientNet_B0_Weights)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# GPU SETUP
# ══════════════════════════════════════════════════════════════════════════════

torch.backends.cudnn.benchmark        = True   # auto-tune CUDA kernels
torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul on RTX 30xx

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    _props = torch.cuda.get_device_properties(0)
    print(f"\n✅ GPU  : {_props.name}")
    print(f"   VRAM : {_props.total_memory / 1e9:.1f} GB")
    print(f"   CUDA : {torch.version.cuda}\n")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  No GPU found — running on CPU\n")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="ISL Classifier — RTX 3050 optimised")

    p.add_argument("--data_dir",   type=str,
                   default=r"C:\D DRIVE\Projects\Sign Language Translator\Datasets\archive\data",
                   help="Root folder containing one sub-folder per class")
    p.add_argument("--output_dir", type=str, default="isl_output")

    p.add_argument("--model",  type=str, default="mobilenet",
                   choices=["mobilenet", "resnet50", "efficientnet"])

    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--img_size",    type=int,   default=64)
    p.add_argument("--num_workers", type=int,   default=4,
                   help="CPU threads for loading — 4 is good for 8-core + SSD")
    p.add_argument("--dropout",     type=float, default=0.4)
    p.add_argument("--freeze_base", action="store_true")

    p.add_argument("--val_split",  type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms(img_size: int, mode: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size + 10, img_size + 10)),
            transforms.RandomCrop(img_size),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), shear=4),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir: str, img_size: int,
              val_frac: float, test_frac: float,
              seed: int, batch_size: int, num_workers: int):

    data_dir = Path(data_dir)
    assert data_dir.exists(), f"\n❌  Dataset folder not found: {data_dir}"

    # Scan folder structure to get paths + labels
    raw         = datasets.ImageFolder(str(data_dir))
    class_names = raw.classes
    all_paths   = [s[0] for s in raw.samples]
    all_labels  = [s[1] for s in raw.samples]

    print(f"  Found   : {len(all_paths):,} images  across  {len(class_names)} classes")
    print(f"  Classes : {class_names}\n")

    # ── Stratified split ──────────────────────────────────────────────────
    idx  = np.arange(len(all_labels))
    sss1 = StratifiedShuffleSplit(1, test_size=test_frac, random_state=seed)
    tv_idx, te_idx = next(sss1.split(idx, all_labels))

    tv_labs = [all_labels[i] for i in tv_idx]
    adj_val = val_frac / (1.0 - test_frac)
    sss2    = StratifiedShuffleSplit(1, test_size=adj_val, random_state=seed)
    rel_tr, rel_val = next(sss2.split(tv_idx, tv_labs))
    tr_idx  = tv_idx[rel_tr]
    val_idx = tv_idx[rel_val]

    print(f"  Split → Train: {len(tr_idx):,}  |  Val: {len(val_idx):,}  |  Test: {len(te_idx):,}\n")

    # ── Build datasets with correct transforms ────────────────────────────
    from torch.utils.data import Subset
    train_ds = Subset(datasets.ImageFolder(str(data_dir),
                      transform=get_transforms(img_size, "train")), tr_idx)
    val_ds   = Subset(datasets.ImageFolder(str(data_dir),
                      transform=get_transforms(img_size, "val")),   val_idx)
    test_ds  = Subset(datasets.ImageFolder(str(data_dir),
                      transform=get_transforms(img_size, "val")),   te_idx)

    # ── WeightedRandomSampler — fixes class 0 imbalance (654 vs 2400) ────
    label_counts  = np.bincount([all_labels[i] for i in tr_idx])
    class_weights = 1.0 / (label_counts.astype(float) + 1e-6)
    sample_weights = torch.tensor(
        [class_weights[all_labels[i]] for i in tr_idx], dtype=torch.float
    )
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    # num_workers=4  : 4 CPU threads pre-load batches while GPU is computing
    # pin_memory     : keeps tensors in page-locked RAM for faster GPU transfer
    # persistent_workers : workers stay alive between epochs (no restart cost)
    # prefetch_factor: each worker pre-loads 2 batches ahead
    loader_kw = dict(
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 2,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False,  **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False,  **loader_kw)

    return train_loader, val_loader, test_loader, class_names


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_model(backbone: str, num_classes: int,
                freeze_base: bool, dropout: float) -> nn.Module:

    if backbone == "mobilenet":
        m    = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
        if freeze_base:
            for p in m.features.parameters():
                p.requires_grad = False

    elif backbone == "resnet50":
        m    = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
        if freeze_base:
            for name, p in m.named_parameters():
                if "fc" not in name:
                    p.requires_grad = False

    elif backbone == "efficientnet":
        m    = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, num_classes),
        )
        if freeze_base:
            for p in m.features.parameters():
                p.requires_grad = False

    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {backbone}  |  {total:,} total params  |  {trainable:,} trainable\n")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()

    loss_sum  = correct = total = 0
    all_preds = []
    all_true  = []

    ctx  = torch.enable_grad() if training else torch.no_grad()
    desc = "  train" if training else "  val  "
    bar  = tqdm(loader, desc=desc, leave=False, ncols=95)

    with ctx:
        for imgs, labels in bar:
            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            out  = model(imgs)
            loss = criterion(out, labels)

            if training:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * imgs.size(0)
            preds     = out.argmax(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1e9
                bar.set_postfix(loss=f"{loss.item():.3f}",
                                acc =f"{correct/total:.3f}",
                                VRAM=f"{vram:.1f}GB")
            else:
                bar.set_postfix(loss=f"{loss.item():.3f}",
                                acc =f"{correct/total:.3f}")

    return loss_sum / total, correct / total, all_preds, all_true


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def save_plots(history, ts_true, ts_preds, class_names, out_dir):
    ep = range(1, len(history["tl"]) + 1)

    # 1 — Loss & accuracy curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(ep, history["tl"], lw=2, label="Train")
    a1.plot(ep, history["vl"], lw=2, label="Val", ls="--")
    a1.set_title("Loss", fontsize=13); a1.legend(); a1.grid(alpha=.3); a1.set_xlabel("Epoch")
    a2.plot(ep, history["ta"], lw=2, label="Train")
    a2.plot(ep, history["va"], lw=2, label="Val", ls="--")
    a2.set_title("Accuracy", fontsize=13); a2.legend(); a2.grid(alpha=.3); a2.set_xlabel("Epoch")
    plt.suptitle("ISL Training Curves", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2 — Confusion matrix
    cm = confusion_matrix(ts_true, ts_preds)
    n  = len(class_names)
    fig, ax = plt.subplots(figsize=(n * .6 + 1, n * .5 + 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=.3, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix — ISL", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()

    # 3 — Per-class accuracy bar chart
    per    = cm.diagonal() / cm.sum(axis=1)
    colors = ["#2ecc71" if a >= .90 else
              "#f39c12" if a >= .75 else
              "#e74c3c" for a in per]
    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(class_names, per * 100, color=colors, edgecolor="white")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_title("Per-Class Accuracy — ISL  (🟢 ≥90%  🟠 ≥75%  🔴 <75%)", fontsize=13)
    ax.axhline(90, color="gray", ls="--", lw=.8)
    for b, a in zip(bars, per):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{a*100:.0f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved → training_curves.png")
    print("  Saved → confusion_matrix.png")
    print("  Saved → per_class_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"{'═'*62}")
    print(f"  ISL Classifier  —  RTX 3050 Optimised")
    print(f"{'═'*62}")
    print(f"  Device      : {DEVICE}  ({gpu_name})")
    print(f"  Backbone    : {args.model}")
    print(f"  Image size  : {args.img_size}×{args.img_size}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Workers     : {args.num_workers}")
    print(f"  Epochs      : {args.epochs}   LR : {args.lr}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"{'═'*62}\n")

    # ── 1. Load data ───────────────────────────────────────────────────────
    print("[1/4] Loading dataset...")
    train_loader, val_loader, test_loader, class_names = load_data(
        args.data_dir, args.img_size,
        args.val_split, args.test_split,
        args.seed, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)

    # ── 2. Build model ─────────────────────────────────────────────────────
    print(f"[2/4] Building {args.model} ({num_classes} output classes)...")
    model = build_model(args.model, num_classes,
                        args.freeze_base, args.dropout).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    def lr_lambda(ep):
        warmup = 3
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 3. Training loop ───────────────────────────────────────────────────
    print(f"[3/4] Training for up to {args.epochs} epochs...\n")

    history   = {"tl": [], "ta": [], "vl": [], "va": []}
    best_acc  = 0.0
    ckpt_path = os.path.join(args.output_dir, "best_isl_model.pth")
    no_imp    = 0
    PATIENCE  = 8

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE, training=True)
        vl_loss, vl_acc, _, _ = run_epoch(
            model, val_loader,   criterion, None,      DEVICE, training=False)
        scheduler.step()
        elapsed = time.time() - t0

        history["tl"].append(tr_loss); history["ta"].append(tr_acc)
        history["vl"].append(vl_loss); history["va"].append(vl_acc)

        vram_str = ""
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_str = f"  VRAM {used:.1f}/{total:.1f}GB"

        star = " ★" if vl_acc > best_acc else ""
        print(f"  Ep {ep:3d}/{args.epochs}  "
              f"Train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"Val {vl_loss:.4f}/{vl_acc:.4f}  "
              f"LR {scheduler.get_last_lr()[0]:.5f}  "
              f"{elapsed:.1f}s{vram_str}{star}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            no_imp   = 0
            torch.save({
                "epoch":                ep,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":              vl_acc,
                "class_names":          class_names,
                "backbone":             args.model,
                "img_size":             args.img_size,
                "num_classes":          num_classes,
            }, ckpt_path)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"\n  Early stopping — no improvement for {PATIENCE} epochs.")
                break

    print(f"\n  Best Val Accuracy : {best_acc * 100:.2f}%\n")

    # ── 4. Test evaluation ─────────────────────────────────────────────────
    print("[4/4] Evaluating on held-out test set...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    ts_loss, ts_acc, ts_preds, ts_true = run_epoch(
        model, test_loader, criterion, None, DEVICE, training=False)

    print(f"\n  Test Accuracy : {ts_acc * 100:.2f}%")
    print(f"  Test Loss     : {ts_loss:.4f}\n")

    report = classification_report(ts_true, ts_preds,
                                   target_names=class_names, digits=4)
    print(report)

    with open(f"{args.output_dir}/classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {ts_acc:.4f}\n\n{report}")
    with open(f"{args.output_dir}/class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # save_plots(history, ts_true, ts_preds, class_names, args.output_dir)

    print(f"\n{'═'*62}")
    print(f"  ✅ Done!")
    print(f"  Model   →  {ckpt_path}")
    print(f"  Reports →  {args.output_dir}/classification_report.txt")
    print(f"  Plots   →  {args.output_dir}/")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()