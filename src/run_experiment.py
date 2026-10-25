import sys
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

warnings.filterwarnings("ignore")

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset import load_vqarad, VQADataset, get_transforms
from models import MODEL_REGISTRY

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, q_ids, labels in loader:
        imgs, q_ids, labels = imgs.to(device), q_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs, q_ids)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, q_ids, labels in loader:
        imgs, q_ids = imgs.to(device), q_ids.to(device)
        with autocast():
            logits = model(imgs, q_ids)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    f1 = f1_score(all_labels, all_preds, average="macro")
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return f1, bal_acc, acc

def run_model(model_name, train_df, val_df, test_df, vocab_size, device, epochs=20, lr=1e-3, batch_size=32, patience=6, mask_p=0.0, seed=42):
    set_seed(seed)
    model = MODEL_REGISTRY[model_name](vocab_size).to(device)

    train_ds = VQADataset(train_df, get_transforms(True), mask_p=mask_p)
    val_ds = VQADataset(val_df, get_transforms(False))
    test_ds = VQADataset(test_df, get_transforms(False))

    counts = train_df["y"].value_counts().sort_index()
    sample_weights = [1.0 / counts[y] for y in train_df["y"]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    cw = 1.0 / counts.values.astype(float)
    cw = cw / cw.sum() * len(counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_f1 = 0.0
    patience_cnt = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return evaluate(model, test_loader, device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to VQA_RAD folder")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--mask_experiment", action="store_true", help="Run image masking experiment")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df, val_df, test_df, vocab_size = load_vqarad(args.data_dir)

    model_names = list(MODEL_REGISTRY.keys()) if args.models == "all" else [m.strip() for m in args.models.split(",")]

    print("\nModel evaluation on VQA-RAD:")
    print(f"{'Model':<16} {'F1':>8} {'BalAcc':>8} {'Acc':>8}")
    print("-" * 44)

    results = []
    for name in model_names:
        f1, balacc, acc = run_model(name, train_df, val_df, test_df, vocab_size, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        print(f"{name:<16} {f1:>8.4f} {balacc:>8.4f} {acc:>8.4f}")
        results.append({"model": name, "F1_macro": f1, "BalAcc": balacc, "Acc": acc})

    pd.DataFrame(results).to_csv("results_table5.csv", index=False)

    if args.mask_experiment:
        print("\nRobustness under image masking (HybridFusion):")
        print(f"{'Mask':>8} {'F1':>8} {'BalAcc':>8} {'Acc':>8}")
        print("-" * 36)

        mask_results = []
        for mp in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            f1, balacc, acc = run_model("HybridFusion", train_df, val_df, test_df, vocab_size, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, mask_p=mp)
            print(f"{mp:>8.1f} {f1:>8.4f} {balacc:>8.4f} {acc:>8.4f}")
            mask_results.append({"mask_p": mp, "F1_macro": f1, "BalAcc": balacc, "Acc": acc})

        pd.DataFrame(mask_results).to_csv("results_table6.csv", index=False)

if __name__ == "__main__":
    main()
