import time
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from config import (
    DEVICE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    WARMUP_EPOCHS, THRESHOLD, CHECKPOINT_DIR
)
from losses import SegLoss


def compute_iou_dice(logits, targets):
    preds  = (torch.sigmoid(logits) > THRESHOLD).float().flatten()
    target = (targets > THRESHOLD).float().flatten()

    inter = (preds * target).sum()
    union = preds.sum() + target.sum() - inter

    iou  = (inter + 1e-6) / (union + 1e-6)
    dice = (2 * inter + 1e-6) / (preds.sum() + target.sum() + 1e-6)
    return iou.item(), dice.item()


def get_lr(epoch, warmup_epochs, total_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def run_epoch(model, loader, criterion, optimizer, training: bool):
    model.train() if training else model.eval()

    total_loss = total_iou = total_dice = 0.0
    cat_scores = {"taping": {"iou": [], "dice": []}, "cracks": {"iou": [], "dice": []}}

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="  train" if training else "  eval ", leave=False):
            pv  = batch["pixel_values"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            am  = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)

            outputs = model(
                pixel_values=pv.contiguous(),
                input_ids=ids.contiguous(),
                attention_mask=am.contiguous(),
            )
            logits = outputs.logits

            if logits.shape[-2:] != lbl.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits.unsqueeze(1), size=lbl.shape[-2:],
                    mode="bilinear", align_corners=False
                ).squeeze(1)

            loss = criterion(logits, lbl)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            iou, dice = compute_iou_dice(logits.detach(), lbl.detach())
            total_loss += loss.item()
            total_iou  += iou
            total_dice += dice

            for i, cat in enumerate(batch["category"]):
                iou_i, dice_i = compute_iou_dice(logits[i:i+1].detach(), lbl[i:i+1].detach())
                cat_scores[cat]["iou"].append(iou_i)
                cat_scores[cat]["dice"].append(dice_i)

    n = len(loader)
    per_cat = {
        cat: {
            "iou":  float(np.mean(v["iou"])),
            "dice": float(np.mean(v["dice"])),
        }
        for cat, v in cat_scores.items() if v["iou"]
    }
    return {
        "loss": total_loss / n,
        "iou":  total_iou  / n,
        "dice": total_dice / n,
        "per_category": per_cat,
    }


def train(model, train_loader, val_loader):
    criterion = SegLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_iou = 0.0
    history      = {"train": [], "val": []}
    t0           = time.time()

    for epoch in range(NUM_EPOCHS):
        lr = get_lr(epoch, WARMUP_EPOCHS, NUM_EPOCHS, LR)
        for g in optimizer.param_groups:
            g["lr"] = lr

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}  lr={lr:.2e}")

        train_m = run_epoch(model, train_loader, criterion, optimizer, training=True)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, training=False)

        history["train"].append(train_m)
        history["val"].append(val_m)

        print(f"  train  loss={train_m['loss']:.4f}  iou={train_m['iou']:.4f}  dice={train_m['dice']:.4f}")
        print(f"  val    loss={val_m['loss']:.4f}  iou={val_m['iou']:.4f}  dice={val_m['dice']:.4f}")
        for cat, m in val_m.get("per_category", {}).items():
            print(f"    [{cat}]  iou={m['iou']:.4f}  dice={m['dice']:.4f}")

        if val_m["iou"] > best_val_iou:
            best_val_iou = val_m["iou"]
            torch.save({
                "epoch":      epoch + 1,
                "state_dict": model.state_dict(),
                "val_iou":    best_val_iou,
            }, CHECKPOINT_DIR / "best.pt")
            print(f"  ✓ saved  (val iou={best_val_iou:.4f})")

    elapsed = time.time() - t0
    print(f"\nDone — {elapsed / 60:.1f} min")
    return history, elapsed