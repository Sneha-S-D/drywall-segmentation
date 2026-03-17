import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import config
from config import (
    DEVICE, SEED, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT,
    CHECKPOINT_DIR
)
from data    import load_all_samples, DrywallDataset
from model   import load_model
from train   import train, run_epoch
from predict import save_predictions
from report  import plot_examples, plot_curves
from losses  import SegLoss


def main():
    print("=" * 55)
    print("  Drywall QA — Prompted Segmentation")
    print(f"  Device : {DEVICE}   Seed : {SEED}")
    print("=" * 55)

    # 1. Data
    print("\n[1/7] Loading data...")
    samples = load_all_samples()
    print(f"\n  Total : {len(samples)}")
    for cat in ["taping", "cracks"]:
        print(f"    {cat}: {sum(1 for s in samples if s['category'] == cat)}")

    # 2. Split 
    print("\n[2/7] Splitting...")
    labels = [s["category"] for s in samples]
    train_val, test_set = train_test_split(
        samples, test_size=TEST_SPLIT, stratify=labels, random_state=SEED
    )
    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    train_set, val_set = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=[s["category"] for s in train_val],
        random_state=SEED,
    )
    print(f"  Train {len(train_set)} | Val {len(val_set)} | Test {len(test_set)}")

    # 3. Model
    print(f"\n[3/7] Loading {config.MODEL_NAME}...")
    model, processor = load_model()

    # 4. Loaders 
    print("\n[4/7] Building dataloaders...")
    def make_loader(split, augment, shuffle):
        ds = DrywallDataset(split, processor, augment=augment)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_set, augment=True,  shuffle=True)
    val_loader   = make_loader(val_set,   augment=False, shuffle=False)
    test_loader  = make_loader(test_set,  augment=False, shuffle=False)

    # 5. Train
    print("\n[5/7] Training...")
    history, train_time = train(model, train_loader, val_loader)

    # Reload best checkpoint
    ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\n  Best checkpoint: epoch {ckpt['epoch']}  val iou={ckpt['val_iou']:.4f}")

    # 6. Test evaluation
    print("\n[6/7] Test set evaluation...")
    criterion   = SegLoss().to(DEVICE)
    test_metrics = run_epoch(model, test_loader, criterion, None, training=False)

    print("\n" + "=" * 55)
    print("  TEST RESULTS")
    print("=" * 55)
    print(f"  mIoU  : {test_metrics['iou']:.4f}")
    print(f"  Dice  : {test_metrics['dice']:.4f}")
    for cat, m in test_metrics.get("per_category", {}).items():
        print(f"  [{cat}]  mIoU={m['iou']:.4f}  Dice={m['dice']:.4f}")

    # 7. Outputs
    print("\n[7/7] Saving predictions and report...")
    avg_inf = save_predictions(model, processor, test_set)
    plot_examples(model, processor, test_set, n=4)
    plot_curves(history)

    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print("\n" + "=" * 55)
    print("  RUNTIME")
    print("=" * 55)
    print(f"  Train time        : {train_time / 60:.1f} min")
    print(f"  Avg inference     : {avg_inf * 1000:.1f} ms/image")
    print(f"  Model (in-memory) : {model_mb:.0f} MB")
    print(f"  Device            : {DEVICE}")
    print("=" * 55)


if __name__ == "__main__":
    main()