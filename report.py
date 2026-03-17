import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict

from config import REPORT_DIR, PROMPTS
from predict import predict_mask


def plot_examples(model, processor, test_samples: List[Dict], n: int = 4):
    samples = random.sample(test_samples, min(n, len(test_samples)))
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.5 * n))
    fig.suptitle("Original  |  Ground Truth  |  Prediction", fontsize=14, y=1.01)

    for row, s in enumerate(samples):
        image  = Image.open(s["image_path"]).convert("RGB")
        gt     = s["mask"]
        prompt = PROMPTS[s["category"]][0]
        pred   = predict_mask(model, processor, s["image_path"], prompt)

        p_b = (pred > 127).astype(np.float32)
        g_b = (gt   > 127).astype(np.float32)
        inter = (p_b * g_b).sum()
        union = p_b.sum() + g_b.sum() - inter
        iou   = inter / (union + 1e-6)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"[{s['category']}]", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].set_title("Ground Truth", fontsize=11)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 2].set_title(f'Pred  IoU={iou:.3f}\n"{prompt}"', fontsize=11)
        axes[row, 2].axis("off")

    plt.tight_layout()
    out = REPORT_DIR / "examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


def plot_curves(history: Dict):
    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key in zip(axes, ["loss", "iou", "dice"]):
        ax.plot(epochs, [h[key] for h in history["train"]], label="Train", marker="o", ms=3)
        ax.plot(epochs, [h[key] for h in history["val"]],   label="Val",   marker="s", ms=3)
        ax.set_title(key.upper())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training History")
    plt.tight_layout()
    out = REPORT_DIR / "curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")