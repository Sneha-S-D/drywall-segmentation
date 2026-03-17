import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import DEVICE, SEED, CHECKPOINT_DIR, REPORT_DIR, PROMPTS
from data import load_all_samples
from model import load_model
from predict import predict_mask
from sklearn.model_selection import train_test_split

random.seed(SEED)
np.random.seed(SEED)

samples = load_all_samples()
labels  = [s["category"] for s in samples]

train_val, test_set = train_test_split(
    samples, test_size=0.10, stratify=labels, random_state=SEED
)
val_ratio = 0.10 / (1 - 0.10)
_, val_set = train_test_split(
    train_val,
    test_size=val_ratio,
    stratify=[s["category"] for s in train_val],
    random_state=SEED,
)

taping_test = [s for s in test_set if s["category"] == "taping"]
print(f"Taping test samples available: {len(taping_test)}")

model, processor = load_model()
ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}  val iou={ckpt['val_iou']:.4f}")

chosen = random.sample(taping_test, min(4, len(taping_test)))
n      = len(chosen)
prompt = PROMPTS["taping"][0]   

fig, axes = plt.subplots(n, 3, figsize=(13, 4.5 * n))
fig.suptitle("Taping — Original  |  Ground Truth  |  Prediction", fontsize=14, y=1.01)

for row, s in enumerate(chosen):
    image = Image.open(s["image_path"]).convert("RGB")
    gt    = s["mask"]
    pred  = predict_mask(model, processor, s["image_path"], prompt)

    p_b   = (pred > 127).astype(np.float32)
    g_b   = (gt   > 127).astype(np.float32)
    inter = (p_b * g_b).sum()
    union = p_b.sum() + g_b.sum() - inter
    iou   = inter / (union + 1e-6)

    axes[row, 0].imshow(image)
    axes[row, 0].set_title("[taping]", fontsize=11)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(gt, cmap="gray")
    axes[row, 1].set_title("Ground Truth", fontsize=11)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(pred, cmap="gray")
    axes[row, 2].set_title(f'Pred  IoU={iou:.3f}\n"{prompt}"', fontsize=11)
    axes[row, 2].axis("off")

plt.tight_layout()
out = REPORT_DIR / "taping_examples.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out}")