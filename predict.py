import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

from config import DEVICE, THRESHOLD, OUTPUT_DIR, PROMPTS


@torch.no_grad()
def predict_mask(model, processor, image_path: str, prompt: str) -> np.ndarray:
    model.eval()
    image        = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    enc = processor(
        text=[prompt], images=[image],
        padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    )
    pv  = enc["pixel_values"].to(DEVICE)
    ids = enc["input_ids"].to(DEVICE)
    am  = enc["attention_mask"].to(DEVICE)

    logits = model(
        pixel_values=pv.contiguous(),
        input_ids=ids.contiguous(),
        attention_mask=am.contiguous(),
    ).logits

    logits_up = nn.functional.interpolate(
        logits.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    binary = (torch.sigmoid(logits_up) > THRESHOLD).cpu().numpy().astype(np.uint8) * 255
    return binary


def save_predictions(model, processor, test_samples: List[Dict]) -> float:
    times = []
    for s in tqdm(test_samples, desc="Saving masks"):
        prompt = PROMPTS[s["category"]][0]
        t0     = time.time()
        mask   = predict_mask(model, processor, s["image_path"], prompt)
        times.append(time.time() - t0)

        slug  = prompt.replace(" ", "_")
        fname = f"{s['image_id']}__{slug}.png"
        Image.fromarray(mask).save(OUTPUT_DIR / fname)

    avg = float(np.mean(times))
    print(f"Saved {len(test_samples)} masks → {OUTPUT_DIR}/")
    print(f"Avg inference: {avg * 1000:.1f} ms/image")
    return avg


@torch.no_grad()
def evaluate_test_set(model, loader, criterion) -> Dict:
    from train import run_epoch
    return run_epoch(model, loader, criterion, None, training=False)