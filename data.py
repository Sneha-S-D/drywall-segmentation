import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

from config import IMAGE_SIZE, PROMPTS, TAPING_DIR, CRACKS_DIR


def bbox_to_mask(bbox, height, width):
    x, y, bw, bh = [int(v) for v in bbox]
    mask = np.zeros((height, width), dtype=np.uint8)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width,  x + bw)
    y2 = min(height, y + bh)
    mask[y1:y2, x1:x2] = 255
    return mask


def polygons_to_mask(polygons, height, width):
    canvas = Image.new("L", (width, height), 0)
    draw   = ImageDraw.Draw(canvas)
    for poly in polygons:
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.array(canvas)


def load_coco_split(annotation_file: Path, image_dir: Path) -> List[Dict]:
    with open(annotation_file) as f:
        coco = json.load(f)

    id_to_image = {img["id"]: img for img in coco["images"]}

    grouped = {}
    for ann in coco["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)

    samples = []
    for img_id, img_info in id_to_image.items():
        img_path = image_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        h, w   = img_info["height"], img_info["width"]
        anns   = grouped.get(img_id, [])
        if not anns:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list) and seg:
                m = polygons_to_mask(seg, h, w)
            else:
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    continue
                m = bbox_to_mask(bbox, h, w)
            mask = np.maximum(mask, m)

        if mask.max() == 0:
            continue

        samples.append({
            "image_path": str(img_path),
            "mask":       mask,
            "image_id":   img_id,
        })

    return samples


def load_all_samples() -> List[Dict]:
    all_samples = []
    for category, base_dir in [("taping", TAPING_DIR), ("cracks", CRACKS_DIR)]:
        for split_name in ["train", "valid", "test", ""]:
            d   = base_dir / split_name if split_name else base_dir
            ann = d / "_annotations.coco.json"
            if not ann.exists():
                continue
            samples = load_coco_split(ann, d)
            for s in samples:
                s["category"] = category
            all_samples.extend(samples)
            print(f"  {len(samples):4d} samples  [{category} / {split_name or 'root'}]")

    return all_samples


def augment_pair(image: Image.Image, mask: Image.Image):
    # Random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    # Random vertical flip
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask  = TF.vflip(mask)

    # Random rotation ±15°
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask, angle)

    # Random crop and resize back
    if random.random() > 0.5:
        w, h  = image.size
        scale = random.uniform(0.7, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        image = TF.resized_crop(image, y, x, ch, cw, (h, w))
        mask  = TF.resized_crop(mask,  y, x, ch, cw, (h, w), interpolation=Image.NEAREST)

    # Color jitter on image only — mask is unaffected
    if random.random() > 0.5:
        image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
        image = TF.adjust_contrast(image,   random.uniform(0.7, 1.3))
        image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

    return image, mask


class DrywallDataset(Dataset):
    def __init__(self, samples: List[Dict], processor, augment: bool = False):
        self.samples   = samples
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image = Image.open(s["image_path"]).convert("RGB")
        mask  = Image.fromarray(s["mask"])

        if self.augment:
            image, mask = augment_pair(image, mask)

        prompt = random.choice(PROMPTS[s["category"]])

        enc = self.processor(
            text=[prompt],
            images=[image],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        mask_resized = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        label = torch.tensor(np.array(mask_resized) / 255.0, dtype=torch.float32)

        return {
            "pixel_values":   enc["pixel_values"].squeeze(0),
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         label,
            "prompt":         prompt,
            "category":       s["category"],
            "image_path":     s["image_path"],
            "image_id":       s["image_id"],
        }