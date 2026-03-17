# Prompted Segmentation for Drywall 

Fine-tuning CLIPSeg to produce binary segmentation masks from natural language prompts.
Given an image and a text prompt, the model outputs a mask highlighting the described region.

Supported prompts:
- `"segment taping area"` — drywall joints and seams
- `"segment crack"` — surface cracks on walls, ceilings, concrete

---

## Setup
```bash
pip install torch torchvision transformers scikit-learn matplotlib tqdm pillow opencv-python
```

Tested on Python 3.12, PyTorch 2.x, macOS (Apple Silicon MPS).
The code automatically detects your device. MPS is used on Apple Silicon, 
CUDA on NVIDIA GPUs, and CPU as fallback. 
---

## Data

Two datasets from Roboflow, exported in COCO format:

| Dataset | Category | Samples | Annotation type |
|---|---|---|---|
| [Drywall-Join-Detect](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | taping | 1020 | Bounding box |
| [Cracks](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) | cracks | 5369 | Polygon segmentation |

Download both datasets in COCO segmentation format and place them as:
```
data/
├── taping/
│   └── train/
│       ├── _annotations.coco.json
│       └── *.jpg
└── cracks/
    └── train/
        ├── _annotations.coco.json
        └── *.jpg
```

**About taping annotations:** The taping dataset is originally an object detection dataset — 
annotations are bounding boxes, not polygon segmentations. Ground truth masks for taping 
are therefore rectangular approximations, not pixel-precise outlines. This is a data 
limitation that affects taping boundary quality.

---

## Project Structure
```
├── config.py       # All hyperparameters, paths, seeds, prompts
├── data.py         # Dataset loading, mask generation, augmentation
├── model.py        # CLIPSeg loading, encoder freezing, MPS patch
├── losses.py       # Focal loss + Dice loss
├── train.py        # Training loop, LR schedule, checkpointing
├── predict.py      # Inference, mask saving
├── report.py       # Visual examples, training curves
├── main.py         # End-to-end pipeline
└── show_taping.py  # Visualise taping results from saved checkpoint
```

---

## Reproducibility

All random seeds are fixed at the top of `config.py`:
```python
SEED = 42
random.seed(SEED)        # Python built-in — controls prompt sampling
np.random.seed(SEED)     # NumPy — controls train/val/test split
torch.manual_seed(SEED)  # PyTorch — controls weight init and dropout
```

The train/val/test split uses `random_state=SEED` in scikit-learn's `train_test_split`.
Running `python main.py` twice will produce identical results.

---

## Training
```bash
python main.py
```

This runs the full pipeline: data loading → splitting → training → evaluation → saving masks and report visuals.

---

## Approach

**Model:** CLIPSeg (`CIDAS/clipseg-rd64-refined`) — a text-conditioned segmentation model 
that takes an image and a text prompt and produces a segmentation mask.

Only the decoder is fine-tuned (1.1M parameters). The CLIP vision and text encoders are 
frozen to preserve pretrained representations. Training uses Focal + Dice loss with a 
3-epoch warmup and cosine LR decay. Full details in the report.

---

## Configuration

Key hyperparameters (all in `config.py`):

| Parameter | Value | Reason |
|---|---|---|
| `BATCH_SIZE` | 4 | Safe for MacBook unified memory with 603MB model |
| `NUM_EPOCHS` | 20 | Val loss converged cleanly by epoch 19 |
| `LR` | 5e-4 | Higher LR is safe since encoder is frozen |
| `WEIGHT_DECAY` | 1e-4 | Standard L2 regularisation for AdamW |
| `WARMUP_EPOCHS` | 3 | Protects pretrained weights at start of training |
| `THRESHOLD` | 0.5 | Neutral sigmoid cutoff for binary mask |
| `IMAGE_SIZE` | 352 | CLIPSeg's native output resolution |

---

## Results

**Data split:** Train 5111 | Val 639 | Test 639 (stratified, seed=42)

| Category | mIoU | Dice |
|---|---|---|
| Taping | 0.6441 | 0.7753 |
| Cracks | 0.5691 | 0.7060 |
| **Overall** | **0.6341** | **0.7725** |

Training converged in **81.9 minutes** on Apple Silicon (MPS).

| Metric | Value |
|---|---|
| Avg inference time | 41.8 ms/image |
| Model size (in-memory) | 603 MB |
| Device | Apple M-series (MPS) |
| Best checkpoint | Epoch 19 |

---

## Output Masks

Saved to `predictions/`. Each file is a single-channel PNG with values in {0, 255}.

Filename format: `{image_id}__{prompt}.png`

Examples:
```
42__segment_crack.png
137__segment_taping_area.png
```

---

## Checkpoint

The trained checkpoint is not included in this repo due to GitHub file size limits (600MB).
To reproduce results, run `python main.py` with seed=42 — training takes ~82 minutes on Apple Silicon.
