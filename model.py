import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from config import MODEL_NAME, DEVICE


def load_model():
    processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
    model     = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)

    # Freeze the CLIP vision + text encoders.
    for name, param in model.named_parameters():
        if "decoder" not in name:
            param.requires_grad = False

    model = model.to(DEVICE)
    _patch_mps(model)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters  total: {total / 1e6:.1f}M   trainable: {trainable / 1e6:.1f}M")

    return model, processor


def _patch_mps(model):
    orig = model.decoder.transposed_convolution.forward

    def patched(x):
        return orig(x.contiguous())

    model.decoder.transposed_convolution.forward = patched