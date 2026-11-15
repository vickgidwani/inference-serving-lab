"""Image decoding and model-card preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor

from inference_lab.model import MODEL_ID, MODEL_REVISION


def load_image_processor() -> AutoImageProcessor:
    return AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def load_labels() -> dict[int, str]:
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return {int(index): label for index, label in config.id2label.items()}


def decode_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def preprocess_path(path: Path, processor: AutoImageProcessor | None = None) -> np.ndarray:
    image_processor = processor or load_image_processor()
    image = decode_rgb(path)
    pixels = image_processor(image, return_tensors="np")["pixel_values"]
    return np.asarray(pixels, dtype=np.float32)


def spatial_uint8_path(path: Path, processor: AutoImageProcessor) -> np.ndarray:
    """Run the processor's spatial steps, leaving rescale/normalize for CUDA."""
    image = decode_rgb(path)
    pixels = processor(
        image,
        do_rescale=False,
        do_normalize=False,
        return_tensors="np",
    )["pixel_values"][0]
    pixels = np.asarray(pixels)
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError(f"expected processor output [3,H,W], got {pixels.shape}")
    pixels = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(pixels.transpose(1, 2, 0))
