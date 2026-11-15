"""Export pinned Hugging Face ResNet-50 weights to the Triton model repository."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForImageClassification

from inference_lab.model import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_NAME,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_NAME,
)


MODEL_NAMES = ("classify_singleton", "classify_dynamic", "classify_bulk")


class LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def export_model(model_repo: Path) -> Path:
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).eval()
    wrapped = LogitsOnly(model).eval()
    example = torch.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
    batch = torch.export.Dim("batch", min=1, max=32)
    onnx_program = torch.onnx.export(
        wrapped,
        (example,),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={INPUT_NAME: {0: batch}},
        opset_version=18,
        dynamo=True,
        verify=True,
    )

    canonical_path = model_repo / "classify_dynamic" / "1" / "model.onnx"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_program.save(str(canonical_path))

    for model_name in MODEL_NAMES:
        destination = model_repo / model_name / "1" / "model.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination != canonical_path:
            shutil.copy2(canonical_path, destination)
    return canonical_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", type=Path, default=Path("models"))
    args = parser.parse_args()
    output = export_model(args.model_repo)
    print(f"Exported {MODEL_ID}@{MODEL_REVISION} to {output}")


if __name__ == "__main__":
    main()
