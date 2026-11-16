"""Run the custom CUDA preprocessing output directly through ORT's CUDA EP."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch  # Load the CUDA/cuDNN libraries before ORT creates its session.
import onnxruntime as ort
import image_preprocess_cuda
from PIL import Image

from inference_lab.model import INPUT_NAME, OUTPUT_NAME
from inference_lab.vision import load_image_processor, load_labels, preprocess_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/classify_dynamic/1/model.onnx"))
    parser.add_argument("--image", type=Path, default=Path("artifacts/parrots.png"))
    parser.add_argument("--profile", action="store_true", help="Report the execution provider used by each node")
    parser.add_argument("--disable-tf32", action="store_true", help="Disable CUDA TF32 to compare numerical error")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA GPU")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable in this ONNX Runtime install")

    # Decode remains on the CPU; upload the RGB bytes once, then keep preprocessing on CUDA.
    with Image.open(args.image) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    if rgb.shape != (512, 768, 3):
        raise ValueError(
            f"This exercise uses the recorded parrots.png geometry (512, 768, 3); got {rgb.shape}"
        )

    device_id = torch.cuda.current_device()
    rgb_cuda = torch.from_numpy(rgb).to(device=f"cuda:{device_id}")
    # For this image, Pillow resizes 768x512 to 384x256, then center-crops at (left=80, top=16).
    spatial_cuda = image_preprocess_cuda.resize_crop_rgb(rgb_cuda, 256, 384, 16, 80)
    pixel_values_cuda = image_preprocess_cuda.rgb_to_normalized(
        spatial_cuda,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )

    # The custom kernels use PyTorch's current stream. Giving ORT that same stream
    # preserves producer-before-consumer ordering without a device-wide sync here.
    stream = torch.cuda.current_stream(device_id)
    provider_options = {
        "device_id": device_id,
        "user_compute_stream": str(stream.cuda_stream),
    }
    if args.disable_tf32:
        provider_options["use_tf32"] = 0

    session_options = ort.SessionOptions()
    if args.profile:
        Path("artifacts").mkdir(parents=True, exist_ok=True)
        session_options.enable_profiling = True
        session_options.profile_file_prefix = "artifacts/ort-cuda-iobinding"

    session = ort.InferenceSession(
        str(args.model),
        sess_options=session_options,
        providers=[
            ("CUDAExecutionProvider", provider_options),
            "CPUExecutionProvider",
        ],
    )

    # I/O Binding gives ORT the CUDA tensor's address and its tensor metadata.
    # Keep pixel_values_cuda alive until run_with_iobinding has completed.
    binding = session.io_binding()
    binding.bind_input(
        name=INPUT_NAME,
        device_type="cuda",
        device_id=device_id,
        element_type=np.float32,
        shape=tuple(pixel_values_cuda.shape),
        buffer_ptr=pixel_values_cuda.data_ptr(),
    )
    # Let ORT allocate its output on the GPU too.
    binding.bind_output(OUTPUT_NAME, device_type="cuda", device_id=device_id)
    session.run_with_iobinding(binding)
    profile_path = session.end_profiling() if args.profile else None

    # This copies only the small logits output [1, 1000] to CPU for comparison.
    cuda_logits = binding.copy_outputs_to_cpu()[0]

    # CPU reference: Hugging Face preprocessing and ORT's CPU execution provider.
    processor = load_image_processor()
    cpu_input = preprocess_path(args.image, processor)
    cpu_session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cpu_logits = cpu_session.run([OUTPUT_NAME], {INPUT_NAME: cpu_input})[0]

    labels = load_labels()
    cuda_top5 = np.argsort(cuda_logits[0])[-5:][::-1]
    cpu_top5 = np.argsort(cpu_logits[0])[-5:][::-1]

    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"CUDA session providers: {session.get_providers()}")
    print(f"CUDA TF32: {'disabled' if args.disable_tf32 else 'enabled (ORT default)'}")
    print(
        "Bound GPU input:"
        f" shape={tuple(pixel_values_cuda.shape)},"
        f" dtype={pixel_values_cuda.dtype},"
        f" device={pixel_values_cuda.device}"
    )
    print(f"Logit shape: {cuda_logits.shape}; max_abs_logit_error: {np.max(np.abs(cuda_logits - cpu_logits)):.8g}")
    print("CUDA top 5:", [(int(i), labels[int(i)]) for i in cuda_top5])
    print("CPU top 5: ", [(int(i), labels[int(i)]) for i in cpu_top5])
    if profile_path is not None:
        with Path(profile_path).open(encoding="utf-8") as profile_file:
            events = json.load(profile_file)
        providers = Counter(
            event["args"]["provider"]
            for event in events
            if event.get("cat") == "Node" and event.get("args", {}).get("provider")
        )
        print(f"Profile node counts by provider: {dict(providers)}")
        print(f"Profile trace: {profile_path}")


if __name__ == "__main__":
    main()
