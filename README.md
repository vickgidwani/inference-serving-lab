# Inference Serving Lab

A hands-on lab for a pretrained Hugging Face image model: export PyTorch ResNet-50 to ONNX, serve it with NVIDIA Triton Inference Server, move image tensor conversion/normalization into a custom CUDA kernel, and profile latency across preprocessing and serving contracts.

The model is [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50), a ResNet-50 v1.5 ImageNet-1k classifier with 1,000 output classes and an Apache 2.0 model card. The model and processor are pinned to Hub revision `34c2154f194f829b11125337b98c8f5f9965ff19` so the weights and preprocessing settings stay reproducible. The processor resizes/crops to 224×224 and applies ImageNet channel normalization; its exact parameters are loaded from the model card rather than duplicated in code. See the [pinned processor config](https://huggingface.co/microsoft/resnet-50/blob/34c2154f194f829b11125337b98c8f5f9965ff19/preprocessor_config.json).

## What you will practice

- Export pretrained weights using PyTorch's `torch.export`-based ONNX exporter.
- Measure image decode and Hugging Face preprocessing separately from ONNX Runtime and Triton inference.
- Serve one image per request, allow Triton to combine concurrent requests, or send an explicit batch.
- Compare request latency percentiles with Triton queue/compute metrics and per-request timestamps.
- Implement a CUDA kernel that fuses uint8 conversion, HWC-to-NCHW layout conversion, scaling, and per-channel normalization.
- Inspect CUDA launches with Nsight Systems and compare correctness against the Hugging Face processor.

## Layout

```text
src/inference_lab/       Pinned model, export, preprocessing, ORT/Triton clients
models/                  Triton configs; ONNX weights are generated locally
cuda/                    PyTorch C++/CUDA preprocessing extension and benchmark
exercises/               Guided experiments
artifacts/                Sample images, reports, and traces (ignored by Git)
```

## Input and preprocessing pipeline

The Triton model accepts normalized `FP32` tensors named `pixel_values` with shape `[N, 3, 224, 224]`; it returns `FP32` logits `[N, 1000]`. The local image path is:

```text
JPEG/PNG -> Pillow decode to RGB -> pinned Hugging Face spatial resize/crop
         -> CUDA uint8 HWC to normalized FP32 NCHW -> ONNX ResNet-50
```

The custom CUDA kernel implements the final conversion and normalization stage. Image decoding and the processor's spatial resize/crop stay in the Hugging Face processor, so the CUDA comparison preserves the model's exact image geometry. The benchmark reports CPU full preprocessing, CUDA end-to-end preprocessing (decode + spatial transform + upload + kernel), and CUDA kernel-only time separately. Extending the kernel to spatial resize/crop is an exercise after the fused normalization path is understood.

## Serving contracts

All three Triton models use the same normalized image tensor and output logits. They differ in who forms batches and how much scheduler delay/queueing is allowed.

| Model | Request shape | Triton policy | Intended workload |
| --- | --- | --- | --- |
| `classify_singleton` | Exactly `[1, 3, 224, 224]` | Maximum batch 1; no dynamic batching | Interactive caller with one image per request |
| `classify_dynamic` | `[N, 3, 224, 224]`, 1≤N≤32 | Server may combine requests; preferred batches 4/8/16/32; up to 2 ms batching delay; queue capped at 128 requests and rejects after 10 ms | Low request overhead at concurrent arrival rates |
| `classify_bulk` | `[N, 3, 224, 224]`, 1≤N≤32 | Client forms the batch; no server-side dynamic batching | Offline or throughput-oriented caller |

The image benchmark client accepts a local JPEG/PNG. By default it runs the pinned processor once and reuses that tensor; `--rows` repeats the same image to control request batch size. Set `--preprocess cpu` or `--preprocess cuda` to run preprocessing for every request and include it in the client latency. The CUDA option requires the extension below. The `--slo-ms` value is a measured latency target. `--client-timeout-ms` configures HTTP connection/network timeouts. Queue rejection and transport errors count as failed requests and SLO misses.

## Requirements

- Python 3.10 or newer for model export and clients.
- An NVIDIA GPU, recent NVIDIA driver, Docker with NVIDIA Container Toolkit, and Docker Compose for Triton and CUDA execution.
- A CUDA toolkit compatible with the installed CUDA-enabled PyTorch build for compiling the extension.

The server image is pinned to Triton release `26.09` (`nvcr.io/nvidia/tritonserver:26.09-py3`). NVIDIA lists Triton 2.73.0, ONNX Runtime 1.28.0, and CUDA 13.4.1 in that release. Check the [release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-09.html) and [compatibility matrix](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html) if you change the image or CUDA stack.

## Setup and run

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[export,client]'
mkdir -p artifacts
curl -L 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png' \
  -o artifacts/parrots.png
inference-lab-export --model-repo models
```

The export downloads the pinned model weights from Hugging Face and writes the same `model.onnx` into each Triton model version directory. Start Triton on the NVIDIA host:

```bash
docker compose up
```

The server exposes HTTP on port 8000, gRPC on 8001, and Prometheus metrics on 8002. Check readiness:

```bash
curl -s localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models/classify_dynamic/ready
```

### Run on a Colab GPU runtime

The Colab setup script installs the lab's Python extras, Docker Engine and Compose, and the NVIDIA Container Toolkit. It checks the GPU compute capability and host driver, starts Docker without assuming systemd is available, and writes a driver-compatible Triton image selection to the ignored `.env.colab` file. It does not install or replace the host NVIDIA driver.

Copy or clone this directory into the Colab VM, then run from `inference-serving-lab`:

```bash
bash scripts/setup_colab.sh
inference-lab-export --model-repo models
mkdir -p artifacts
curl -L 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png' \
  -o artifacts/parrots.png
docker compose --env-file .env.colab up
```

The script selects Triton `25.03-py3` for driver branches 570–579, `25.11-py3` for branches 580–614, and `26.09-py3` for branch 615 or newer. These images pair CUDA 12.8, 13.0, and 13.4 respectively; NVIDIA's CUDA compatibility guide lists the corresponding driver branches. The T4's compute capability is 7.5, supported by these Triton releases. If Colab blocks the Docker daemon or GPU passthrough, use a GPU VM with Docker and the NVIDIA Container Toolkit instead.

To use a different image, set `COLAB_TRITON_IMAGE` before running the setup script. That override is written to `.env.colab`; check that its CUDA runtime is compatible with the VM's driver.

### ONNX Runtime

Run inference on CPU or CUDA. `--profile` writes ONNX Runtime's per-operator trace; use an unprofiled run for final latency numbers because detailed profiling adds overhead.

```bash
inference-lab-bench-ort models/classify_dynamic/1/model.onnx \
  --image artifacts/parrots.png --provider cpu --batch-size 1 \
  --warmup 20 --requests 200 --profile \
  --output artifacts/ort-b1.json
```

For `--provider cuda`, install an ONNX Runtime GPU package compatible with the host CUDA stack. This benchmark measures inference only; image decode and preprocessing happen before the timed loop. It records the top five ImageNet labels to make the prediction easy to inspect.

### Triton serving contracts

```bash
inference-lab-bench-triton --image artifacts/parrots.png \
  --contract singleton --rows 1 --concurrency 1 --requests 500 \
  --output artifacts/triton-single.json

inference-lab-bench-triton --image artifacts/parrots.png \
  --contract dynamic --rows 1 --concurrency 32 --requests 2000 \
  --slo-ms 25 --output artifacts/triton-dynamic.json

inference-lab-bench-triton --image artifacts/parrots.png \
  --contract bulk --rows 16 --concurrency 4 --requests 250 \
  --output artifacts/triton-bulk.json

inference-lab-bench-triton --image artifacts/parrots.png \
  --contract dynamic --preprocess cuda --rows 1 \
  --concurrency 16 --requests 500 --slo-ms 40 \
  --output artifacts/triton-e2e-cuda.json
```

Sweep request concurrency, rows per request, preprocessing mode, and `max_queue_delay_microseconds` in `models/classify_dynamic/config.pbtxt`. Reports include successful requests/s and images/s, p50/p95/p99/max latency, SLO misses, failures, and top classes. The load generator is closed-loop: each worker issues its next request after the previous one completes, so it does not model a fixed arrival rate.

Inspect Triton's queue and compute aggregates alongside the client report:

```bash
curl -s localhost:8002/metrics | rg 'nv_inference_(request|queue|compute_input|compute_infer|compute_output)_duration_us'
```

For per-request server timestamps, stop the normal server and start a short trace run:

```bash
docker compose stop
docker compose run --rm --service-ports triton \
  tritonserver --model-repository=/models \
  --trace-config triton,file=/artifacts/triton-trace.json \
  --trace-config level=TIMESTAMPS \
  --trace-config rate=1 --trace-config count=100
```

Run a short client load in another terminal, then stop the trace container to flush the file. Tracing adds overhead; use it to inspect request/queue/compute timing and use an untraced run for final percentiles. See [Triton tracing](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/trace.html).

### Custom CUDA preprocessing

Install a CUDA-enabled PyTorch build, then compile the extension from this directory:

```bash
python -m pip install --no-build-isolation ./cuda
python cuda/bench_preprocess.py --image artifacts/parrots.png \
  --warmup 10 --samples 100 --repeats 20 \
  --output artifacts/preprocess.json
```

The script checks the custom kernel against Hugging Face's normalized output before measuring. CUDA end-to-end time includes image decoding, the same spatial resize/crop, host-to-device copy, and the custom conversion kernel. Kernel-only timing uses CUDA events on an already uploaded image.

Capture a short Nsight Systems trace separately:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output=artifacts/preprocess \
  python cuda/bench_preprocess.py --image artifacts/parrots.png \
    --warmup 2 --samples 10 --repeats 2
```

Profiling adds overhead. Use traces to inspect copies and launches; use unprofiled runs for latency comparisons.

## Exercises

Start with [the exercise index](exercises/README.md). It covers model export, image preprocessing, serving contracts, queue/compute profiling, and a CUDA spatial-transform extension.

Lip reading is a useful later extension after the image pipeline is working. A [visual-only Auto-AVSR checkpoint](https://huggingface.co/simonlesaumon/lrs3-lipreader-visual-only) documents 88×88 grayscale mouth crops at 25 fps and a research/non-commercial license. That version adds face/mouth tracking, temporal sampling, variable-length clips, and CTC decoding, which makes it a good sequence-serving follow-up.

## References

- [Hugging Face ResNet-50 model card](https://huggingface.co/microsoft/resnet-50)
- [Pinned image processor configuration](https://huggingface.co/microsoft/resnet-50/blob/34c2154f194f829b11125337b98c8f5f9965ff19/preprocessor_config.json)
- [Hugging Face image processor docs](https://huggingface.co/docs/transformers/main/image_processors)
- [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx_export.html)
- [Triton model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [Triton HTTP inference protocol](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/README.html)
- [ONNX Runtime profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
