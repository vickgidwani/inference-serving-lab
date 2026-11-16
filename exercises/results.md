# Recorded results

Use this file for concise, reproducible experiment summaries. Selected machine-readable reports are stored in [`results/`](results/); temporary output under the ignored top-level `artifacts/` directory is not tracked.

## Environment

- Host: GCP `g2-standard-4` in `us-central1-c`; NVIDIA L4; driver 580.
- PyTorch: `2.15.0+cu132`.
- Pillow: `13.0.0` (used by the pinned Hugging Face PIL image processor).
- CUDA toolkit used to build the preprocessing extension: `nvcc 13.2.86`.
- Model and processor: `microsoft/resnet-50`, revision `34c2154c194f829b11125337b98c8f5f9965ff19`.
- Sample: `artifacts/parrots.png`.

## Exercise 1: export and inspect

- Exported ONNX opset 18. Input is `pixel_values` with shape `[batch, 3, 224, 224]`; output is `logits` with shape `[batch, 1000]`.
- Hugging Face PyTorch and CPU ONNX Runtime produced the same top-five class IDs and labels, in the same order.

## Exercise 2: preprocessing fidelity

### Baseline comparison

Command:

```bash
python cuda/bench_preprocess.py --image artifacts/parrots.png \
  --warmup 10 --samples 100 --repeats 20 \
  --output artifacts/preprocess.json
```

The CUDA preprocessing output matched Hugging Face exactly for this image (`max_absolute_error: 0.0`, shape `[1, 3, 224, 224]`). The copied raw report is [`results/exercise-2-preprocess-l4.json`](results/exercise-2-preprocess-l4.json).

| Path | p50 | p95 | Max |
|---|---:|---:|---:|
| CPU decode + full processor | 15.179 ms | 15.460 ms | 16.502 ms |
| CPU spatial steps + CUDA normalization, end to end | 17.638 ms | 17.792 ms | 18.120 ms |
| CUDA kernel only | 9.062 µs | 9.884 µs | 12.851 µs |

The raw report currently names the final field `cuda_kernel_only_ms`, but its values are microseconds: the timing code multiplies CUDA event milliseconds by 1,000. The end-to-end CUDA path was about 2.46 ms slower at p50 than the CPU path. Exercise 3 measured H2D + D2H at about 0.275 ms and the kernel at about 0.009 ms, so those stages explain only part of the gap.

### Normalization perturbation

Using CPU ONNX Runtime with the same image and model, the red-channel mean was increased from `0.485` to `0.535`, with red standard deviation unchanged at `0.229`. This shifted every normalized red pixel down by about `0.218341` and changed the logits by up to `1.015862`.

| Rank | Baseline | Shifted mean |
|---:|---|---|
| 1 | 88 macaw: `10.98539` | 88 macaw: `11.02827` |
| 2 | 23 vulture: `-4.36369` | 23 vulture: `-5.37955` |
| 3 | 89 sulphur-crested cockatoo: `-6.56798` | 503 cocktail shaker: `-6.45665` |
| 4 | 503 cocktail shaker: `-6.82568` | 89 sulphur-crested cockatoo: `-6.54244` |
| 5 | 96 toucan: `-7.31146` | 96 toucan: `-7.19740` |

Top-1 stayed macaw for this sample, while classes 89 and 503 swapped ranks. This is a one-image experiment; it shows that preprocessing changes affect logits even when the top prediction remains unchanged.

## Exercise 3: preprocess cost model

**Attribution status: partially explained, not fully resolved.** The rough accounting below separates the CPU preparation difference from measured transfers and kernel time, but the independent measurements do not isolate every component. Revisit this if we need to optimize the preprocessing path.

On the VM, Transformers `5.23.0` loaded the checkpoint's image processor as `ConvNextImageProcessorPil` (processor implementation and model architecture are separate). Both paths use the same `decode_rgb()` helper and the same pinned Hugging Face image processor for spatial processing. Its config uses bicubic resampling, resizes the shortest edge to 256 pixels (`224 / crop_pct 0.875`), then center-crops to 224×224. The CPU baseline also asks that processor to rescale and normalize to NCHW float32. The CUDA path disables only rescaling and normalization in the processor, then the lab code converts its CHW spatial output to contiguous HWC `uint8` for the custom kernel.

| Stage | p50 | p95 | Max |
|---|---:|---:|---:|
| CPU image decode | 4.445 ms | 4.558 ms | 4.656 ms |
| CPU spatial resize/crop | 8.893 ms | 9.037 ms | 9.422 ms |
| CPU HWC `uint8` packing | 3.892 ms | 3.938 ms | 4.005 ms |
| CPU spatial resize/crop + HWC `uint8` packing (combined run) | 12.639 ms | 12.803 ms | 12.967 ms |
| H2D copy (224×224×3 `uint8`) | 0.067 ms | 0.078 ms | 0.090 ms |
| CUDA normalization/layout kernel | 9.062 µs | 9.884 µs | 12.851 µs |
| D2H copy (1×3×224×224 `float32`) | 0.208 ms | 0.253 ms | 0.269 ms |

Previously measured end-to-end medians are 15.179 ms for CPU preprocessing and 17.638 ms for CPU spatial preprocessing plus CUDA normalization, a 2.459 ms gap. H2D and D2H together are about 0.275 ms at p50, and the kernel is about 0.009 ms. The transfer and kernel times therefore explain only about 0.284 ms of the gap.

An approximate accounting helps locate the rest. Decode plus the combined CPU spatial-and-packing measurement is 4.445 + 12.639 = 17.084 ms. Adding the transfers and kernel gives about 17.368 ms, around 0.270 ms below the measured end-to-end median. Using the separately timed spatial and packing medians instead gives 4.445 + 8.893 + 3.892 + 0.275 + 0.009 = 17.514 ms, around 0.124 ms below it. These independently measured p50s are not perfectly additive, so the residual is an estimate, not an isolated component. The arithmetic suggests the custom path's CPU portion may be roughly 1.9–2.1 ms slower than the CPU baseline (17.084–17.230 ms versus 15.179 ms). This is a rough net comparison, not a direct measurement of the delta or a cost assigned to the shared decode/resize/crop steps. Those shared steps use the same helpers and processor; the paths differ in post-spatial conversion and where rescaling/normalization happen.

That difference is not attributable to one measured operation yet. The CUDA route clips, rounds, casts the processor's spatial output to `uint8`, transposes CHW to HWC, and makes it contiguous; that HWC conversion is measured at about 3.9 ms. The CPU baseline also packages its output, but as normalized NCHW float32. So 3.9 ms is the cost of the custom path's conversion, not a measured 3.9 ms net penalty versus Hugging Face's output conversion. Comparing separate medians cannot tell us the net cost of each differing operation. The end-to-end CUDA timing also includes Python/PyTorch dispatch, tensor and output allocations, CUDA launch overhead, and synchronization. In this benchmark, the output is copied back to CPU for comparison; that D2H copy can be avoided only when a downstream GPU stage consumes the tensor directly. An external Triton client may still need to transfer image data through host memory. The kernel-only CUDA-event timing measures device execution, not the full host launch and wrapper overhead.

For this 224×224 workload, the end-to-end CUDA path is slower than the CPU path. This does not show that CUDA normalization itself is slower than CPU normalization: the benchmark compares two complete, different preprocessing paths, and the CPU-side preparation and benchmark's D2H copy are included in the CUDA path. If we return to this, a quick first check is to compare wall time with and without the explicit `torch.cuda.synchronize()` inside `custom_cuda_path()`; `.cpu().numpy()` already waits for the D2H copy. Then we can separately measure a GPU-resident output path and the CPU output conversions.

## Exercise 4: serving contracts

### Singleton sequential baseline

Command:

```bash
inference-lab-bench-triton --image artifacts/parrots.png \
  --contract singleton --rows 1 --concurrency 1 --requests 500 \
  --output artifacts/triton-single.json
```

| Requests | Concurrency | Rows/request | p50 | p95 | p99 | Max | Requests/s | Images/s | Failures / 20 ms misses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1 | 1 | 3.656 ms | 3.787 ms | 3.991 ms | 10.166 ms | 269.29 | 269.29 | 0 / 0 |

All requests succeeded and the top-five predictions matched the earlier model checks. Rates match because each request carried one image. Preprocessing was precomputed outside the timed loop; measured client latency still includes request construction/serialization, the HTTP round trip, and response handling. The raw report copied from the VM is [`exercise-4-singleton-l4.json`](results/exercise-4-singleton-l4.json).

### Bulk client batch: 16 images/request

| Requests | Concurrency | Rows/request | p50 | p95 | p99 | Max | Requests/s | Images/s | Failures / 20 ms misses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 4 | 16 | 79.578 ms | 83.548 ms | 134.196 ms | 163.955 ms | 49.07 | 785.11 | 0 / 250 |

All 250 requests succeeded, carrying 4,000 image rows total (the same sample repeated 16 times per request). The 20 ms deadline marked every request as late; it did not cause failures. Bulk achieved about 9% more images/s than the dynamic contract at concurrency 8 and 8 ms delay (785 vs. 718), with much higher per-request latency (p50 79.6 vs. 10.2 ms). The raw report is [`exercise-4-bulk-b16-c4-l4.json`](results/exercise-4-bulk-b16-c4-l4.json).

### Dynamic batching under concurrency

| Attempted | Concurrency | Rows/request | Successful | Failed | p50 of successes | p95 of successes | Requests/s (successes) | SLO misses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 32 | 1 | 217 | 1,783 | 108.968 ms | 166.168 ms | 32.17 | 1,976 (98.8%) |

The first failure was HTTP 503 `Request timeout expired`, consistent with the dynamic model's queue policy: reject requests that wait in the queue beyond 10 ms. The SLO was 20 ms; it did not cancel requests. The reported latency percentiles cover only the 217 successful responses, so they are survivor-only statistics. The success rate and successful throughput are both far below the singleton baseline; this run is an overload/rejection result, not evidence of a batching throughput gain. The raw report copied from the VM is [`exercise-4-dynamic-c32-l4.json`](results/exercise-4-dynamic-c32-l4.json).

The model statistics endpoint reported this execution distribution:

| Batch size | All-run executions | Measured-run executions (subtracting 50 sequential warmups) |
|---:|---:|---:|
| 1 | 64 | 14 |
| 2 | 52 | 52 |
| 3 | 29 | 29 |
| 4 | 3 | 3 |
| 8 or larger | 0 | 0 |

The total average is `267 / 148 = 1.80` images per execution. Removing the 50 singleton warmups gives `217 / 98 = 2.21` for the measured run. Most measured executions had batch size 2 or 3, and only three reached the preferred size 4. Triton sends the available smaller batch when the 2 ms collection delay expires without reaching a preferred size; the preferred sizes are targets, not a minimum. This distribution is consistent with requests arriving too spread out for the configured 2 ms window. See [Triton's batching guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).

Triton's cumulative server metrics after this run showed 267 successes (the 217 measured successes plus 50 warmups), 1,783 scheduler rejections, and zero cancellations or backend failures. Dividing the cumulative duration counters by 267 gives approximate per-success averages; these are server-side averages, not percentiles, and include warmups.

| Server metric | Cumulative total | Approx. average per success |
|---|---:|---:|
| Request duration | 15,162,854 µs | 56.79 ms |
| Queue duration | 490,522 µs | 1.84 ms |
| Compute input | 114,318 µs | 0.43 ms |
| Compute infer | 14,534,962 µs | 54.44 ms |
| Compute output | 8,323 µs | 0.03 ms |

For accepted requests, model execution dominates the server-side duration; requests that exceeded the scheduler's queue timeout were rejected and are counted separately. See [Triton's metric definitions](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html) for request, queue, compute, and batching count semantics.

## Exercise 5: queue delay and concurrency

### 0 µs queue delay at concurrency 32

| Queue delay | Successful | Rejected | Requests/s (successes) | p50 success latency | p95 success latency | p99 success latency | SLO misses |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 ms | 152 | 1,848 | 23.29 | 103.751 ms | 161.347 ms | 169.475 ms | 1,985 (99.25%) |

The stats endpoint reported 202 successes (152 measured plus 50 warmups) and 164 executions. Removing the 50 singleton warmups gives measured batches of size 1: 81, size 2: 28, and size 3: 5, for an average of `152 / 114 = 1.33`. A zero delay still permits batching when multiple requests are already queued as an execution becomes available, but it does not intentionally wait for more arrivals. The copied report is [`exercise-5-dynamic-c32-delay0us-l4.json`](results/exercise-5-dynamic-c32-delay0us-l4.json).

### 500 µs queue delay at concurrency 32

| Queue delay | Successful | Rejected | Requests/s (successes) | p50 success latency | p95 success latency | p99 success latency | SLO misses |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 ms | 145 | 1,855 | 21.38 | 108.415 ms | 159.258 ms | 168.741 ms | 1,984 (99.2%) |

The stats endpoint reported 195 successes (145 measured plus 50 warmups) and 152 executions. After removing the 50 sequential warmups, measured batch executions were size 1: 65, size 2: 32, size 3: 4, and size 4: 1. The average measured batch size was `145 / 102 = 1.42`. This is worse than the 2 ms run's 2.21 average and shows that 500 µs is too short for this arrival pattern. The copied benchmark report is [`exercise-5-dynamic-c32-delay500us-l4.json`](results/exercise-5-dynamic-c32-delay500us-l4.json).

### 5 ms queue delay at concurrency 32

This run kept the dynamic model, one image per request, concurrency 32, 2,000 attempts, and 10 ms queue timeout, changing only `max_queue_delay_microseconds` from 2,000 to 5,000.

| Queue delay | Successful | Rejected | Requests/s (successes) | p50 success latency | p95 success latency | p99 success latency | SLO misses |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 ms | 217 | 1,783 | 32.17 | 108.968 ms | 166.168 ms | 175.040 ms | 1,976 (98.8%) |
| 5 ms | 433 | 1,567 | 67.63 | 55.557 ms | 158.190 ms | 170.336 ms | 1,942 (97.1%) |

The 5 ms window roughly doubled successful throughput and lowered the success-only p50, while most requests still failed or missed the 20 ms SLO. Triton stats for the 5 ms run included 50 sequential warmups: `inference_count=483`, `execution_count=171`, so the measured run averaged `433 / (171 - 50) = 3.58` inferences per execution. Its measured batch distribution was 3 executions of size 1, 5 of size 2, 34 of size 3, 78 of size 4, and 1 of size 6; none reached size 8 or higher. This supports the hypothesis that the 2 ms window was too short to collect larger batches, though it does not explain the remaining queue rejections. The copied benchmark report is [`exercise-5-dynamic-c32-delay5ms-l4.json`](results/exercise-5-dynamic-c32-delay5ms-l4.json).

The server counters after the 5 ms run showed 483 successes, 1,567 `REJECTED` requests, and no canceled or backend-failed requests. Cumulative timing totals divided by 483 successful requests (including warmups) give rough server-side averages: request 45.76 ms, queue 2.80 ms, compute input 0.63 ms, model compute 42.22 ms, and output 0.04 ms. These are averages, not latency percentiles.

### 8 ms queue delay at concurrency 32

| Queue delay | Successful | Rejected | Requests/s (successes) | p50 success latency | p95 success latency | p99 success latency | SLO misses |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 ms | 1,219 | 781 | 259.28 | 34.981 ms | 126.759 ms | 158.226 ms | 1,766 (88.3%) |

Compared with 5 ms, successful throughput rose from 67.63 to 259.28 requests/s, while failures fell from 1,567 to 781. The 8 ms stats snapshot showed `inference_count=1,313` and `execution_count=392`, an aggregate average of 3.35 inferences per execution. Its batch distribution was size 1: 96 executions; size 2: 1; size 3: 3; size 4: 282; size 5: 1; size 6: 1; size 7: 1; size 8: 5; size 10: 2. The large increase in size-4 executions and appearance of size-8 and size-10 batches show that the longer window changed batching behavior.

This stats snapshot is cumulative: it reports 1,313 successful inferences, while the benchmark report has 1,219 measured successes plus 50 warmups (1,269 total). There are 44 additional successful inferences in the stats counters, so the distribution cannot be attributed exactly to this benchmark run alone. The client report is [`exercise-5-dynamic-c32-delay8ms-l4.json`](results/exercise-5-dynamic-c32-delay8ms-l4.json).

### 8 ms queue delay at concurrency 8

| Queue delay | Concurrency | Successful | Rejected | Requests/s (successes) | p50 | p95 | p99 | SLO misses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 ms | 8 | 1,991 | 9 | 718.04 | 10.197 ms | 12.698 ms | 14.223 ms | 16 (0.8%) |

Server metrics reported 2,041 successes (1,991 measured plus 50 warmups), matching the benchmark, and 9 scheduler rejections. The 548 total executions consisted of 50 warmup batches of size 1, 497 measured batches of size 4, and one measured batch of size 3. Thus the measured average batch size was `1,991 / 498 = 4.00`. The cumulative server timing counters correspond to rough per-success averages of request 7.16 ms, queue 1.82 ms, compute input 0.59 ms, model compute 4.67 ms, and output 0.03 ms. Compared with concurrency 32 at the same delay, successful throughput rose from 259.28 to 718.04 requests/s, while p50 fell from 34.981 to 10.197 ms and rejections fell from 781 to 9. The raw report is [`exercise-5-dynamic-c8-delay8ms-l4.json`](results/exercise-5-dynamic-c8-delay8ms-l4.json).

### 8 ms queue delay at concurrency 16

| Queue delay | Concurrency | Successful | Rejected | Requests/s (successes) | p50 | p95 | p99 | SLO misses |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 ms | 16 | 1,508 | 492 | 239.03 | 18.192 ms | 110.719 ms | 121.738 ms | 1,116 (55.8%) |

Server stats reported 1,558 successes (1,508 measured plus 50 warmups), matching the benchmark. After removing the 50 singleton warmups, its 356 measured executions were size 2: 1; size 3: 1; size 4: 331; size 5: 2; size 6: 1; size 7: 3; size 8: 15; size 10: 1; and size 12: 1. The measured average batch size was `1,508 / 356 = 4.24`, only slightly above concurrency 8's 4.00. At 16, throughput fell from 718.04 to 239.03 requests/s, rejections rose from 9 to 492, and p95 latency rose from 12.698 to 110.719 ms. This places a sharp overload transition somewhere between these two tested concurrency levels. The report is [`exercise-5-dynamic-c16-delay8ms-l4.json`](results/exercise-5-dynamic-c16-delay8ms-l4.json).

## Exercise 6: CUDA spatial transform

The CUDA extension accepts decoded RGB input as contiguous HWC `uint8`, performs bicubic resize and center crop on the GPU, and returns a `[224, 224, 3]` HWC `uint8` tensor. The Python caller supplies the resized dimensions and crop offsets. The current implementation supports Pillow's horizontal-first pass order and rejects geometries where Pillow would choose vertical-first.

For `artifacts/parrots.png` (768×512), the reference geometry is resize to 384×256, then center-crop with `left=80`, `top=16`. Comparing the CUDA output against `spatial_uint8_path()` gave shape `(224, 224, 3)`, dtype `uint8`, maximum absolute pixel error `0`, and `0` mismatched channel values. The normalized CUDA tensor also matched Hugging Face exactly: shape `(1, 3, 224, 224)`, maximum and mean absolute error `0.0`, and `allclose=True`. With CPU ONNX Runtime, `inputs_equal=True`, maximum logit error was `0.0`, and top five classes matched: macaw (88), vulture (23), sulphur-crested cockatoo (89), cocktail shaker (503), and toucan (96). This is correctness evidence for one image; it does not measure latency or cover vertical-first image geometries.

### ORT CUDA Execution Provider with I/O Binding

The normalized CUDA tensor was passed directly to ONNX Runtime 1.32.0 with I/O Binding. The session listed `CUDAExecutionProvider` and `CPUExecutionProvider`; TensorRT was available in the installation but was not selected by the script. ORT returned the same top-five class IDs as CPU ORT. Maximum absolute logit error was `0.016289711`. The script bound logits on CUDA and copied only the `[1, 1000]` logits to CPU for comparison.

ORT also emitted `No registered plugin EP device found for 'CUDAExecutionProvider' with device_id=0`. In the 1.32.0 source, this warning is emitted when ORT finds no device registered through the newer plugin EP API; it then attempts the regular CUDA provider factory. The session succeeded.

With `--profile --disable-tf32`, the profile recorded 122 nodes on `CUDAExecutionProvider` and none on CPU, confirming that the graph ran on CUDA. Maximum absolute logit error versus CPU decreased to `0.0010852814`; top five class IDs remained identical. The 113,244-byte profile trace copied from the VM is [`exercise-6-ort-cuda-iobinding-tf32-disabled-profile.json`](results/exercise-6-ort-cuda-iobinding-tf32-disabled-profile.json).

This run also logged two cuDNN Frontend messages, `No execution plans support the graph`, but inference completed. ORT 1.32's convolution code catches some plan-construction failures and retries with an unfused convolution path; one fallback path can re-enable TF32. So the lower error is an observed result with the disable option requested, not proof that every convolution used strict FP32 arithmetic. The source documents this fallback in [`conv.cc`](https://github.com/microsoft/onnxruntime/blob/v1.32.0/onnxruntime/core/providers/cuda/nn/conv.cc#L226-L241).

### Four-way end-to-end comparison

The sequential benchmark ran 100 measured requests per condition after 10 warmups on the NVIDIA L4. It includes decode, preprocessing, inference, and making logits available on CPU. The two CUDA-preprocessing rows keep resize/crop, normalization, and inference on CUDA after CPU image decode; they differ only in ORT's explicit TF32 setting. Full report: [`exercise-6-ort-four-way-l4.json`](results/exercise-6-ort-four-way-l4.json).

| Path | ORT provider | TF32 | p50 | p95 | p99 | Requests/s | Max logit error vs CPU reference |
|---|---|---|---:|---:|---:|---:|---:|
| CUDA preprocess + I/O Binding | CUDA | on | 7.435 ms | 7.591 ms | 7.680 ms | 134.09 | 0.016290 |
| CUDA preprocess + I/O Binding | CUDA | off | 8.517 ms | 8.574 ms | 8.668 ms | 117.24 | 0.001085 |
| Hugging Face CPU preprocess | CUDA (`session.run`) | on | 14.541 ms | 14.757 ms | 15.483 ms | 68.57 | 0.016290 |
| Hugging Face CPU preprocess | CPU | n/a | 52.501 ms | 54.138 ms | 57.760 ms | 18.94 | 0.000000 |

All four conditions returned the same top five classes in the same order. With CPU preprocessing held constant, GPU ORT reduced p50 from 52.50 ms to 14.54 ms (about 3.6× the sequential request rate). With GPU ORT and TF32 enabled held constant, the CUDA-preprocessing path reduced p50 from 14.54 ms to 7.43 ms (about 1.96× the request rate). That second comparison is end-to-end: the CPU path uses `session.run` with a CPU tensor, while the CUDA path uses I/O Binding and sends raw RGB to the GPU before keeping the normalized tensor there. It measures the combined effect of CPU preprocessing and transfer behavior, not just the CUDA kernel speed.

Disabling TF32 raised p50 from 7.43 ms to 8.52 ms, roughly a 15% latency increase, while reducing maximum logit error against CPU ORT from 0.01629 to 0.001085 (about 15× lower). Top-five predictions did not change for this image. This is one sequential run with one image and is not a throughput-under-load result.

## Exercise 7: API contract design review

This is a proposed contract for the lab, not a production SLO already demonstrated by the benchmarks. The online endpoint is latency-oriented; callers send one image while Triton dynamically batches requests from multiple callers. The bulk endpoint lets a caller send a known group of images. Larger jobs use an asynchronous job workflow.

### Online classification

| Contract item | Proposed behavior |
|---|---|
| Route and request | `POST /v1/classify`; one image as the raw request body. |
| Encodings | JPEG or PNG only. Reject animated image formats. |
| Size limits | Up to 10 MiB encoded, at most 16 million decoded pixels, and no side longer than 4,096 pixels. |
| Decode policy | Apply EXIF orientation; decode to 8-bit RGB; composite transparent PNG pixels onto white. |
| Preprocessing owner | The service decodes, resizes the shortest edge to 256 with bicubic resampling, center-crops to 224×224, rescales, and normalizes with the pinned processor's channel means and standard deviations. |
| Model tensor | `float32`, NCHW, `[1, 3, 224, 224]`; color order is RGB. The caller sends encoded image bytes, not a tensor. |
| Batching | Triton may combine separate one-image requests. Start with the measured 8 ms maximum batching delay and a bounded queue. |
| Latency objective | Initial target: p99 ≤ 50 ms at eight concurrent online requests, measured from receipt of the full body to completion of the response. This excludes client upload and network round-trip time. |
| Deadline | 250 ms server-side request deadline. The 50 ms p99 target is a proposal that still needs raw-image testing under concurrent load. |
| Response | Request ID, model version, and the top five `{class_id, label, logit}` predictions. |

### Synchronous bulk and longer jobs

| Contract item | Proposed behavior |
|---|---|
| Synchronous bulk route | `POST /v1/classify/bulk`; multipart request with 1–16 images. Preserve input order in the response. |
| Bulk size limits | Same per-image limits as online; 64 MiB total request body. |
| Model tensor | Service preprocessing produces `float32` RGB NCHW `[N, 3, 224, 224]`, where `1 ≤ N ≤ 16`. The caller groups image files but does not preprocess them. |
| Bulk objective | Initial p99 target ≤ 2 s for 16-image requests at four concurrent bulk callers; track images/s as the primary throughput measure. |
| Synchronous deadline | 10 s. Requests expected to take longer should use an asynchronous job route. |
| Async job route | Submit a larger collection or estimated >10 s workload, receive `202 Accepted` with a job ID, then poll status and retrieve ordered results. |
| Per-image errors | Return an item-level error in that item's result slot while preserving successful results for other valid images. Reject malformed or oversized request bodies before inference. |

### Errors and implementation caveats

- `413` for an encoded-size or body-size limit, `415` for unsupported media types, and `422` for corrupt images or decoded dimensions outside the contract.
- `429` for caller-specific rate limits, `503` with `Retry-After` when service capacity or the bounded queue is exhausted, and `504` when the server deadline expires.
- The 50 ms and 2 s p99 objectives are design targets, not achieved measurements. The four-way Exercise 6 run was sequential, while the serving tests used preprocessed tensors. Validate both targets with raw-image requests at the stated concurrency.
- The current CUDA resize kernel only matches the tested horizontal-first Pillow geometry. Before promising every aspect ratio in the input limits, generalize the kernel to vertical-first cases or provide a correct CPU preprocessing fallback.
