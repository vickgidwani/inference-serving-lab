# Exercises

Record the GPU, driver, server image, model revision/hash, processor revision, contract, batch size, concurrency, and command for every result. Use the same host and an idle GPU for comparisons.

1. **Export and inspect.** Export the pinned ResNet-50. Inspect ONNX inputs/outputs, opset, dynamic batch range, and graph size. Compare top five labels from ONNX Runtime with Hugging Face PyTorch for the same image.
2. **Preprocessing fidelity.** Compare Hugging Face's full processor output to the uint8 spatial output used by the custom kernel. Confirm the CUDA tensor conversion matches within tolerance. Change the resampling mode or normalization values and observe how the classifier logits move.
3. **Preprocess cost model.** Measure image decode, spatial resize/crop, H2D copy, and CUDA conversion separately. Compare CPU end-to-end and CPU-spatial-plus-CUDA end-to-end. Explain when transfer and launch overhead outweigh the fused kernel.
4. **Serving contracts.** Compare one image/request with `classify_singleton`, one image/request with concurrent `classify_dynamic`, and 16 images/request with `classify_bulk`. Report both requests/s and images/s, plus p50/p95/p99 latency.
5. **Queue versus compute.** Sweep dynamic batch queue delay (0, 500, 2000, and 5000 microseconds) and concurrency. Compare HTTP latency with Triton's request, queue, and compute metrics. Raise concurrency enough to observe queue growth and rejection behavior.
6. **CUDA spatial transform.** Extend the CUDA code to resize and center-crop directly from decoded RGB input. Preserve the pinned processor's geometry and interpolation behavior, compare pixels and logits against its reference, then measure the full decode/preprocess/model path.
7. **Contract design review.** Define an online raw-image endpoint with a p99 target and an offline bulk endpoint. Specify accepted encodings, maximum encoded size, decoded dimensions, color order, tensor dtype/shape, deadline, overload response, and whether preprocessing or batching belongs to the caller or server.
8. **Lip-reading stretch.** Replace the image model with a visual-only Auto-AVSR checkpoint. Its documented input is 88×88 grayscale mouth crops at 25 fps; add face/mouth tracking, fixed-rate temporal sampling, variable-length clip batching, and CTC decoding. Review the checkpoint's research/non-commercial terms before using it.

## Result template

| Field | Value |
| --- | --- |
| Host / GPU / driver | |
| Triton image / ONNX Runtime | |
| Model / processor revision | |
| Contract / input rows / concurrency / queue delay | |
| Decode / preprocess / H2D / CUDA kernel | |
| p50 / p95 / p99 / max request latency | |
| Requests/s / images/s | |
| Triton queue / compute time | |
| Top prediction / numerical error | |
| Interpretation / caveat | |
