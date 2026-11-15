#include <torch/extension.h>

#include <cstdint>
#include <vector>

torch::Tensor rgb_to_normalized_cuda(
    torch::Tensor input, std::vector<double> mean, std::vector<double> standard_deviation);

torch::Tensor rgb_to_normalized(
    torch::Tensor input, std::vector<double> mean,
    std::vector<double> standard_deviation) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kByte, "input must have dtype uint8");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous HWC data");
  TORCH_CHECK(input.dim() == 3 && input.size(2) == 3, "input must have shape [H, W, 3]");
  TORCH_CHECK(input.numel() > 0, "input must not be empty");
  TORCH_CHECK(
      mean.size() == 3 && standard_deviation.size() == 3,
      "mean and std must each contain 3 values");
  TORCH_CHECK(
      input.size(0) <= 46340 && input.size(1) <= 46340,
      "image dimensions are too large");
  return rgb_to_normalized_cuda(input, mean, standard_deviation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "rgb_to_normalized", &rgb_to_normalized,
      "Convert HWC uint8 RGB to normalized NCHW float32 (CUDA)");
}
