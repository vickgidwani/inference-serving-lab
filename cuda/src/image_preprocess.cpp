#include <torch/extension.h>

#include <cstdint>
#include <vector>

torch::Tensor rgb_to_normalized_cuda(
    torch::Tensor input, std::vector<double> mean, std::vector<double> standard_deviation);
torch::Tensor resize_crop_rgb_cuda(
    torch::Tensor input, int64_t resized_height, int64_t resized_width,
    int64_t crop_top, int64_t crop_left);

namespace {

constexpr int64_t kResizeCropSize = 224;

}  // namespace

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

torch::Tensor resize_crop_rgb(
    torch::Tensor input, int64_t resized_height, int64_t resized_width,
    int64_t crop_top, int64_t crop_left) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kByte, "input must have dtype uint8");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous HWC data");
  TORCH_CHECK(input.dim() == 3 && input.size(2) == 3, "input must have shape [H, W, 3]");
  TORCH_CHECK(input.numel() > 0, "input must not be empty");

  // Keep kernel indexing and output-grid calculations within their supported range.
  TORCH_CHECK(
      input.size(0) <= 46340 && input.size(1) <= 46340,
      "input image dimensions are too large");
  TORCH_CHECK(
      resized_height >= kResizeCropSize && resized_width >= kResizeCropSize,
      "resized dimensions must be at least 224x224");
  TORCH_CHECK(
      resized_height <= 46340 && resized_width <= 46340,
      "resized dimensions are too large");
  TORCH_CHECK(
      crop_top >= 0 && crop_top <= resized_height - kResizeCropSize,
      "crop_top must select a 224-pixel window inside the resized image");
  TORCH_CHECK(
      crop_left >= 0 && crop_left <= resized_width - kResizeCropSize,
      "crop_left must select a 224-pixel window inside the resized image");

  // Pillow switches pass order for some unusually tall images. This first
  // implementation mirrors its horizontal-first path and rejects other shapes.
  const int64_t vertical_reduction = input.size(0) - resized_height;
  const int64_t horizontal_reduction = input.size(1) - resized_width;
  const bool pillow_uses_vertical_first =
      vertical_reduction > 0 && vertical_reduction > horizontal_reduction * 2;
  TORCH_CHECK(
      !pillow_uses_vertical_first,
      "vertical-first Pillow resize order is not supported by this kernel yet");

  return resize_crop_rgb_cuda(
      input, resized_height, resized_width, crop_top, crop_left);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "rgb_to_normalized", &rgb_to_normalized,
      "Convert HWC uint8 RGB to normalized NCHW float32 (CUDA)");
  module.def(
      "resize_crop_rgb", &resize_crop_rgb,
      "Bicubic resize and center-crop HWC uint8 RGB to 224x224 on CUDA (horizontal-first)");
}
