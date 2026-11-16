#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kCropSize = 224;
constexpr int kPillowPrecisionBits = 22;

// Pillow's bicubic filter uses the Keys cubic with a = -0.5 and radius 2.
// This helper returns one unnormalized tap weight; the valid weights for an
// output pixel are divided by their sum below.
__device__ __forceinline__ double pillow_bicubic_weight(double distance) {
  if (distance < 0.0) {
    distance = -distance;
  }

  constexpr double a = -0.5;
  if (distance < 1.0) {
    return ((a + 2.0) * distance - (a + 3.0)) * distance * distance + 1.0;
  }
  if (distance < 2.0) {
    return (((distance - 5.0) * distance + 8.0) * distance - 4.0) * a;
  }
  return 0.0;
}

// Pillow converts normalized 8-bit resampling weights to signed Q22 integers.
// Rounding away from zero matches Pillow's conversion before the pixel sum.
__device__ __forceinline__ int32_t pillow_fixed_weight(double weight) {
  const double scaled = weight * (1 << kPillowPrecisionBits);
  return static_cast<int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
}

// Add Pillow's half-unit rounding bias, shift from Q22, and saturate to uint8.
__device__ __forceinline__ uint8_t pillow_clip8(int32_t accumulator) {
  const int32_t value = accumulator >> kPillowPrecisionBits;
  if (value < 0) {
    return 0;
  }
  if (value > 255) {
    return 255;
  }
  return static_cast<uint8_t>(value);
}

// Horizontal pass for the GPU resize + center-crop path.
// One CUDA thread owns one output pixel and computes all RGB channels for it.
// The temporary image stays in HWC uint8 format for the vertical pass.
__global__ void resize_crop_horizontal_kernel(
    const uint8_t* input, uint8_t* intermediate,
    int input_height, int input_width, int resized_width, int crop_left) {
  // Flatten the [input_height, kCropSize] intermediate pixel grid across blocks.
  const int output_pixel_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int output_pixel_count = input_height * kCropSize;
  if (output_pixel_index >= output_pixel_count) {
    return;
  }

  // Recover this thread's row and column in the cropped horizontal result.
  const int y = output_pixel_index / kCropSize;
  const int x = output_pixel_index - y * kCropSize;

  // The crop begins inside the resized image, so shift x into resized-image space.
  const int resized_x = crop_left + x;

  // Pillow measures output positions from pixel centers. `center` is in the
  // source image's edge-based coordinate system; `source_x` converts it to
  // zero-based pixel-index coordinates.
  const double scale_x = static_cast<double>(input_width) / resized_width;
  const double center = (static_cast<double>(resized_x) + 0.5) * scale_x;
  const double source_x = center - 0.5;

  // During downsampling, widen the filter so it covers the full source area
  // represented by one destination pixel. Pillow keeps the radius at two for
  // upsampling and scales it by `scale_x` for downsampling.
  const double filter_scale = scale_x > 1.0 ? scale_x : 1.0;
  const double support = 2.0 * filter_scale;

  // Find the in-bounds source taps. `x_max` is exclusive. The +0.5 then int
  // conversion matches Pillow's bound calculation and truncation behavior.
  int x_min = static_cast<int>(center - support + 0.5);
  int x_max = static_cast<int>(center + support + 0.5);
  if (x_min < 0) {
    x_min = 0;
  }
  if (x_max > input_width) {
    x_max = input_width;
  }
  const int tap_count = x_max - x_min;

  // Sum the weights over the in-bounds taps. The accumulation pass will divide
  // each tap's weight by this sum, matching Pillow's edge renormalization.
  const double inverse_filter_scale = 1.0 / filter_scale;
  double weight_sum = 0.0;
  for (int tap = 0; tap < tap_count; ++tap) {
    const int source_index = x_min + tap;
    const double distance =
        (static_cast<double>(source_index) - source_x) * inverse_filter_scale;
    weight_sum += pillow_bicubic_weight(distance);
  }

  // Pixel-center coordinates must always have at least one nonzero tap.
  if (tap_count <= 0 || weight_sum == 0.0) {
    return;
  }

  // Pillow quantizes each normalized coefficient to Q22, then accumulates the
  // three channels independently with a half-unit bias for final rounding.
  int32_t red = 1 << (kPillowPrecisionBits - 1);
  int32_t green = 1 << (kPillowPrecisionBits - 1);
  int32_t blue = 1 << (kPillowPrecisionBits - 1);
  const int64_t input_row_offset = static_cast<int64_t>(y) * input_width * 3;
  for (int tap = 0; tap < tap_count; ++tap) {
    const int source_index = x_min + tap;
    const double distance =
        (static_cast<double>(source_index) - source_x) * inverse_filter_scale;
    const double normalized_weight = pillow_bicubic_weight(distance) / weight_sum;
    const int32_t coefficient = pillow_fixed_weight(normalized_weight);
    const int64_t input_offset =
        input_row_offset + static_cast<int64_t>(source_index) * 3;
    red += static_cast<int32_t>(input[input_offset + 0]) * coefficient;
    green += static_cast<int32_t>(input[input_offset + 1]) * coefficient;
    blue += static_cast<int32_t>(input[input_offset + 2]) * coefficient;
  }

  // Store this horizontal result in HWC byte order. The vertical pass will
  // consume the intermediate image and produce the final 224x224 crop.
  const int64_t output_offset = static_cast<int64_t>(output_pixel_index) * 3;
  intermediate[output_offset + 0] = pillow_clip8(red);
  intermediate[output_offset + 1] = pillow_clip8(green);
  intermediate[output_offset + 2] = pillow_clip8(blue);
}

// Vertical pass for the horizontal-first resize path.
// Input is [input_height, 224, 3]; output is the final [224, 224, 3] crop.
// Each thread owns one output pixel and computes all three color channels.
__global__ void resize_crop_vertical_kernel(
    const uint8_t* intermediate, uint8_t* output,
    int input_height, int resized_height, int crop_top) {
  // Flatten the final 224x224 output pixel grid across blocks.
  const int output_pixel_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int output_pixel_count = kCropSize * kCropSize;
  if (output_pixel_index >= output_pixel_count) {
    return;
  }

  // Recover this thread's row and column in the final crop.
  const int y = output_pixel_index / kCropSize;
  const int x = output_pixel_index - y * kCropSize;

  // Select this crop row within the full resized-image coordinate space.
  const int resized_y = crop_top + y;
  const double scale_y = static_cast<double>(input_height) / resized_height;
  const double center = (static_cast<double>(resized_y) + 0.5) * scale_y;
  const double source_y = center - 0.5;

  // Widen the bicubic filter footprint for downsampling.
  const double filter_scale = scale_y > 1.0 ? scale_y : 1.0;
  const double support = 2.0 * filter_scale;

  // Find the in-bounds source rows. The upper bound is exclusive.
  int y_min = static_cast<int>(center - support + 0.5);
  int y_max = static_cast<int>(center + support + 0.5);
  if (y_min < 0) {
    y_min = 0;
  }
  if (y_max > input_height) {
    y_max = input_height;
  }
  const int tap_count = y_max - y_min;

  // Normalize the surviving edge taps, matching the horizontal pass.
  const double inverse_filter_scale = 1.0 / filter_scale;
  double weight_sum = 0.0;
  for (int tap = 0; tap < tap_count; ++tap) {
    const int source_index = y_min + tap;
    const double distance =
        (static_cast<double>(source_index) - source_y) * inverse_filter_scale;
    weight_sum += pillow_bicubic_weight(distance);
  }
  if (tap_count <= 0 || weight_sum == 0.0) {
    return;
  }

  // Apply Pillow's Q22 coefficient rounding and half-unit accumulation bias.
  int32_t red = 1 << (kPillowPrecisionBits - 1);
  int32_t green = 1 << (kPillowPrecisionBits - 1);
  int32_t blue = 1 << (kPillowPrecisionBits - 1);
  for (int tap = 0; tap < tap_count; ++tap) {
    const int source_index = y_min + tap;
    const double distance =
        (static_cast<double>(source_index) - source_y) * inverse_filter_scale;
    const double normalized_weight = pillow_bicubic_weight(distance) / weight_sum;
    const int32_t coefficient = pillow_fixed_weight(normalized_weight);
    const int64_t input_offset =
        (static_cast<int64_t>(source_index) * kCropSize + x) * 3;
    red += static_cast<int32_t>(intermediate[input_offset + 0]) * coefficient;
    green += static_cast<int32_t>(intermediate[input_offset + 1]) * coefficient;
    blue += static_cast<int32_t>(intermediate[input_offset + 2]) * coefficient;
  }

  // Write the final cropped RGB pixel in contiguous HWC order.
  const int64_t output_offset = static_cast<int64_t>(output_pixel_index) * 3;
  output[output_offset + 0] = pillow_clip8(red);
  output[output_offset + 1] = pillow_clip8(green);
  output[output_offset + 2] = pillow_clip8(blue);
}

__global__ void rgb_to_normalized_kernel(
    const uint8_t* input, float* output, int64_t pixels,
    float mean0, float mean1, float mean2,
    float std0, float std1, float std2) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = pixels * 3;
  if (index < total) {
    const int64_t channel = index / pixels;
    const int64_t pixel = index - channel * pixels;
    const float value = static_cast<float>(input[pixel * 3 + channel]) / 255.0f;
    const float mean = channel == 0 ? mean0 : (channel == 1 ? mean1 : mean2);
    const float std = channel == 0 ? std0 : (channel == 1 ? std1 : std2);
    output[index] = (value - mean) / std;
  }
}

}  // namespace

torch::Tensor resize_crop_rgb_cuda(
    torch::Tensor input, int64_t resized_height, int64_t resized_width,
    int64_t crop_top, int64_t crop_left) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const int64_t input_height = input.size(0);
  const int64_t input_width = input.size(1);

  // Keep the horizontal result on the GPU for the vertical pass.
  auto intermediate = torch::empty(
      {input_height, kCropSize, 3}, input.options());
  auto output = torch::empty(
      {kCropSize, kCropSize, 3}, input.options());

  constexpr int threads = 256;
  const int64_t horizontal_pixel_count = input_height * kCropSize;
  const int horizontal_blocks = static_cast<int>(
      (horizontal_pixel_count + threads - 1) / threads);
  const int vertical_blocks =
      (kCropSize * kCropSize + threads - 1) / threads;
  const auto stream = at::cuda::getCurrentCUDAStream();

  // First resize horizontally and keep only the columns inside the center crop.
  resize_crop_horizontal_kernel<<<horizontal_blocks, threads, 0, stream>>>(
      input.data_ptr<uint8_t>(), intermediate.data_ptr<uint8_t>(),
      static_cast<int>(input_height), static_cast<int>(input_width),
      static_cast<int>(resized_width), static_cast<int>(crop_left));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Then resize vertically, selecting the crop rows and writing the final image.
  resize_crop_vertical_kernel<<<vertical_blocks, threads, 0, stream>>>(
      intermediate.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
      static_cast<int>(input_height), static_cast<int>(resized_height),
      static_cast<int>(crop_top));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

torch::Tensor rgb_to_normalized_cuda(
    torch::Tensor input, std::vector<double> mean,
    std::vector<double> standard_deviation) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const int64_t height = input.size(0);
  const int64_t width = input.size(1);
  const int64_t pixels = height * width;
  auto output = torch::empty(
      {1, 3, height, width}, input.options().dtype(at::kFloat));
  constexpr int threads = 256;
  const int blocks = static_cast<int>((pixels * 3 + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  rgb_to_normalized_kernel<<<blocks, threads, 0, stream>>>(
      input.data_ptr<uint8_t>(), output.data_ptr<float>(), pixels,
      static_cast<float>(mean[0]), static_cast<float>(mean[1]),
      static_cast<float>(mean[2]), static_cast<float>(standard_deviation[0]),
      static_cast<float>(standard_deviation[1]),
      static_cast<float>(standard_deviation[2]));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
