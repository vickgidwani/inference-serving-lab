#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

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
