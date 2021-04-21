#ifndef COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_
#define COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_

#include <memory>

#include "cuda_array_wrapper.h"
#include "gpu_mat.h"

namespace colmap {
namespace mvs {

class GpuMatRefImage {
 public:
  GpuMatRefImage(const size_t width, const size_t height);

  // Filter image using sum convolution kernel to compute local sum of
  // intensities. The filtered images can then be used for repeated, efficient
  // NCC computation.
  void Filter(const uint8_t* image_data, const size_t window_radius,
              const float sigma_spatial, const float sigma_color);

  // Image intensities.
  std::unique_ptr<GpuMat<uint8_t>> image;

  // Local sum of image intensities.
  std::unique_ptr<GpuMat<float>> sum_image;

  // Local sum of squared image intensities.
  std::unique_ptr<GpuMat<float>> squared_sum_image;

 private:
  template <int kWindowRadius>
  void Filter(const uint8_t* image_data, const float sigma_spatial,
              const float sigma_color);

  const static size_t kBlockDimX = 16;
  const static size_t kBlockDimY = 12;

  size_t width_;
  size_t height_;
};

__device__ __host__ inline float ComputeBilateralWeight(
    const float row1, const float col1, const float row2, const float col2,
    const float color1, const float color2, const float sigma_spatial,
    const float sigma_color) {
  const float row_diff = row1 - row2;
  const float col_diff = col1 - col2;
  const float spatial_dist = sqrt(row_diff * row_diff + col_diff * col_diff);
  const float color_dist = abs(color1 - color2);
  return exp(-spatial_dist / (2.0f * sigma_spatial * sigma_spatial) -
             color_dist / (2.0f * sigma_color * sigma_color));
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_
