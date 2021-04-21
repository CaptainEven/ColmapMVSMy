#ifndef COLMAP_SRC_MVS_PATCH_MATCH_CUDA_H_
#define COLMAP_SRC_MVS_PATCH_MATCH_CUDA_H_

#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "cuda_array_wrapper.h"
#include "depth_map.h"
#include "gpu_mat.h"
#include "gpu_mat_prng.h"
#include "gpu_mat_ref_image.h"
#include "image.h"
#include "normal_map.h"
#include "patch_match.h"

namespace colmap {
namespace mvs {


class PatchMatchCuda {
 public:
  PatchMatchCuda(const PatchMatch::Options& options,
                 const PatchMatch::Problem& problem);
  ~PatchMatchCuda();

  void Run();

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIds() const;

 private:
  template <int kWindowSize>
  void RunWithWindowSize();

  void ComputeCudaConfig();

  void InitRefImage();
  void InitSourceImages();
  void InitTransforms();
  void InitWorkspaceMemory();

  // Rotate reference image by 90 degrees in counter-clockwise direction.
  void Rotate();

  const PatchMatch::Options options_;
  const PatchMatch::Problem problem_;

  // Dimensions for sweeping from top to bottom, i.e. one thread per column.
  dim3 sweep_block_size_;
  dim3 sweep_grid_size_;
  // Dimensions for element-wise operations, i.e. one thread per pixel.
  dim3 elem_wise_block_size_;
  dim3 elem_wise_grid_size_;

  // Original (not rotated) dimension of reference image.
  size_t ref_width_;
  size_t ref_height_;

  // Rotation of reference image in pi/2. This is equivalent to the number of
  // calls to `rotate` mod 4.
  int rotation_in_half_pi_;

  std::unique_ptr<CudaArrayWrapper<uint8_t>> ref_image_device_;
  std::unique_ptr<CudaArrayWrapper<uint8_t>> src_images_device_;
  std::unique_ptr<CudaArrayWrapper<float>> src_depth_maps_device_;

  // Relative poses from rotated versions of reference image to source images
  // corresponding to _rotationInHalfPi:
  //
  //    [S(1), S(2), S(3), ..., S(n)]
  //
  // where n is the number of source images and:
  //
  //    S(i) = [K_i(0, 0), K_i(0, 2), K_i(1, 1), K_i(1, 2), R_i(:), T_i(:)
  //            C_i(:), P(:), P^-1(:)]
  //
  // where i denotes the index of the source image and K is its calibration.
  // R, T, C, P, P^-1 denote the relative rotation, translation, camera
  // center, projection, and inverse projection from there reference to the
  // i-th source image.
  std::unique_ptr<CudaArrayWrapper<float>> poses_device_[4];

  // Calibration matrix for rotated versions of reference image
  // as {K[0, 0], K[0, 2], K[1, 1], K[1, 2]} corresponding to _rotationInHalfPi.
  float ref_K_host_[4][4];
  float ref_inv_K_host_[4][4];

  // Data for reference image.
  std::unique_ptr<GpuMatRefImage> ref_image_;
  std::unique_ptr<GpuMat<float>> depth_map_;
  std::unique_ptr<GpuMat<float>> normal_map_;
  std::unique_ptr<GpuMat<float>> sel_prob_map_;
  std::unique_ptr<GpuMat<float>> prev_sel_prob_map_;
  std::unique_ptr<GpuMat<float>> cost_map_;
  std::unique_ptr<GpuMatPRNG> rand_state_map_;
  std::unique_ptr<GpuMat<uint8_t>> consistency_mask_;

  // Shared memory is too small to hold local state for each thread,
  // so this is workspace memory in global memory.
  std::unique_ptr<GpuMat<float>> global_workspace_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_PATCH_MATCH_CUDA_H_
