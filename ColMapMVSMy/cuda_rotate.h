#ifndef COLMAP_SRC_MVS_CUDA_ROTATE_H_
#define COLMAP_SRC_MVS_CUDA_ROTATE_H_

#include <cuda_runtime.h>

namespace colmap {
namespace mvs {

// Rotate the input matrix by 90 degrees in counter-clockwise direction.
template <typename T>
void CudaRotate(const T* input, T* output, const int width, const int height,
                const int pitch_input, const int pitch_output);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

#define TILE_DIM_ROTATE 32

namespace internal {

template <typename T>
__global__ void CudaRotateKernel(T* output_data, const T* input_data,
                                 const int width, const int height,
                                 const int input_pitch,
                                 const int output_pitch) {
  int input_x = blockDim.x * blockIdx.x + threadIdx.x;
  int input_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (input_x >= width || input_y >= height) {
    return;
  }

  int output_x = input_y;
  int output_y = width - 1 - input_x;

  output_data[output_y * output_pitch + output_x] =
      input_data[input_y * input_pitch + input_x];
}

}  // namespace internal

template <typename T>
void CudaRotate(const T* input, T* output, const int width, const int height,
                const int pitch_input, const int pitch_output) {
  dim3 block_dim(TILE_DIM_ROTATE, 1, 1);
  dim3 grid_dim;
  grid_dim.x = (width - 1) / TILE_DIM_ROTATE + 1;
  grid_dim.y = height;

  internal::CudaRotateKernel<<<grid_dim, block_dim>>>(
      output, input, width, height, pitch_input / sizeof(T),
      pitch_output / sizeof(T));
}

#undef TILE_DIM_ROTATE

#endif  // __CUDACC__

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CUDA_ROTATE_H_
