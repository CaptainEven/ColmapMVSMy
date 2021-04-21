#ifndef COLMAP_SRC_UTIL_CUDA_H_
#define COLMAP_SRC_UTIL_CUDA_H_

namespace colmap {

int GetNumCudaDevices();

void SetBestCudaDevice(const int gpu_index);

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CUDA_H_
