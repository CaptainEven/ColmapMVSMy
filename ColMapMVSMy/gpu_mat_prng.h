#ifndef COLMAP_SRC_MVS_GPU_MAT_PRNG_H_
#define COLMAP_SRC_MVS_GPU_MAT_PRNG_H_

#include "gpu_mat.h"

namespace colmap {
namespace mvs {

class GpuMatPRNG : public GpuMat<curandState> {
 public:
  GpuMatPRNG(const int width, const int height);

 private:
  void InitRandomState();
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_GPU_MAT_PRNG_H_
