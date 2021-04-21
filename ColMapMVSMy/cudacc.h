#ifndef COLMAP_SRC_UTIL_CUDACC_H_
#define COLMAP_SRC_UTIL_CUDACC_H_

#include <iostream>
#include <string>

#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

namespace colmap {

class CudaTimer {
 public:
  CudaTimer();
  ~CudaTimer();

  void Print(const std::string& message);

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  float elapsed_time_;
};

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line);

void CudaCheckError(const char* file, const int line);

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CUDACC_H_
