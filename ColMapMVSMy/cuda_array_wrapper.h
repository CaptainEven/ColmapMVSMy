#ifndef COLMAP_SRC_MVS_CUDA_ARRAY_WRAPPER_H_
#define COLMAP_SRC_MVS_CUDA_ARRAY_WRAPPER_H_

#include <memory>

#include <cuda_runtime.h>

#include "gpu_mat.h"
#include "cudacc.h"


namespace colmap {
namespace mvs {

template <typename T>
class CudaArrayWrapper {
 public:
  CudaArrayWrapper(const size_t width, const size_t height, const size_t depth);
  ~CudaArrayWrapper();

  const cudaArray* GetPtr() const;
  cudaArray* GetPtr();

  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;

  void CopyToDevice(const T* data);
  void CopyToHost(const T* data);
  void CopyFromGpuMat(const GpuMat<T>& array);

 private:
  // Define class as non-copyable and non-movable.
  CudaArrayWrapper(CudaArrayWrapper const&) = delete;
  void operator=(CudaArrayWrapper const& obj) = delete;
  CudaArrayWrapper(CudaArrayWrapper&&) = delete;

  void Allocate();
  void Deallocate();

  cudaArray* array_;

  size_t width_;
  size_t height_;
  size_t depth_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
CudaArrayWrapper<T>::CudaArrayWrapper(const size_t width, const size_t height,
                                      const size_t depth)
    : width_(width), height_(height), depth_(depth), array_(nullptr) {}

template <typename T>
CudaArrayWrapper<T>::~CudaArrayWrapper() {
  Deallocate();
}

template <typename T>
const cudaArray* CudaArrayWrapper<T>::GetPtr() const {
  return array_;
}

template <typename T>
cudaArray* CudaArrayWrapper<T>::GetPtr() {
  return array_;
}

template <typename T>
size_t CudaArrayWrapper<T>::GetWidth() const {
  return width_;
}

template <typename T>
size_t CudaArrayWrapper<T>::GetHeight() const {
  return height_;
}

template <typename T>
size_t CudaArrayWrapper<T>::GetDepth() const {
  return depth_;
}

template <typename T>
void CudaArrayWrapper<T>::CopyToDevice(const T* data) {
  cudaMemcpy3DParms params = {0};
  Allocate();
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyHostToDevice;
  params.dstArray = array_;
  params.srcPtr =
      make_cudaPitchedPtr((void*)data, width_ * sizeof(T), width_, height_);
  checkCudaErrors(cudaMemcpy3D(&params));
}

template <typename T>
void CudaArrayWrapper<T>::CopyToHost(const T* data) {
  cudaMemcpy3DParms params = {0};
  params.extent = make_cudaExtent(width_, height_, depth_);
  params.kind = cudaMemcpyDeviceToHost;
  params.dstPtr =
      make_cudaPitchedPtr((void*)data, width_ * sizeof(T), width_, height_);
  params.srcArray = array_;
  checkCudaErrors(cudaMemcpy3D(&params));
}

template <typename T>
void CudaArrayWrapper<T>::CopyFromGpuMat(const GpuMat<T>& array) {
  Allocate();
  cudaMemcpy3DParms parameters = {0};
  parameters.extent = make_cudaExtent(width_, height_, depth_);
  parameters.kind = cudaMemcpyDeviceToDevice;
  parameters.dstArray = array_;
  parameters.srcPtr = make_cudaPitchedPtr((void*)array.GetPtr(),
                                          array.GetPitch(), width_, height_);
  checkCudaErrors(cudaMemcpy3D(&parameters));
}

template <typename T>
void CudaArrayWrapper<T>::Allocate() {
  Deallocate();
  struct cudaExtent extent = make_cudaExtent(width_, height_, depth_);
  cudaChannelFormatDesc fmt = cudaCreateChannelDesc<T>();
  checkCudaErrors(cudaMalloc3DArray(&array_, &fmt, extent, cudaArrayLayered));
}

template <typename T>
void CudaArrayWrapper<T>::Deallocate() {
  if (array_ != nullptr) {
    cudaFreeArray(array_);
    array_ = nullptr;
  }
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CUDA_ARRAY_WRAPPER_H_
