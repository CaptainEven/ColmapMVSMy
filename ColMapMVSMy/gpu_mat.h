#ifndef COLMAP_SRC_MVS_GPU_MAT_H_
#define COLMAP_SRC_MVS_GPU_MAT_H_

#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

//#include "cuda_flip.h"
#include "cuda_rotate.h"
//#include "cuda_transpose.h"
#include "mat.h"
#include "cuda.h"
#include "cudacc.h"


namespace colmap {
namespace mvs {

template <typename T>
class GpuMat {
 public:
  GpuMat(const size_t width, const size_t height, const size_t depth = 1);
  ~GpuMat();

  __host__ __device__ const T* GetPtr() const;
  __host__ __device__ T* GetPtr();

  __host__ __device__ size_t GetPitch() const;
  __host__ __device__ size_t GetWidth() const;
  __host__ __device__ size_t GetHeight() const;
  __host__ __device__ size_t GetDepth() const;

  __device__ T Get(const size_t row, const size_t col,
                   const size_t slice = 0) const;
  __device__ void GetSlice(const size_t row, const size_t col, T* values) const;

  __device__ T& GetRef(const size_t row, const size_t col);
  __device__ T& GetRef(const size_t row, const size_t col, const size_t slice);

  __device__ void Set(const size_t row, const size_t col, const T value);
  __device__ void Set(const size_t row, const size_t col, const size_t slice,
                      const T value);
  __device__ void SetSlice(const size_t row, const size_t col, const T* values);

  void FillWithScalar(const T value);
  void FillWithVector(const T* values);
  void FillWithRandomNumbers(const T min_value, const T max_value,
                             GpuMat<curandState> random_state);

  void CopyToDevice(const T* data, const size_t pitch);
  void CopyToHost(T* data, const size_t pitch) const;
  Mat<T> CopyToMat() const;

  // Transpose array by swapping x and y coordinates.
  void Transpose(GpuMat<T>* output);

  // Flip array along vertical axis.
  void FlipHorizontal(GpuMat<T>* output);

  // Rotate array in counter-clockwise direction.
  void Rotate(GpuMat<T>* output);

  void Read(const std::string& path);
  void Write(const std::string& path);
  void Write(const std::string& path, const size_t slice);

 protected:
  void ComputeCudaConfig();

  const static size_t kBlockDimX = 32;
  const static size_t kBlockDimY = 16;

  std::shared_ptr<T> array_;
  T* array_ptr_;

  size_t pitch_;
  size_t width_;
  size_t height_;
  size_t depth_;

  dim3 blockSize_;
  dim3 gridSize_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template <typename T>
__global__ void FillWithVectorKernel(const T* values, GpuMat<T> output) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      output.Set(row, col, slice, values[slice]);
    }
  }
}

template <typename T>
__global__ void FillWithRandomNumbersKernel(GpuMat<T> output,
                                            GpuMat<curandState> random_state,
                                            const T min_value,
                                            const T max_value) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    curandState local_state = random_state.Get(row, col);
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      const T random_value =
          curand_uniform(&local_state) * (max_value - min_value) + min_value;
      output.Set(row, col, slice, random_value);
    }
    random_state.Set(row, col, local_state);
  }
}

}  // namespace internal

template <typename T>
GpuMat<T>::GpuMat(const size_t width, const size_t height, const size_t depth)
    : array_(nullptr),
      array_ptr_(nullptr),
      width_(width),
      height_(height),
      depth_(depth) {
  checkCudaErrors(cudaMallocPitch((void**)&array_ptr_, &pitch_,
                                 width_ * sizeof(T), height_ * depth_));

  array_ = std::shared_ptr<T>(array_ptr_, cudaFree);

  ComputeCudaConfig();
}

template <typename T>
GpuMat<T>::~GpuMat() {
  array_.reset();
  array_ptr_ = nullptr;
  pitch_ = 0;
  width_ = 0;
  height_ = 0;
  depth_ = 0;
}

template <typename T>
__host__ __device__ const T* GpuMat<T>::GetPtr() const {
  return array_ptr_;
}

template <typename T>
__host__ __device__ T* GpuMat<T>::GetPtr() {
  return array_ptr_;
}

template <typename T>
__host__ __device__ size_t GpuMat<T>::GetPitch() const {
  return pitch_;
}

template <typename T>
__host__ __device__ size_t GpuMat<T>::GetWidth() const {
  return width_;
}

template <typename T>
__host__ __device__ size_t GpuMat<T>::GetHeight() const {
  return height_;
}

template <typename T>
__host__ __device__ size_t GpuMat<T>::GetDepth() const {
  return depth_;
}

template <typename T>
__device__ T GpuMat<T>::Get(const size_t row, const size_t col,
                            const size_t slice) const {
  return *((T*)((char*)array_ptr_ + pitch_ * (slice * height_ + row)) + col);
}

template <typename T>
__device__ void GpuMat<T>::GetSlice(const size_t row, const size_t col,
                                    T* values) const {
  for (size_t slice = 0; slice < depth_; ++slice) {
    values[slice] = Get(row, col, slice);
  }
}

template <typename T>
__device__ T& GpuMat<T>::GetRef(const size_t row, const size_t col) {
  return GetRef(row, col, 0);
}

template <typename T>
__device__ T& GpuMat<T>::GetRef(const size_t row, const size_t col,
                                const size_t slice) {
  return *((T*)((char*)array_ptr_ + pitch_ * (slice * height_ + row)) + col);
}

template <typename T>
__device__ void GpuMat<T>::Set(const size_t row, const size_t col,
                               const T value) {
  Set(row, col, 0, value);
}

template <typename T>
__device__ void GpuMat<T>::Set(const size_t row, const size_t col,
                               const size_t slice, const T value) {
  *((T*)((char*)array_ptr_ + pitch_ * (slice * height_ + row)) + col) = value;
}

template <typename T>
__device__ void GpuMat<T>::SetSlice(const size_t row, const size_t col,
                                    const T* values) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    Set(row, col, slice, values[slice]);
  }
}

template <typename T>
void GpuMat<T>::FillWithScalar(const T value) {
  cudaMemset(array_ptr_, value, width_ * height_ * depth_ * sizeof(T));
  getLastCudaError("FillWithScalar execution failed\n");
}

template <typename T>
void GpuMat<T>::FillWithVector(const T* values) {
  T* values_device;
  cudaMalloc((void**)&values_device, depth_ * sizeof(T));
  cudaMemcpy(values_device, values, depth_ * sizeof(T), cudaMemcpyHostToDevice);
  internal::FillWithVectorKernel<T>
      <<<gridSize_, blockSize_>>>(values_device, *this);
  cudaFree(values_device);
  getLastCudaError("FillWithVectorKernel execution failed\n");
}

template <typename T>
void GpuMat<T>::FillWithRandomNumbers(const T min_value, const T max_value,
                                      const GpuMat<curandState> random_state) {
  internal::FillWithRandomNumbersKernel<T>
      <<<gridSize_, blockSize_>>>(*this, random_state, min_value, max_value);
  getLastCudaError("FillWithRandomNumbersKernel execution failed\n");
}

template <typename T>
void GpuMat<T>::CopyToDevice(const T* data, const size_t pitch) {
	checkCudaErrors(cudaMemcpy2D((void*)array_ptr_, (size_t)pitch_, (void*)data,
                              pitch, width_ * sizeof(T), height_ * depth_,
                              cudaMemcpyHostToDevice));
}

template <typename T>
void GpuMat<T>::CopyToHost(T* data, const size_t pitch) const {
  checkCudaErrors(cudaMemcpy2D((void*)data, pitch, (void*)array_ptr_,
                              (size_t)pitch_, width_ * sizeof(T),
                              height_ * depth_, cudaMemcpyDeviceToHost));
}

template <typename T>
Mat<T> GpuMat<T>::CopyToMat() const {
  Mat<T> mat(width_, height_, depth_);
  CopyToHost(mat.GetPtr(), mat.GetWidth() * sizeof(T));
  return mat;
}

template <typename T>
void GpuMat<T>::Transpose(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaTranspose(array_ptr_ + slice * pitch_ / sizeof(T) * GetHeight(),
                  output->GetPtr() +
                      slice * output->pitch_ / sizeof(T) * output->GetHeight(),
                  width_, height_, pitch_, output->pitch_);
	getLastCudaError("CudaTranspose execution failed\n");
  }
}

template <typename T>
void GpuMat<T>::FlipHorizontal(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaFlipHorizontal(array_ptr_ + slice * pitch_ / sizeof(T) * GetHeight(),
                       output->GetPtr() + slice * output->pitch_ / sizeof(T) *
                                              output->GetHeight(),
                       width_, height_, pitch_, output->pitch_);
	getLastCudaError("CudaFlipHorizontal execution failed\n");
  }
}

template <typename T>
void GpuMat<T>::Rotate(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaRotate(array_ptr_ + slice * pitch_ / sizeof(T) * GetHeight(),
               output->GetPtr() +
                   slice * output->pitch_ / sizeof(T) * output->GetHeight(),
               width_, height_, pitch_, output->pitch_);
	getLastCudaError("CudaRotate execution failed\n");
  }
  // This is equivalent to the following code:
  //   GpuMat<T> flipped_array(width_, height_, GetDepth());
  //   FlipHorizontal(&flipped_array);
  //   flipped_array.Transpose(output);
}

template <typename T>
void GpuMat<T>::Read(const std::string& path) {
  std::fstream text_file(path, std::ios::in);
  if (!text_file.is_open())
  {
	  std::cout << path << "Open failed !" << std::endl;
	  std::system("pause");
	  std::exit(1);
  }

  size_t width;
  size_t height;
  size_t depth;
  char unused_char;
  text_file >> width >> unused_char >> height >> unused_char >> depth >>
      unused_char;

  std::vector<T> source(width_ * height_ * depth_);

  for (int r = 0; r < height; r++)
  {
	  for (int c = 0; c < width; c++)
	  {
		  for (int d = 0; d < depth; d++)
		  {
			  T value;
			  text_file >> value;
			  source.at(d*width*height + r*width + c) = value;
		  }
	  }
  }

  text_file.close();

  CopyToDevice(source.data(), width_ * sizeof(T));
}

template <typename T>
void GpuMat<T>::Write(const std::string& path) {
  std::vector<T> dest(width_ * height_ * depth_);
  CopyToHost(dest.data(), width_ * sizeof(T));

  std::fstream text_file(path, std::ios::out);
  text_file << width_ << "&" << height_ << "&" << depth_ << "&";

  for (int r = 0; r < height_; r++)
  {
	  for (int c = 0; c < width_; c++)
	  {
		  for (int d = 0; d < depth_; d++)
		  {
			  text_file << dest.at(d*width_*height_+r*width_+c) << " ";
		  }
	  }
	  text_file << std::endl;
  }

  text_file.close();
}

template <typename T>
void GpuMat<T>::Write(const std::string& path, const size_t slice) {
  std::vector<T> dest(width_ * height_);
  checkCudaErrors(cudaMemcpy2D(
      (void*)dest.data(), width_ * sizeof(T),
      (void*)(array_ptr_ + slice * height_ * pitch_ / sizeof(T)), pitch_,
      width_ * sizeof(T), height_, cudaMemcpyDeviceToHost));

  std::fstream text_file(path, std::ios::out);
  text_file << width_ << "&" << height_ << "&" << 1 << "&";

  for (int r = 0; r < height_; r++)
  {
	  for (int c = 0; c < width_; c++)
	  {
		
			 text_file << dest.at(r*width_ + c) << " ";
	  }
	  text_file << std::endl;
  }

  text_file.close();
}

template <typename T>
void GpuMat<T>::ComputeCudaConfig() {
  blockSize_.x = kBlockDimX;
  blockSize_.y = kBlockDimY;
  blockSize_.z = 1;

  gridSize_.x = (width_ - 1) / kBlockDimX + 1;
  gridSize_.y = (height_ - 1) / kBlockDimY + 1;
  gridSize_.z = 1;
}


}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_GPU_MAT_H_
