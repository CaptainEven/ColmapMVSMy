#include "cudacc.h"

namespace colmap {

CudaTimer::CudaTimer() {
  CUDA_SAFE_CALL(cudaEventCreate(&start_));
  CUDA_SAFE_CALL(cudaEventCreate(&stop_));
  CUDA_SAFE_CALL(cudaEventRecord(start_, 0));
}

CudaTimer::~CudaTimer() {
  CUDA_SAFE_CALL(cudaEventDestroy(start_));
  CUDA_SAFE_CALL(cudaEventDestroy(stop_));
}

void CudaTimer::Print(const std::string& message) {
  CUDA_SAFE_CALL(cudaEventRecord(stop_, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop_));
  CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time_, start_, stop_));
  std::cout<< message.c_str() << ": "<< elapsed_time_ / 1000.0f << "s" << std::endl;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
	  std::cerr << cudaGetErrorString(error) << " in " <<file.c_str()<<" at line "<<line<< std::endl;
	  std::system("pause");
    exit(EXIT_FAILURE);
  }
}

void CudaCheckError(const char* file, const int line) {
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess) {
	  std::cerr <<"cudaCheckError() failed at" << file << ": "<<line<<" : "<< cudaGetErrorString(error)<< std::endl;
	 std::system("pause");
    exit(EXIT_FAILURE);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
 // error = cudaDeviceSynchronize();
 // if (cudaSuccess != error) {
 //   std::cerr << "cudaCheckError() with sync failed at"<<file << ": "
	//	<< line << " : " << cudaGetErrorString(error)
 //             << std::endl;
 //   std::cerr
 //       << "This error is likely caused by the graphics card timeout "
 //          "detection mechanism of your operating system. Please refer to "
 //          "the FAQ in the documentation on how to solve this problem."
 //       << std::endl;
	//std::system("pause");
 //   exit(EXIT_FAILURE);
 // }
}

}  // namespace colmap
