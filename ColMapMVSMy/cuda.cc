#include "cuda.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "common/cpu_anim.h"


namespace colmap {
	namespace {

		// Check whether the first Cuda device is better than the second.
		bool CompareCudaDevice(const cudaDeviceProp& d1, const cudaDeviceProp& d2)
		{
			bool result = (d1.major > d2.major) ||
				((d1.major == d2.major) && (d1.minor > d2.minor)) ||
				((d1.major == d2.major) && (d1.minor == d2.minor) &&
				(d1.multiProcessorCount > d2.multiProcessorCount));
			return result;
		}

	}  // namespace

	int GetNumCudaDevices() {
		int num_cuda_devices;
		cudaGetDeviceCount(&num_cuda_devices);
		return num_cuda_devices;
	}

	void SetBestCudaDevice(const int gpu_index) {
		const int num_cuda_devices = GetNumCudaDevices();
		assert(num_cuda_devices > 0);

		int selected_gpu_index = -1;
		if (gpu_index >= 0) {
			selected_gpu_index = gpu_index;
		}
		else {
			std::vector<cudaDeviceProp> all_devices(num_cuda_devices);
			for (int device_id = 0; device_id < num_cuda_devices; ++device_id) {
				cudaGetDeviceProperties(&all_devices[device_id], device_id);
			}
			std::sort(all_devices.begin(), all_devices.end(), CompareCudaDevice);
			checkCudaErrors(cudaChooseDevice(&selected_gpu_index, all_devices.data()));
		}

		assert(selected_gpu_index >= 0);
		assert(selected_gpu_index < num_cuda_devices);

		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, selected_gpu_index);
		checkCudaErrors(cudaSetDevice(selected_gpu_index));
	}

}  // namespace colmap
