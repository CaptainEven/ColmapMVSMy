// C++ headers
#include<numeric>
#include<unordered_map>

// opencv headers
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// cuda headers
#include <cuda.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdio.h>


/* 计算MRF的核函数...
先不做任何优化(全在global memory上运行)
尝试每一个Neighbor
*/
__global__ void MRFKernel(
	int* d_labels,
	int* d_pts2d_size,
	int* d_sp_labels,  // 所有候选平面label数组
	float* d_sp_depths,
	cv::Point2f* d_no_depth_pts2d,
	float* d_sp_label_plane_arrs,  // label对应的平面方程数组(附加K_inv_arr)
	int radius, float beta,
	int WIDTH, int HEIGHT,
	int Num_Pts2d, int Num_Labels,  // pt2d点的个数和label的个数
	int* d_sp_labels_ret)
{
	// 待处理pt2d点编号 
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < Num_Pts2d)
	{
		// 初始化最小化能量和最佳label
		float energy_min = FLT_MAX;
		int best_label_id = -1;

		// 初始化中心点指针
		cv::Point2f* ptr_pt2d = d_no_depth_pts2d + tid;

		// 计算二元项范围
		int y_begin = int((*ptr_pt2d).y) - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = int((*ptr_pt2d).y) + radius;
		y_end = y_end <= HEIGHT - 1 ? y_end : HEIGHT - 1;

		int x_begin = int((*ptr_pt2d).x) - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = int((*ptr_pt2d).x) + radius;
		x_end = x_end <= WIDTH - 1 ? x_end : WIDTH - 1;

		// 遍历所有label(plane), 找出能量最小的plane
		for (int label_i = 0; label_i < Num_Labels; ++label_i)
		{
			// ----- 计算二元项能量
			float energy_binary = 0.0f;  // 初始化二元项能量
			for (int y = y_begin; y <= y_end; ++y)
			{
				for (int x = x_begin; x <= x_end; ++x)
				{
					if (d_sp_labels[label_i] == d_labels[y*WIDTH + x])
					{
						energy_binary -= beta;
					}
					else
					{
						energy_binary += beta;
					}
				}
			}
			//energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

			// ----- 计算一元项
			// 计算该取label_i时, 中心点的深度值
			float* K_inv_arr = d_sp_label_plane_arrs + (Num_Labels << 2);
			float* plane_arr = d_sp_label_plane_arrs + (label_i << 2);
			float the_depth = -plane_arr[3] / 
				(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
				+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
				+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
			//printf("The depth of center point: %.3f\n", the_depth);

			// 计算depths的offset
			int offset = 0;
			for (int j = 0; j < label_i; ++j)
			{
				offset += d_pts2d_size[j];
			}

			// 该label对应的深度值起点位置
			const float* ptr_depths = d_sp_depths + offset;

			// --- 计算该label对应的depth的均值和标准差
			float depth_mean = 0.0f, depth_std = 0.0f;

			// 计算该label对应的深度均值
			for (int k = 0; k < d_pts2d_size[label_i]; ++k)
			{
				depth_mean += ptr_depths[k];
			}
			depth_mean /= float(d_pts2d_size[label_i]);

			// 计算该label对应的depth标准差
			for (int k = 0; k < d_pts2d_size[label_i]; ++k)
			{
				depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
			}
			depth_std /= float(d_pts2d_size[label_i]);
			depth_std = sqrtf(depth_std);

			// 计算一元能量
			float energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
				+ 0.5f * (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std);

			// 计算最终的能量
			const float energy = energy_binary + energy_unary;
			if (energy < energy_min)
			{
				energy_min = energy;
				best_label_id = label_i;
			}
		}

		// 写入新的label
		d_sp_labels_ret[tid] = d_sp_labels[best_label_id];
	}
}

int MRFGPU(const cv::Mat& labels,  // 每个pt2d点的label
	const std::vector<int>& SPLabels,  // 候选的的labels
	const std::vector<float>& SPLabelDepths,  // 按照label排列点的深度值
	const std::vector<int>& pts2d_size, // 按照label排列点的pt2d点个数
	const std::vector<cv::Point2f>& NoDepthPts2d,  // 无深度值pt2d点
	const std::vector<float>& sp_label_plane_arrs,  // 按照label排列点的平面方程
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet)
{
	assert(HEIGHT == labels.rows && WIDTH == labels.cols);
	assert(std::accumulate(pts2d_size.begin(), pts2d_size.end(), 0) == HEIGHT * WIDTH);
	assert(SPLabels.size() == pts2d_size.size());

	// 在GPU上分配内存
	int* dev_labels, *dev_pts2d_size, *dev_sp_labels, *dev_sp_labels_ret;
	float *dev_sp_depths, *dev_sp_label_plane_arrs;
	cv::Point2f* dev_no_depth_pts2d;

	cudaMalloc((int**)&dev_labels, sizeof(int) * HEIGHT*WIDTH);
	cudaMalloc((int**)&dev_pts2d_size, sizeof(int) * pts2d_size.size());
	cudaMalloc((int**)&dev_sp_labels, sizeof(int) * SPLabels.size());  // 候选label个数
	cudaMalloc((int**)&dev_sp_labels_ret, sizeof(int) * NoDepthPts2d.size());  // 返回数组
	cudaMalloc((float**)&dev_sp_depths, sizeof(float) * SPLabelDepths.size());
	cudaMalloc((float**)&dev_sp_label_plane_arrs, sizeof(float) * sp_label_plane_arrs.size());
	cudaMalloc((cv::Point2f**)&dev_no_depth_pts2d, sizeof(cv::Point2f) * NoDepthPts2d.size());

	// 将数据拷贝到GPU端
	cudaMemcpy(dev_labels,
		(int*)labels.data,  // uchar* -> int*
		sizeof(int) * labels.rows*labels.cols,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pts2d_size,
		pts2d_size.data(),
		sizeof(int) * pts2d_size.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_labels,
		SPLabels.data(),
		sizeof(int) * SPLabels.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_depths,
		SPLabelDepths.data(),
		sizeof(float) * SPLabelDepths.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_no_depth_pts2d,
		NoDepthPts2d.data(),
		sizeof(cv::Point2f) * NoDepthPts2d.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_label_plane_arrs,
		sp_label_plane_arrs.data(),
		sizeof(float) * sp_label_plane_arrs.size(),
		cudaMemcpyHostToDevice);

	// Kernel参数设置与调用
	int threads_per_block = 128;
	int blocks_per_grid = NoDepthPts2d.size() / threads_per_block + 1;
	MRFKernel << <blocks_per_grid, threads_per_block >> > (
		dev_labels,
		dev_pts2d_size,
		dev_sp_labels,
		dev_sp_depths,
		dev_no_depth_pts2d,
		dev_sp_label_plane_arrs,
		Radius, Beta,
		WIDTH, HEIGHT,
		(int)NoDepthPts2d.size(), (int)SPLabels.size(),
		dev_sp_labels_ret);
	std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
	cudaDeviceSynchronize();  // host等待device同步

	// GPU端数据返回
	cudaMemcpy(NoDepthPt2DSPLabelsRet.data(),
		dev_sp_labels_ret,
		sizeof(int) * NoDepthPts2d.size(),
		cudaMemcpyDeviceToHost);

	// 释放GPU端内存
	cudaFree(dev_labels);
	cudaFree(dev_pts2d_size);
	cudaFree(dev_sp_labels);
	cudaFree(dev_sp_labels_ret);
	cudaFree(dev_sp_depths);
	cudaFree(dev_no_depth_pts2d);
	cudaFree(dev_sp_label_plane_arrs);

	return 0;
}

/*
	只尝试每个Neighbor的label
*/
__global__ void MRFKernel2(
	int* d_labels,
	int* d_pts2d_size,
	int* d_sp_labels,  // 所有候选平面label数组
	const float* d_sp_depths,
	cv::Point2f* d_no_depth_pts2d,  // 待处理的pt2d点数组
	int* d_NoDepthPts2dLabelIdx,  // 每个pt2d点对应的label idx
	float* d_sp_label_plane_arrs,  // label对应的平面方程数组(附加K_inv_arr)
	int* d_sp_label_neighs_idx,  // label_idx对应的neighbors(label idx)
	int* d_sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
	int radius, float beta,
	int WIDTH, int HEIGHT,  // 图像宽高
	int Num_Pts2d, int Num_Labels,  // pt2d点的个数和label的个数
	int* d_sp_labels_ret)
{
	// 待处理pt2d点编号 
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < Num_Pts2d)
	{
		// 初始化中心点指针
		cv::Point2f* ptr_pt2d = d_no_depth_pts2d + tid;

		// 计算二元项范围
		int y_begin = int((*ptr_pt2d).y) - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = int((*ptr_pt2d).y) + radius;
		y_end = y_end <= HEIGHT - 1 ? y_end : HEIGHT - 1;

		int x_begin = int((*ptr_pt2d).x) - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = int((*ptr_pt2d).x) + radius;
		x_end = x_end <= WIDTH - 1 ? x_end : WIDTH - 1;

		// 取相机内参矩阵的逆
		const float* K_inv_arr = d_sp_label_plane_arrs + (Num_Labels << 2);

		// 中心点的label idx
		int the_label_idx = d_NoDepthPts2dLabelIdx[tid];

		// ----- 计算初始化能量(中心点取原label的能量, 与Neighbors比较)
		// 初始化最小化能量和最佳label(初始化为中心点的能量和label)
		int label_idx = the_label_idx;
		int best_label_idx = the_label_idx;

		// ----- 计算二元项能量
		float energy_binary = 0.0f;  // 初始化二元项能量
		for (int y = y_begin; y <= y_end; ++y)
		{
			for (int x = x_begin; x <= x_end; ++x)
			{
				if (d_sp_labels[label_idx] == d_labels[y*WIDTH + x])
				{
					energy_binary -= beta;
				}
				else
				{
					energy_binary += beta;
				}
			}
		}
		energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

		// ----- 计算一元项
		// 计算该取label_i时, 中心点的深度值
		float* plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
		float the_depth = -plane_arr[3] /
			(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
				+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
				+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
		//printf("The depth of center point: %.3f\n", the_depth);

		// 计算depths的offset
		int offset = 0;
		for (int j = 0; j < label_idx; ++j)
		{
			offset += d_pts2d_size[j];
		}

		// 该label对应的深度值起点位置
		const float* ptr_depths = d_sp_depths + offset;

		// --- 计算该label对应的depth的均值和标准差
		float depth_mean = 0.0f, depth_std = 0.0f;

		// 计算该label对应的深度均值
		for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
		{
			depth_mean += ptr_depths[k];
		}
		depth_mean /= float(d_pts2d_size[label_idx]);

		// 计算该label对应的depth标准差
		for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
		{
			depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
		}
		depth_std /= float(d_pts2d_size[label_idx]);
		depth_std = sqrtf(depth_std);

		// 计算一元能量
		float energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
			+ 0.5f * (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std);

		// 初始化最小能量为中心点取原label时的能量
		float energy_min = energy_binary + energy_unary;

		// ---- 计算Neighbors的能量
		// 计算neighbor offset
		offset = 0;
		for (int i = 0; i < the_label_idx; ++i)
		{
			offset += d_sp_label_neigh_num[i];
		}

		// 计算中心点的neigh label idx开始的指针
		const int* ptr_neigh = d_sp_label_neighs_idx + offset;

		// 遍历每一个Neighbor
		for (int i = 0; i < d_sp_label_neigh_num[the_label_idx]; ++i)
		{
			label_idx = ptr_neigh[i];

			// ----- 计算二元项能量
			energy_binary = 0.0f;  // 初始化二元项能量
			for (int y = y_begin; y <= y_end; ++y)
			{
				for (int x = x_begin; x <= x_end; ++x)
				{
					if (d_sp_labels[label_idx] == d_labels[y*WIDTH + x])
					{
						energy_binary -= beta;
					}
					else
					{
						energy_binary += beta;
					}
				}
			}
			energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

			// ----- 计算一元项
			// 计算该取label_i时, 中心点的深度值
			plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
			the_depth = -plane_arr[3] /
				(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
					+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
					+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
			//printf("The depth of center point: %.3f\n", the_depth);

			// 计算depths的offset
			offset = 0;
			for (int j = 0; j < label_idx; ++j)
			{
				offset += d_pts2d_size[j];
			}

			// 该label对应的深度值起点位置
			const float* ptr_depths = d_sp_depths + offset;

			// --- 计算该label对应的depth的均值和标准差
			depth_mean = 0.0f;
			depth_std = 0.0f;

			// 计算该label对应的深度均值
			for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
			{
				depth_mean += ptr_depths[k];
			}
			depth_mean /= float(d_pts2d_size[label_idx]);

			// 计算该label对应的depth标准差
			for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
			{
				depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
			}
			depth_std /= float(d_pts2d_size[label_idx]);
			depth_std = sqrtf(depth_std);

			// 计算一元能量
			energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
				+ 0.5f * (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std);

			// 计算最终的能量
			const float energy = energy_binary + energy_unary;
			if (energy < energy_min)
			{
				energy_min = energy;
				best_label_idx = label_idx;
			}
		}

		// 写入新的label
		d_sp_labels_ret[tid] = d_sp_labels[best_label_idx];
	}
}

int MRFGPU2(const cv::Mat& labels,  // 每个pt2d点的label
	const std::vector<int>& SPLabels,  // 候选的的labels
	const std::vector<float>& SPLabelDepths,  // 按照label排列点的深度值
	const std::vector<int>& pts2d_size, // 按照label排列点的pt2d点个数
	const std::vector<cv::Point2f>& NoDepthPts2d,  // 无深度值pt2d点
	const std::vector<int>& NoDepthPts2dLabelIdx,  // 每个pt2d点的label idx
	const std::vector<float>& sp_label_plane_arrs,  // 按照label排列点的平面方程
	const std::vector<int>& sp_label_neighs_idx,  // 每个label_idx对应的sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet)
{
	assert(HEIGHT == labels.rows && WIDTH == labels.cols);
	assert(std::accumulate(pts2d_size.begin(), pts2d_size.end(), 0) == HEIGHT * WIDTH);
	assert(SPLabels.size() == pts2d_size.size() == sp_label_neigh_num.size());
	assert(NoDepthPts2d.size() == NoDepthPts2dLabelIdx.size());

	// 在GPU上分配内存
	int* dev_labels, *dev_pts2d_size,
		*dev_sp_labels, *dev_sp_labels_ret,
		*dev_sp_label_neighs_idx, *dev_sp_label_neigh_num;
	float*dev_sp_depths, *dev_sp_label_plane_arrs;
	cv::Point2f* dev_no_depth_pts2d;
	int* dev_NoDepthPts2dLabelIdx;

	cudaMalloc((int**)&dev_labels, sizeof(int) * HEIGHT*WIDTH);
	cudaMalloc((int**)&dev_pts2d_size, sizeof(int) * pts2d_size.size());
	cudaMalloc((int**)&dev_sp_labels, sizeof(int) * SPLabels.size());  // 候选label个数
	cudaMalloc((int**)&dev_sp_labels_ret, sizeof(int) * NoDepthPts2d.size());  // 返回数组
	cudaMalloc((int**)&dev_sp_label_neighs_idx, sizeof(int) * sp_label_neighs_idx.size());
	cudaMalloc((int**)&dev_sp_label_neigh_num, sizeof(int) * sp_label_neigh_num.size());
	cudaMalloc((int**)&dev_NoDepthPts2dLabelIdx, sizeof(int) * NoDepthPts2dLabelIdx.size());
	cudaMalloc((float**)&dev_sp_depths, sizeof(float) * SPLabelDepths.size());
	cudaMalloc((float**)&dev_sp_label_plane_arrs, sizeof(float) * sp_label_plane_arrs.size());
	cudaMalloc((cv::Point2f**)&dev_no_depth_pts2d, sizeof(cv::Point2f) * NoDepthPts2d.size());

	// 将数据拷贝到GPU端
	cudaMemcpy(dev_labels,
		(int*)labels.data,  // uchar* -> int*
		sizeof(int) * labels.rows*labels.cols,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pts2d_size,
		pts2d_size.data(),
		sizeof(int) * pts2d_size.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_labels,
		SPLabels.data(),
		sizeof(int) * SPLabels.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_depths,
		SPLabelDepths.data(),
		sizeof(float) * SPLabelDepths.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_no_depth_pts2d,
		NoDepthPts2d.data(),
		sizeof(cv::Point2f) * NoDepthPts2d.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_label_plane_arrs,
		sp_label_plane_arrs.data(),
		sizeof(float) * sp_label_plane_arrs.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_label_neighs_idx,
		sp_label_neighs_idx.data(),
		sizeof(int) * sp_label_neighs_idx.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_label_neigh_num,
		sp_label_neigh_num.data(),
		sizeof(int) * sp_label_neigh_num.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_NoDepthPts2dLabelIdx,
		NoDepthPts2dLabelIdx.data(),
		sizeof(int) * NoDepthPts2dLabelIdx.size(),
		cudaMemcpyHostToDevice);

	// Kernel参数设置与调用
	int threads_per_block = 128;
	int blocks_per_grid = NoDepthPts2d.size() / threads_per_block + 1;
	MRFKernel2 << <blocks_per_grid, threads_per_block >> > (
		dev_labels,
		dev_pts2d_size,
		dev_sp_labels,
		dev_sp_depths,
		dev_no_depth_pts2d,  // 待处理的pt2d点数组
		dev_NoDepthPts2dLabelIdx,  // 每个pt2d点对应的label idx
		dev_sp_label_plane_arrs,
		dev_sp_label_neighs_idx,  // label idx对应的neighbor label idx数组串联
		dev_sp_label_neigh_num,  // label idx对用的neighbor数量数组
		Radius, Beta,
		WIDTH, HEIGHT,
		(int)NoDepthPts2d.size(), (int)SPLabels.size(),
		dev_sp_labels_ret);
	std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
	cudaDeviceSynchronize();  // host等待device同步

	// GPU端数据返回
	cudaMemcpy(NoDepthPt2DSPLabelsRet.data(),
		dev_sp_labels_ret,
		sizeof(int) * NoDepthPts2d.size(),
		cudaMemcpyDeviceToHost);

	// 释放GPU端内存
	cudaFree(dev_labels);
	cudaFree(dev_pts2d_size);
	cudaFree(dev_sp_labels);
	cudaFree(dev_sp_labels_ret);
	cudaFree(dev_sp_depths);
	cudaFree(dev_no_depth_pts2d);
	cudaFree(dev_sp_label_plane_arrs);
	cudaFree(dev_sp_label_neighs_idx);
	cudaFree(dev_sp_label_neigh_num);
	cudaFree(dev_NoDepthPts2dLabelIdx);

	return 0;
}

__global__ void BlockMRFKernel(float* d_depth_mat,
	int* d_proc_blk_ids,
	int* d_label_blk_ids,
	int* d_all_blks_labels,
	int* d_blks_pt_cnt,
	cv::Point2f* d_blks_pts2d,
	cv::Point2f* d_proc_blks_pt2d_non,
	int* d_proc_blks_pts2d_non_num,
	float* d_proc_blks_depths_has,
	int* d_proc_blks_pts2d_has_num,
	float* d_pl_euqa_K_inv_arr,
	int NProcBlk, int NLabels,  // 待处理block数量, Labels数量
	int WIDTH, int HEIGHT,
	int num_x, int num_y,
	int radius, float beta, float depth_range,
	int* labels_ret)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < NProcBlk)
	{
		// 取待处理block的blk_id
		const int proc_blk_id = d_proc_blk_ids[tid];

		// 取待处理block行列索引
		const int proc_blk_y = proc_blk_id / num_x;
		const int proc_blk_x = proc_blk_id % num_x;

		// 取相机内参矩阵的逆
		const float* K_inv_arr = d_pl_euqa_K_inv_arr + (NLabels << 2);

		// 计算二元项范围
		int y_begin = proc_blk_y - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = proc_blk_y + radius;
		y_end = y_end <= num_y - 1 ? y_end : num_y - 1;

		int x_begin = proc_blk_x - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = proc_blk_x + radius;
		x_end = x_end <= num_x - 1 ? x_end : num_x - 1;

		// 初始化最小化能量和最佳label
		float energy_min = FLT_MAX;
		int best_label_i = -1;

		// 遍历每一个候选label
		for (int label_i = 0; label_i < NLabels; ++label_i)
		{
			// 选择当前label_i的3D空间平面方程
			const float* plane_arr = d_pl_euqa_K_inv_arr + (label_i << 2);

			// ----- 计算二元项能量
			// 初始化二元项能量
			float energy_binary = 0.0f;  
			for (int y = y_begin; y <= y_end; ++y)
			{
				for (int x = x_begin; x <= x_end; ++x)
				{
					int idx = y * num_x + x;
					if (d_label_blk_ids[label_i] == d_all_blks_labels[idx])
					{
						energy_binary -= 5.0f*beta;
					}
					else
					{
						energy_binary += beta;
					}
				}
			}
			energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

			// ----- 计算一元项能量
			// 初始化一元项能量
			float energy_unary = 0.0f;

			// 计算待处理的block有深度点个数
			int proc_blk_pt2d_num_non = d_proc_blks_pts2d_non_num[tid];  // 待处理block无深度点数
			if (proc_blk_pt2d_num_non == d_blks_pt_cnt[proc_blk_id])  // 该block全部需要计算深度
			{
				// 计算pt2d点集点数offset
				int offset = 0;
				for (int idx = 0; idx < proc_blk_id; ++idx)
				{
					offset += d_blks_pt_cnt[idx];
				}

				// 计算该block点集开始指针
				const cv::Point2f* ptr_blks_pts2d = d_blks_pts2d + offset;

				// --- 计算该block平面约束的深度均值
				float the_depth_mean = 0.0f;

				for (int i = 0; i < d_blks_pt_cnt[proc_blk_id]; ++i)
				{
					// 计算每个点的深度值(基于当前选择的label: plane array)
					float depth = -plane_arr[3] /
						(plane_arr[0] * (K_inv_arr[0] * ptr_blks_pts2d[i].x + K_inv_arr[1] * ptr_blks_pts2d[i].y + K_inv_arr[2])
							+ plane_arr[1] * (K_inv_arr[3] * ptr_blks_pts2d[i].x + K_inv_arr[4] * ptr_blks_pts2d[i].y + K_inv_arr[5])
							+ plane_arr[2] * (K_inv_arr[6] * ptr_blks_pts2d[i].x + K_inv_arr[7] * ptr_blks_pts2d[i].y + K_inv_arr[8]));

					the_depth_mean += depth;
				}
				the_depth_mean /= float(d_blks_pt_cnt[proc_blk_id]);

				// --- 计算该block原始深度均值
				float orig_depth_mean = 0.0f;
				for (int i = 0; i < proc_blk_pt2d_num_non; ++i)
				{
					// 计算每个点的深度值(取原始深度值)
					orig_depth_mean += d_depth_mat[int(ptr_blks_pts2d[i].y)*WIDTH + int(ptr_blks_pts2d[i].x)];
				}
				orig_depth_mean /= float(d_blks_pt_cnt[proc_blk_id]);

				// 计算一元能量
				energy_unary = fabsf(the_depth_mean - orig_depth_mean) / depth_range;
			}
			else if (proc_blk_pt2d_num_non < d_blks_pt_cnt[proc_blk_id])  // 该blk有部分已知深度值
			{
				// ----- 计算该block平面约束的深度均值
				float the_depth_mean = 0.0f;

				// --- 计算该blk中已经存在的深度值
				// 计算有深度值数组的offset
				int offset = 0;
				for (int i = 0; i < tid; ++i)
				{
					offset += d_proc_blks_pts2d_has_num[i];
				}

				// 计算有深度值数组开始指针
				const float* ptr_d_proc_blks_depths_has = d_proc_blks_depths_has + offset;

				// 统计有深度数组部分的深度值
				for (int i = 0; i < d_proc_blks_pts2d_has_num[tid]; ++i)
				{
					the_depth_mean += ptr_d_proc_blks_depths_has[i];
				}

				// --- 计算无深度值部分的深度值(基于当前选择的label: plane array)
				// 计算无深度点数组的offset
				offset = 0;  // offset清零
				for (int i = 0; i < tid; ++i)
				{
					offset += d_proc_blks_pts2d_non_num[i];
				}

				// 计算无深度点数组开始指针
				const cv::Point2f* ptr_d_proc_blks_pt2d_non = d_proc_blks_pt2d_non + offset;

				// 统计无深度点的深度值
				for (int i = 0; i < d_proc_blks_pts2d_non_num[tid]; ++i)
				{
					float depth = -plane_arr[3] /
						(plane_arr[0] * (K_inv_arr[0] * ptr_d_proc_blks_pt2d_non[i].x + K_inv_arr[1] * ptr_d_proc_blks_pt2d_non[i].y + K_inv_arr[2])
							+ plane_arr[1] * (K_inv_arr[3] * ptr_d_proc_blks_pt2d_non[i].x + K_inv_arr[4] * ptr_d_proc_blks_pt2d_non[i].y + K_inv_arr[5])
							+ plane_arr[2] * (K_inv_arr[6] * ptr_d_proc_blks_pt2d_non[i].x + K_inv_arr[7] * ptr_d_proc_blks_pt2d_non[i].y + K_inv_arr[8]));
					the_depth_mean += depth;
				}
				the_depth_mean /= float(d_blks_pt_cnt[proc_blk_id]);

				// ----- 计算该block原始深度均值
				float orig_depth_mean = 0.0f;

				// 计算所有block的pt2d点集点数offset
				offset = 0;  // offset清零
				for (int idx = 0; idx < proc_blk_id; ++idx)
				{
					offset += d_blks_pt_cnt[idx];
				}

				// 计算该block点集开始指针
				const cv::Point2f* ptr_blks_pts2d = d_blks_pts2d + offset;

				for (int i = 0; i < d_blks_pt_cnt[proc_blk_id]; ++i)
				{
					// 计算每个点的深度值(取原始深度值)
					orig_depth_mean += d_depth_mat[int(ptr_blks_pts2d[i].y)*WIDTH + int(ptr_blks_pts2d[i].x)];
				}
				orig_depth_mean /= float(d_blks_pt_cnt[proc_blk_id]);

				// 计算一元能量
				energy_unary = fabsf(the_depth_mean - orig_depth_mean) / depth_range;
			}
			else
			{
				std::printf("Wrong number of pts2d\n");
			}
			//energy_unary *= 30.0f;

#ifdef LOG
			// for debug...
			if (tid % 600 == 0 && label_i % 10000 == 0)
			{
				std::printf("tid %d, label_i %d | energy_unary: %.3f, energy_binary: %.3f\n",
					tid, label_i, energy_unary, energy_binary);
			}
#endif // LOG

			// 计算最终的能量
			const float energy = energy_binary + energy_unary;

			// 比较, 更新
			if (energy < energy_min)
			{
				energy_min = energy;
				best_label_i = label_i;
			}
		}

		// 填充返回数组
		//labels_ret[tid] = d_label_blk_ids[best_label_i];  // label即block id
		labels_ret[tid] = best_label_i;  // 返回label idx
	}
}

int BlockMRF(const cv::Mat& depth_mat,
	const int blk_size,  // block size
	const float* K_inv_arr,  // 相机内参矩阵的逆
	const std::vector<int>& blks_pt_cnt,  // 记录所有block的pt2d数量
	const std::vector<cv::Point2f>& blks_pts2d,  // 记录所有block的pt2d点坐标
	const std::vector<int>& blks_pt_cnt_has,  // 记录待处理block的有深度值点个数
	const std::vector<int>& blks_pt_cnt_non,  // 记录待处理block的无深度值点个数
	const std::vector<std::vector<float>>& plane_equa_arr,  // 记录作为label的blk_id对应的平面方程
	const std::vector<int>& label_blk_ids,  // 记录有足够多深度值点的blk_id: 可当作label
	const std::vector<int>& process_blk_ids,  // 记录待(MRF)处理的blk_id
	const std::vector<float>& process_blks_depths_has,  // 记录待处理block的有深度值(组成的数组)
	const std::vector<int>& process_blks_pts2d_has_num,  // 记录待处理block的有深度值点个数
	const std::vector<int>& process_blks_pts2d_non_num,  // 记录待处理block无深度点个数
	const std::vector<cv::Point2f>& process_blks_pt2d_non,  // 记录待处理block的无深度值点坐标
	const std::vector<int>& all_blks_labels,  // 记录每个block对应的label(blk_id): 初始label数组
	const int num_x, const int num_y,  // y方向block数量, x方向block数量
	const int radius, const float beta, const float depth_range,
	std::vector<int>& labels_ret)  
{
	const int& HEIGHT = depth_mat.rows;
	const int& WIDTH = depth_mat.cols;

	// 构建plane_equa_arr+K_inv_arr联合数组
	std::vector<float> pl_euqa_K_inv_arr(plane_equa_arr.size() * 4 + 9, 0.0f);
	int stride = 0;
	for (int i = 0; i < (int)plane_equa_arr.size(); ++i)
	{
		memcpy(pl_euqa_K_inv_arr.data() + stride,
			plane_equa_arr[i].data(),
			sizeof(float) * 4);
		stride += 4;
	}
	memcpy(pl_euqa_K_inv_arr.data() + stride,
		K_inv_arr,
		sizeof(float) * 9);

	// 在GPU上分配内存
	int* dev_blks_pt_cnt, *dev_blks_pt_cnt_has, *dev_blks_pt_cnt_non,
		*dev_label_blk_ids, *dev_process_blk_ids,
		*dev_process_blks_pts2d_has_num, *dev_process_blks_pts2d_non_num,
		*dev_all_blks_labels, *dev_labels_ret;  // 9
	float* dev_depth_mat, *dev_pl_euqa_K_inv_arr, *dev_process_blks_depths_has;  // 3
	cv::Point2f* dev_blks_pts2d, *dev_process_blks_pt2d_non;  // 2

	cudaMalloc((int**)&dev_blks_pt_cnt, sizeof(int) * blks_pt_cnt.size());
	cudaMalloc((int**)&dev_blks_pt_cnt_has, sizeof(int) * blks_pt_cnt_has.size());
	cudaMalloc((int**)&dev_blks_pt_cnt_non, sizeof(int) * blks_pt_cnt_non.size());
	cudaMalloc((int**)&dev_label_blk_ids, sizeof(int) * label_blk_ids.size());
	cudaMalloc((int**)&dev_process_blk_ids, sizeof(int) * process_blk_ids.size());
	cudaMalloc((int**)&dev_process_blks_pts2d_has_num, sizeof(int) * process_blks_pts2d_has_num.size());
	cudaMalloc((int**)&dev_process_blks_pts2d_non_num, sizeof(int) * process_blks_pts2d_non_num.size());
	cudaMalloc((int**)&dev_all_blks_labels, sizeof(int) * all_blks_labels.size());
	cudaMalloc((int**)&dev_labels_ret, sizeof(int) * process_blk_ids.size());  // 9

	cudaMalloc((int**)&dev_depth_mat, sizeof(float) * HEIGHT * WIDTH);
	cudaMalloc((float**)&dev_pl_euqa_K_inv_arr, sizeof(float) * pl_euqa_K_inv_arr.size());
	cudaMalloc((float**)&dev_process_blks_depths_has, sizeof(float)* process_blks_depths_has.size());

	cudaMalloc((cv::Point2f**)&dev_blks_pts2d, sizeof(cv::Point2f) * blks_pts2d.size());
	cudaMalloc((cv::Point2f**)&dev_process_blks_pt2d_non, sizeof(cv::Point2f) * process_blks_pt2d_non.size());

	// 将数据拷贝到GPU端
	cudaMemcpy(dev_blks_pt_cnt,
		blks_pt_cnt.data(),
		sizeof(int) * blks_pt_cnt.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_blks_pt_cnt_has,
		blks_pt_cnt_has.data(),
		sizeof(int) * blks_pt_cnt_has.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_blks_pt_cnt_non,
		blks_pt_cnt_non.data(),
		sizeof(int) * blks_pt_cnt_non.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_label_blk_ids,
		label_blk_ids.data(),
		sizeof(int) * label_blk_ids.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_process_blk_ids,
		process_blk_ids.data(),
		sizeof(int) * process_blk_ids.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_process_blks_pts2d_has_num,
		process_blks_pts2d_has_num.data(),
		sizeof(int) * process_blks_pts2d_has_num.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_process_blks_pts2d_non_num,
		process_blks_pts2d_non_num.data(),
		sizeof(int) * process_blks_pts2d_non_num.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_all_blks_labels,
		all_blks_labels.data(),
		sizeof(int) * all_blks_labels.size(),
		cudaMemcpyHostToDevice);

	cudaMemcpy(dev_depth_mat,
		(float*)depth_mat.data,  // uchar* -> float*
		sizeof(float) * HEIGHT * WIDTH,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pl_euqa_K_inv_arr,
		pl_euqa_K_inv_arr.data(),
		sizeof(float) * pl_euqa_K_inv_arr.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_process_blks_depths_has,
		process_blks_depths_has.data(),
		sizeof(float)* process_blks_depths_has.size(),
		cudaMemcpyHostToDevice);

	cudaMemcpy(dev_blks_pts2d,
		blks_pts2d.data(),
		sizeof(cv::Point2f) * blks_pts2d.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_process_blks_pt2d_non,
		process_blks_pt2d_non.data(),
		sizeof(cv::Point2f) * process_blks_pt2d_non.size(),
		cudaMemcpyHostToDevice);

	// Kernel参数设置与调用
	int threads_per_block = 128;
	int blocks_per_grid = process_blk_ids.size() / threads_per_block + 1;
	BlockMRFKernel << <blocks_per_grid, threads_per_block >> > (dev_depth_mat,
		dev_process_blk_ids,
		dev_label_blk_ids,
		dev_all_blks_labels,
		dev_blks_pt_cnt,
		dev_blks_pts2d,
		dev_process_blks_pt2d_non,
		dev_process_blks_pts2d_non_num,
		dev_process_blks_depths_has,
		dev_process_blks_pts2d_has_num,
		dev_pl_euqa_K_inv_arr,
		(int)process_blk_ids.size(), (int)label_blk_ids.size(),
		WIDTH, HEIGHT,
		num_x, num_y,
		radius, beta, depth_range,
		dev_labels_ret);
	std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
	cudaDeviceSynchronize();  // host等待device同步

	// GPU端数据返回
	cudaMemcpy(labels_ret.data(),
		dev_labels_ret, 
		sizeof(int)*process_blk_ids.size(), 
		cudaMemcpyDeviceToHost);

	// GPU端内存释放1
	cudaFree(dev_blks_pt_cnt);
	cudaFree(dev_blks_pt_cnt_has);
	cudaFree(dev_blks_pt_cnt_non);
	cudaFree(dev_label_blk_ids);
	cudaFree(dev_process_blk_ids);
	cudaFree(dev_process_blks_pts2d_has_num);
	cudaFree(dev_process_blks_pts2d_non_num);
	cudaFree(dev_all_blks_labels);
	cudaFree(dev_labels_ret);
	cudaFree(dev_depth_mat);
	cudaFree(dev_pl_euqa_K_inv_arr);
	cudaFree(dev_process_blks_depths_has);
	cudaFree(dev_blks_pts2d);
	cudaFree(dev_process_blks_pt2d_non);  // 14

	return 0;
}

/*
	只尝试每个Neighbor的label
	一元能量项abs(原始深度-该label下的深度)
*/
__global__ void MRFKernel3(
	float* d_depth_mat,
	int* d_labels,
	int* d_sp_labels,  // 所有候选平面label数组
	cv::Point2f* d_no_depth_pts2d,  // 待处理的pt2d点数组
	int* d_NoDepthPts2dLabelIdx,  // 每个pt2d点对应的label idx
	float* d_sp_label_plane_arrs,  // label对应的平面方程数组(附加K_inv_arr)
	int* d_sp_label_neighs_idx,  // label_idx对应的neighbors(label idx)
	int* d_sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
	int radius, float beta,
	int WIDTH, int HEIGHT,  // 图像宽高
	int Num_Pts2d, int Num_Labels,  // pt2d点的个数和label的个数
	int* d_sp_labels_ret)
{
	// 待处理pt2d点编号 
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < Num_Pts2d)
	{
		// 初始化中心点指针
		cv::Point2f* ptr_pt2d = d_no_depth_pts2d + tid;

		// 计算二元项范围
		int y_begin = int((*ptr_pt2d).y) - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = int((*ptr_pt2d).y) + radius;
		y_end = y_end <= HEIGHT - 1 ? y_end : HEIGHT - 1;

		int x_begin = int((*ptr_pt2d).x) - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = int((*ptr_pt2d).x) + radius;
		x_end = x_end <= WIDTH - 1 ? x_end : WIDTH - 1;

		// 取相机内参矩阵的逆
		const float* K_inv_arr = d_sp_label_plane_arrs + (Num_Labels << 2);

		// 中心点的label idx
		int the_label_idx = d_NoDepthPts2dLabelIdx[tid];

		// ----- 计算初始化能量(中心点取原label的能量, 与Neighbors比较)
		// 初始化最小化能量和最佳label(初始化为中心点的能量和label)
		int label_idx = the_label_idx;
		int Best_Label_Idx = the_label_idx;

		// ----- 计算二元项能量
		float energy_binary = 0.0f;  // 初始化二元项能量
		for (int y = y_begin; y <= y_end; ++y)
		{
			for (int x = x_begin; x <= x_end; ++x)
			{
				if (d_sp_labels[label_idx] == d_labels[y*WIDTH + x])
				{
					energy_binary -= beta;
				}
				else
				{
					energy_binary += beta;
				}
			}
		}
		energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

		// ----- 计算一元项
		// 计算该取label_i时, 中心点的深度值
		float* plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
		float the_depth = -plane_arr[3] /
			(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
				+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
				+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
		//printf("The depth of center point: %.3f\n", the_depth);

		const float orig_depth = d_depth_mat[int(ptr_pt2d[0].y)*WIDTH + int(ptr_pt2d[0].x)];

		// 计算一元能量
		float energy_unary = fabsf(the_depth - orig_depth);

		// 初始化最小能量为中心点取原label时的能量
		float Energy_Min = energy_binary + energy_unary;

		// ---- 计算Neighbors的能量
		// 计算neighbor offset
		int offset = 0;
		for (int i = 0; i < the_label_idx; ++i)
		{
			offset += d_sp_label_neigh_num[i];
		}

		// 计算中心点的neigh label idx开始的指针
		const int* ptr_neigh = d_sp_label_neighs_idx + offset;

		// 遍历每一个Neighbor
		for (int i = 0; i < d_sp_label_neigh_num[the_label_idx]; ++i)
		{
			label_idx = ptr_neigh[i];

			// ----- 计算二元项能量
			energy_binary = 0.0f;  // 初始化二元项能量
			for (int y = y_begin; y <= y_end; ++y)
			{
				for (int x = x_begin; x <= x_end; ++x)
				{
					if (d_sp_labels[label_idx] == d_labels[y*WIDTH + x])
					{
						energy_binary -= beta;
					}
					else
					{
						energy_binary += beta;
					}
				}
			}
			energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

			// ----- 计算一元项
			// 计算该取label_i时, 中心点的深度值
			plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
			the_depth = -plane_arr[3] /
				(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
					+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
					+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
			//printf("The depth of center point: %.3f\n", the_depth);

			// 计算一元能量
			energy_unary = fabsf(the_depth - orig_depth);

			//// for debug...
			//if (tid % 100 == 0)
			//{
			//	std::printf("energy_unary: %.3f, energy_binary: %.3f\n",
			//		energy_unary, energy_binary);
			//}

			// 计算最终的能量
			const float energy = energy_binary + energy_unary;
			if (energy < Energy_Min)
			{
				Energy_Min = energy;
				Best_Label_Idx = label_idx;
			}
		}

		// 写入新的label
		d_sp_labels_ret[tid] = d_sp_labels[Best_Label_Idx];
	}
}

int MRFGPU3(const cv::Mat& depth_mat,
	const cv::Mat& labels,  // 每个pt2d点的label
	const std::vector<int>& SPLabels,  // 候选的的labels
	const std::vector<cv::Point2f>& NoDepthPts2d,  // 无深度值pt2d点
	const std::vector<int>& NoDepthPts2dLabelIdx,  // 每个pt2d点的label idx
	const std::vector<float>& sp_label_plane_arrs,  // 按照label排列点的平面方程
	const std::vector<int>& sp_label_neighs_idx,  // 每个label_idx对应的sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet)
{
	assert(HEIGHT == labels.rows && WIDTH == labels.cols);
	assert(std::accumulate(pts2d_size.begin(), pts2d_size.end(), 0) == HEIGHT * WIDTH);
	assert(SPLabels.size() == pts2d_size.size() == sp_label_neigh_num.size());
	assert(NoDepthPts2d.size() == NoDepthPts2dLabelIdx.size());

	// 在GPU上分配内存
	int* dev_labels, *dev_sp_labels, *dev_sp_labels_ret,
		*dev_sp_label_neighs_idx, *dev_sp_label_neigh_num;
	float *dev_depth_mat, *dev_sp_label_plane_arrs;
	cv::Point2f* dev_no_depth_pts2d;
	int* dev_NoDepthPts2dLabelIdx;

	cudaMalloc((int**)&dev_labels, sizeof(int) * HEIGHT*WIDTH);
	cudaMalloc((int**)&dev_sp_labels, sizeof(int) * SPLabels.size());  // 候选label个数
	cudaMalloc((int**)&dev_sp_labels_ret, sizeof(int) * NoDepthPts2d.size());  // 返回数组
	cudaMalloc((int**)&dev_sp_label_neighs_idx, sizeof(int) * sp_label_neighs_idx.size());
	cudaMalloc((int**)&dev_sp_label_neigh_num, sizeof(int) * sp_label_neigh_num.size());
	cudaMalloc((int**)&dev_NoDepthPts2dLabelIdx, sizeof(int) * NoDepthPts2dLabelIdx.size());
	cudaMalloc((float**)&dev_depth_mat, sizeof(float) * HEIGHT*WIDTH);
	cudaMalloc((float**)&dev_sp_label_plane_arrs, sizeof(float) * sp_label_plane_arrs.size());
	cudaMalloc((cv::Point2f**)&dev_no_depth_pts2d, sizeof(cv::Point2f) * NoDepthPts2d.size());

	// 将数据拷贝到GPU端
	cudaMemcpy(dev_labels,
		(int*)labels.data,  // uchar* -> int*
		sizeof(int) * HEIGHT*WIDTH,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_labels,
		SPLabels.data(),
		sizeof(int) * SPLabels.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_no_depth_pts2d,
		NoDepthPts2d.data(),
		sizeof(cv::Point2f) * NoDepthPts2d.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_depth_mat,
		(float*)depth_mat.data,  //uchar* -> float*
		sizeof(float)* HEIGHT*WIDTH,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_label_plane_arrs,
		sp_label_plane_arrs.data(),
		sizeof(float) * sp_label_plane_arrs.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_label_neighs_idx,
		sp_label_neighs_idx.data(),
		sizeof(int) * sp_label_neighs_idx.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_label_neigh_num,
		sp_label_neigh_num.data(),
		sizeof(int) * sp_label_neigh_num.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_NoDepthPts2dLabelIdx,
		NoDepthPts2dLabelIdx.data(),
		sizeof(int) * NoDepthPts2dLabelIdx.size(),
		cudaMemcpyHostToDevice);

	// Kernel参数设置与调用
	int threads_per_block = 128;
	int blocks_per_grid = NoDepthPts2d.size() / threads_per_block + 1;
	MRFKernel3 << <blocks_per_grid, threads_per_block >> > (
		dev_depth_mat,
		dev_labels,
		dev_sp_labels,
		dev_no_depth_pts2d,  // 待处理的pt2d点数组
		dev_NoDepthPts2dLabelIdx,  // 每个pt2d点对应的label idx
		dev_sp_label_plane_arrs,
		dev_sp_label_neighs_idx,  // label idx对应的neighbor label idx数组串联
		dev_sp_label_neigh_num,  // label idx对用的neighbor数量数组
		Radius, Beta,
		WIDTH, HEIGHT,
		(int)NoDepthPts2d.size(), (int)SPLabels.size(),
		dev_sp_labels_ret);
	std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
	cudaDeviceSynchronize();  // host等待device同步

	// GPU端数据返回
	cudaMemcpy(NoDepthPt2DSPLabelsRet.data(),
		dev_sp_labels_ret,
		sizeof(int) * NoDepthPts2d.size(),
		cudaMemcpyDeviceToHost);

	// 释放GPU端内存
	cudaFree(dev_depth_mat);
	cudaFree(dev_labels);
	cudaFree(dev_sp_labels);
	cudaFree(dev_sp_labels_ret);
	cudaFree(dev_no_depth_pts2d);
	cudaFree(dev_sp_label_plane_arrs);
	cudaFree(dev_sp_label_neighs_idx);
	cudaFree(dev_sp_label_neigh_num);
	cudaFree(dev_NoDepthPts2dLabelIdx);

	return 0;
}

// ICM(local minimum)
__global__ void MRFKernel4(float* d_depth_mat,
	int* d_labels,
	int* d_pts2d_size,
	int* d_sp_labels,  // 所有候选平面label数组
	const float* d_sp_depths,
	cv::Point2f* d_no_depth_pts2d,  // 待处理的pt2d点数组
	int* d_NoDepthPts2dLabelIdx,  // 每个pt2d点对应的label idx
	float* d_sp_label_plane_arrs,  // label对应的平面方程数组(附加K_inv_arr)
	int* d_sp_label_neighs_idx,  // label_idx对应的neighbors(label idx)
	int* d_sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
	int radius, float beta,
	int WIDTH, int HEIGHT,  // 图像宽高
	int Num_Pts2d, int Num_Labels,  // pt2d点的个数和label的个数
	int* d_sp_labels_ret)
{
	// 待处理pt2d点编号 
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < Num_Pts2d)
	{
		// 初始化中心点指针
		cv::Point2f* ptr_pt2d = d_no_depth_pts2d + tid;

		// 计算原始中心点深度值
		const float orig_depth = d_depth_mat[int(ptr_pt2d[0].y)*WIDTH + int(ptr_pt2d[0].x)];

		// 计算二元项范围
		int y_begin = int((*ptr_pt2d).y) - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = int((*ptr_pt2d).y) + radius;
		y_end = y_end <= HEIGHT - 1 ? y_end : HEIGHT - 1;

		int x_begin = int((*ptr_pt2d).x) - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = int((*ptr_pt2d).x) + radius;
		x_end = x_end <= WIDTH - 1 ? x_end : WIDTH - 1;

		// 取相机内参矩阵的逆
		const float* K_inv_arr = d_sp_label_plane_arrs + (Num_Labels << 2);

		// 中心点的label idx
		int the_label_idx = d_NoDepthPts2dLabelIdx[tid];

		// ----- 计算初始化能量(中心点取原label的能量, 与Neighbors比较)
		// 初始化最小化能量和最佳label(初始化为中心点的能量和label)
		int label_idx = the_label_idx;
		int best_label_idx = the_label_idx;

		// ----- 计算二元项能量
		float energy_binary = 0.0f;  // 初始化二元项能量
		for (int y = y_begin; y <= y_end; ++y)
		{
			for (int x = x_begin; x <= x_end; ++x)
			{
				if (d_sp_labels[label_idx] == d_labels[y*WIDTH + x])
				{
					energy_binary -= beta;
				}
				else
				{
					energy_binary += beta;
				}
			}
		}
		energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

		// ----- 计算一元项能量
		// 计算该取label_i时, 中心点的深度值
		float* plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
		float the_depth = -plane_arr[3] /
			(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
				+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y +K_inv_arr[5])
				+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y +K_inv_arr[8]));
		//printf("The depth of center point: %.3f\n", the_depth);

		// 计算depths的offset
		int offset = 0;
		for (int j = 0; j < label_idx; ++j)
		{
			offset += d_pts2d_size[j];
		}

		// 该label对应的深度值起点位置
		const float* ptr_depths = d_sp_depths + offset;

		// --- 计算该label对应的depth的均值和标准差
		float depth_mean = 0.0f, depth_std = 0.0f;

		// 计算该label对应的深度均值
		for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
		{
			depth_mean += ptr_depths[k];
		}
		depth_mean /= float(d_pts2d_size[label_idx]);

		// 计算该label对应的depth标准差
		for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
		{
			depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
		}
		depth_std /= float(d_pts2d_size[label_idx]);
		depth_std = sqrtf(depth_std);

		// 计算一元能量
		float energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
			+ (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std)
			+ fabsf(the_depth - orig_depth);

		// 初始化最小能量为中心点取原label时的能量
		float energy_min = 2.0f*energy_binary + energy_unary;

		// ---- 计算Neighbors的能量
		// 计算neighbor offset
		offset = 0;
		for (int i = 0; i < the_label_idx; ++i)
		{
			offset += d_sp_label_neigh_num[i];
		}

		// 计算中心点的neigh label idx开始的指针
		const int* ptr_neigh = d_sp_label_neighs_idx + offset;

		// 遍历每一个Neighbor
		for (int i = 0; i < d_sp_label_neigh_num[the_label_idx]; ++i)
		{
			label_idx = ptr_neigh[i];

			// ----- 计算二元项能量
			energy_binary = 0.0f;  // 初始化二元项能量
			for (int y = y_begin; y <= y_end; ++y)
			{
				for (int x = x_begin; x <= x_end; ++x)
				{
					if (d_sp_labels[label_idx] == d_labels[y*WIDTH + x])
					{
						energy_binary -= beta;
					}
					else
					{
						energy_binary += beta;
					}
				}
			}
			energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

			// ----- 计算一元项能量
			// 计算该取label_i时, 中心点的深度值
			plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
			the_depth = -plane_arr[3] /
				(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y +K_inv_arr[2])
					+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y +K_inv_arr[5])
					+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y +K_inv_arr[8]));
			//printf("The depth of center point: %.3f\n", the_depth);

			// 计算depths的offset
			offset = 0;
			for (int j = 0; j < label_idx; ++j)
			{
				offset += d_pts2d_size[j];
			}

			// 该label对应的深度值起点位置
			const float* ptr_depths = d_sp_depths + offset;

			// --- 计算该label对应的depth的均值和标准差
			depth_mean = 0.0f;
			depth_std = 0.0f;

			// 计算该label对应的深度均值
			for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
			{
				depth_mean += ptr_depths[k];
			}
			depth_mean /= float(d_pts2d_size[label_idx]);

			// 计算该label对应的depth标准差
			for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
			{
				depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
			}
			depth_std /= float(d_pts2d_size[label_idx]);
			depth_std = sqrtf(depth_std);

			// 计算一元能量
			energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
				+ (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std)
				+ fabsf(the_depth - orig_depth);

			//// for debug...
			//if (tid % 1000 == 0)
			//{
			//	std::printf("energy unary: %.3f, energy_binary: %.3f\n",
			//		energy_unary, energy_binary);
			//}

			// 计算最终的能量
			const float energy = 2.0f*energy_binary + energy_unary;
			if (energy < energy_min)
			{
				energy_min = energy;
				best_label_idx = label_idx;
			}
		}

		// 写入新的label
		d_sp_labels_ret[tid] = d_sp_labels[best_label_idx];
	}
}

int MRFGPU4(const cv::Mat& depth_mat,
	const cv::Mat& labels,  // 每个pt2d点的label
		const std::vector<int>& SPLabels,  // 候选的的labels
		const std::vector<float>& SPLabelDepths,  // 按照label排列点的深度值
		const std::vector<int>& pts2d_size, // 按照label排列点的pt2d点个数
		const std::vector<cv::Point2f>& NoDepthPts2d,  // 无深度值pt2d点
		const std::vector<int>& NoDepthPts2dLabelIdx,  // 每个pt2d点的label idx
		const std::vector<float>& sp_label_plane_arrs,  // 按照label排列点的平面方程
		const std::vector<int>& sp_label_neighs_idx,  // 每个label_idx对应的sp_label idx
		const std::vector<int>& sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
		const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
		std::vector<int>& NoDepthPt2DSPLabelsRet)
	{
		assert(HEIGHT == labels.rows && WIDTH == labels.cols);
		assert(std::accumulate(pts2d_size.begin(), pts2d_size.end(), 0) == HEIGHT * WIDTH);
		assert(SPLabels.size() == pts2d_size.size() == sp_label_neigh_num.size());
		assert(NoDepthPts2d.size() == NoDepthPts2dLabelIdx.size());

		// 在GPU上分配内存
		int* dev_labels, *dev_pts2d_size,
			*dev_sp_labels, *dev_sp_labels_ret,
			*dev_sp_label_neighs_idx, *dev_sp_label_neigh_num;
		float* dev_depth_mat, *dev_sp_depths, *dev_sp_label_plane_arrs;
		cv::Point2f* dev_no_depth_pts2d;
		int* dev_NoDepthPts2dLabelIdx;

		cudaMalloc((int**)&dev_labels, sizeof(int) * HEIGHT*WIDTH);
		cudaMalloc((int**)&dev_pts2d_size, sizeof(int) * pts2d_size.size());
		cudaMalloc((int**)&dev_sp_labels, sizeof(int) * SPLabels.size());  // 候选label个数
		cudaMalloc((int**)&dev_sp_labels_ret, sizeof(int) * NoDepthPts2d.size());  // 返回数组
		cudaMalloc((int**)&dev_sp_label_neighs_idx, sizeof(int) * sp_label_neighs_idx.size());
		cudaMalloc((int**)&dev_sp_label_neigh_num, sizeof(int) * sp_label_neigh_num.size());
		cudaMalloc((int**)&dev_NoDepthPts2dLabelIdx, sizeof(int) * NoDepthPts2dLabelIdx.size());
		cudaMalloc((float**)&dev_depth_mat, sizeof(float) * HEIGHT*WIDTH);
		cudaMalloc((float**)&dev_sp_depths, sizeof(float) * SPLabelDepths.size());
		cudaMalloc((float**)&dev_sp_label_plane_arrs, sizeof(float) * sp_label_plane_arrs.size());
		cudaMalloc((cv::Point2f**)&dev_no_depth_pts2d, sizeof(cv::Point2f) * NoDepthPts2d.size());

		// 将数据拷贝到GPU端
		cudaMemcpy(dev_labels,
			(int*)labels.data,  // uchar* -> int*
			sizeof(int) * labels.rows*labels.cols,
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_pts2d_size,
			pts2d_size.data(),
			sizeof(int) * pts2d_size.size(),
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_sp_labels,
			SPLabels.data(),
			sizeof(int) * SPLabels.size(),
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_depth_mat,
			(float*)depth_mat.data,  //uchar* -> float*
			sizeof(float)* HEIGHT*WIDTH,
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_sp_depths,
			SPLabelDepths.data(),
			sizeof(float) * SPLabelDepths.size(),
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_no_depth_pts2d,
			NoDepthPts2d.data(),
			sizeof(cv::Point2f) * NoDepthPts2d.size(),
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_sp_label_plane_arrs,
			sp_label_plane_arrs.data(),
			sizeof(float) * sp_label_plane_arrs.size(),
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_sp_label_neighs_idx,
			sp_label_neighs_idx.data(),
			sizeof(int) * sp_label_neighs_idx.size(),
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_sp_label_neigh_num,
			sp_label_neigh_num.data(),
			sizeof(int) * sp_label_neigh_num.size(),
			cudaMemcpyHostToDevice);
		cudaMemcpy(dev_NoDepthPts2dLabelIdx,
			NoDepthPts2dLabelIdx.data(),
			sizeof(int) * NoDepthPts2dLabelIdx.size(),
			cudaMemcpyHostToDevice);

		// Kernel参数设置与调用
		int threads_per_block = 128;
		int blocks_per_grid = NoDepthPts2d.size() / threads_per_block + 1;
		MRFKernel4 << <blocks_per_grid, threads_per_block >> > (
			dev_depth_mat,
			dev_labels,
			dev_pts2d_size,
			dev_sp_labels,
			dev_sp_depths,
			dev_no_depth_pts2d,  // 待处理的pt2d点数组
			dev_NoDepthPts2dLabelIdx,  // 每个pt2d点对应的label idx
			dev_sp_label_plane_arrs,
			dev_sp_label_neighs_idx,  // label idx对应的neighbor label idx数组串联
			dev_sp_label_neigh_num,  // label idx对用的neighbor数量数组
			Radius, Beta,
			WIDTH, HEIGHT,
			(int)NoDepthPts2d.size(), (int)SPLabels.size(),
			dev_sp_labels_ret);
		std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
		cudaDeviceSynchronize();  // host等待device同步

		// GPU端数据返回
		cudaMemcpy(NoDepthPt2DSPLabelsRet.data(),
			dev_sp_labels_ret,
			sizeof(int) * NoDepthPts2d.size(),
			cudaMemcpyDeviceToHost);

		// 释放GPU端内存
		cudaFree(dev_depth_mat);
		cudaFree(dev_labels);
		cudaFree(dev_pts2d_size);
		cudaFree(dev_sp_labels);
		cudaFree(dev_sp_labels_ret);
		cudaFree(dev_sp_depths);
		cudaFree(dev_no_depth_pts2d);
		cudaFree(dev_sp_label_plane_arrs);
		cudaFree(dev_sp_label_neighs_idx);
		cudaFree(dev_sp_label_neigh_num);
		cudaFree(dev_NoDepthPts2dLabelIdx);

		return 0;
}

__global__ void JBUSPKernel(uchar* d_src,
	float* d_depth_mat,
	cv::Point2f* d_pts2d_has_no_depth_jbu,  // 需要处理的pt2d点数组
	int* d_sp_labels_idx_jbu,  // 每个无深度pt2d点对应的label idx
	cv::Point2f* d_pts2d_has_depth_jbu,
	int* d_sp_has_depth_pt2ds_num,
	float* d_sigmas_s_jbu,  // 每个label对应一个sigma_s
	int WIDTH, int N_pts2d_no_depth,  // 待处理数据个数
	float* d_depths_ret)
{
	// 待处理pt2d点编号 
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N_pts2d_no_depth)
	{
		// 初始化深度值和权重
		float depth = 0.0f, sum_depth = 0.0f, sum_weight = 0.0f;

		// 待处理pt2d点指针
		cv::Point2f* ptr_pt2d_no_depth = d_pts2d_has_no_depth_jbu + tid;

		// ----- 获取无深度值点对应的有深度值点集合指针
		// 计算指针偏移量
		int offset = 0, label_idx = d_sp_labels_idx_jbu[tid];
		for (int i = 0; i < label_idx; ++i)
		{
			offset += d_sp_has_depth_pt2ds_num[i];
		}
		cv::Point2f* ptr_pts2d_depth = d_pts2d_has_depth_jbu + offset;

		// ----- 计算每一个有depth的pt2d点的权重
		// 取无深度值pt2d点的src指针
		uchar* ptr_color_no_depth = d_src +
			(int(ptr_pt2d_no_depth[0].y)*WIDTH + int(ptr_pt2d_no_depth[0].x)) * 3;

		for (int i = 0; i < d_sp_has_depth_pt2ds_num[label_idx]; ++i)
		{
			// --- 计算2D空间距离权重
			float delta_dist = sqrtf(
				  (ptr_pt2d_no_depth[0].x - ptr_pts2d_depth[i].x)
				* (ptr_pt2d_no_depth[0].x - ptr_pts2d_depth[i].x)
				+ (ptr_pt2d_no_depth[0].y - ptr_pts2d_depth[i].y)
				* (ptr_pt2d_no_depth[0].y - ptr_pts2d_depth[i].y)
			);
			float space_weight = expf(-0.5f*delta_dist*delta_dist
				/ (d_sigmas_s_jbu[label_idx] * d_sigmas_s_jbu[label_idx]));

			// --- 计算色差权重
			// 计算当前有深度值的pt2d点的src指针
			uchar* ptr_color_depth = d_src +
				(int(ptr_pts2d_depth[i].y)*WIDTH + int(ptr_pts2d_depth[i].x)) * 3;

			// 计算delta_color: L1 norm of color difference
			float delta_color = fabsf(ptr_color_no_depth[0] - ptr_color_depth[0]) +
				fabsf(ptr_color_no_depth[1] - ptr_color_depth[1]) +
				fabsf(ptr_color_no_depth[2] - ptr_color_depth[2]);
			float color_weight = expf(-0.5f*delta_color*delta_color / 16384.0f);  //128*128(color的sigma值)
			float weight = space_weight * color_weight;

			// 统计权重值
			float depth_neighbor = d_depth_mat[int(ptr_pts2d_depth[i].y)*WIDTH 
				+ int(ptr_pts2d_depth[i].x)];

			sum_depth += weight * depth_neighbor;
			sum_weight += weight;
		}

		// to prevent overflow
		depth = sum_depth / (0.00001f+sum_weight);

		// 写入结果数组
		d_depths_ret[tid] = depth;
	}
}

int JBUSPGPU(const cv::Mat& src,
	const cv::Mat& depth_mat,
	const std::vector<cv::Point2f>& pts2d_has_no_depth_jbu,  // 待处理的pt2d点
	const std::vector<int>& sp_labels_idx_jbu,  // 每个待处理的pt2d点对应的label_idx
	const std::vector<cv::Point2f>& pts2d_has_depth_jbu,  // 用来计算JBU有深度值的pt2d点
	const std::vector<int>& sp_has_depth_pt2ds_num,  // 每个label_idx对应的有深度值pt2d点数
	const std::vector<float>& sigmas_s_jbu, // // 每个label_idx对应的sigma_s
	std::vector<float>& depths_ret)
{
	assert(pts2d_has_no_depth_jbu.size() == sp_labels_idx_jbu.size()
		== depths_ret.size());
	assert(sp_has_depth_pt2ds_num.size() == sigmas_s_jbu.size());

	const int WIDTH = depth_mat.cols;
	const int HEIGHT = depth_mat.rows;

	// 在GPU上分配内存
	uchar *dev_src;
	float* dev_depth_mat;
	cv::Point2f* dev_pts2d_has_no_depth_jbu;
	int* dev_sp_labels_idx_jbu;
	cv::Point2f* dev_pts2d_has_depth_jbu;
	int * dev_sp_has_depth_pt2ds_num;
	float* dev_sigmas_s_jbu;
	float* dev_depths_ret;

	cudaMalloc((uchar**)&dev_src, sizeof(uchar) * WIDTH * HEIGHT * 3);
	cudaMalloc((float**)&dev_depth_mat, sizeof(float) * WIDTH * HEIGHT);
	cudaMalloc((cv::Point2f**)&dev_pts2d_has_no_depth_jbu,
		sizeof(cv::Point2f) * pts2d_has_no_depth_jbu.size());
	cudaMalloc((int**)&dev_sp_labels_idx_jbu, sizeof(int) * sp_labels_idx_jbu.size());
	cudaMalloc((cv::Point2f**)&dev_pts2d_has_depth_jbu,
		sizeof(cv::Point2f) * pts2d_has_depth_jbu.size());
	cudaMalloc((int**)&dev_sp_has_depth_pt2ds_num, sizeof(int) * sp_has_depth_pt2ds_num.size());
	cudaMalloc((float**)&dev_sigmas_s_jbu, sizeof(float) * sigmas_s_jbu.size());
	cudaMalloc((float**)&dev_depths_ret, sizeof(float) * pts2d_has_no_depth_jbu.size());

	// 将数据拷贝到GPU端
	cudaMemcpy(dev_src,
		src.data,  // uchar*
		sizeof(uchar) * WIDTH * HEIGHT * 3,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_depth_mat,
		(float*)depth_mat.data,
		sizeof(float) * WIDTH * HEIGHT,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pts2d_has_no_depth_jbu,
		pts2d_has_no_depth_jbu.data(),
		sizeof(cv::Point2f) * pts2d_has_no_depth_jbu.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_labels_idx_jbu,
		sp_labels_idx_jbu.data(),
		sizeof(int) * sp_labels_idx_jbu.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pts2d_has_depth_jbu,
		pts2d_has_depth_jbu.data(),
		sizeof(cv::Point2f) * pts2d_has_depth_jbu.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sp_has_depth_pt2ds_num,
		sp_has_depth_pt2ds_num.data(),
		sizeof(int) * sp_has_depth_pt2ds_num.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sigmas_s_jbu,
		sigmas_s_jbu.data(),
		sizeof(float) * sigmas_s_jbu.size(),
		cudaMemcpyHostToDevice);

	// Kernel参数设置与调用
	int threads_per_block = 128;  // 初步设置为block最大线程数
	int blocks_per_grid = pts2d_has_no_depth_jbu.size() / threads_per_block + 1;
	JBUSPKernel << <blocks_per_grid, threads_per_block >> > (dev_src,
		dev_depth_mat,
		dev_pts2d_has_no_depth_jbu,
		dev_sp_labels_idx_jbu,
		dev_pts2d_has_depth_jbu,
		dev_sp_has_depth_pt2ds_num,
		dev_sigmas_s_jbu,
		WIDTH, (int)pts2d_has_no_depth_jbu.size(),
		dev_depths_ret);
	std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
	cudaDeviceSynchronize();  // host等待device同步

	// GPU端数据返回
	cudaMemcpy(depths_ret.data(),
		dev_depths_ret,
		sizeof(float) * depths_ret.size(),
		cudaMemcpyDeviceToHost);

	// 释放GPU端内存
	cudaFree(dev_src);
	cudaFree(dev_depth_mat);
	cudaFree(dev_pts2d_has_no_depth_jbu);
	cudaFree(dev_sp_labels_idx_jbu);
	cudaFree(dev_pts2d_has_depth_jbu);
	cudaFree(dev_sp_has_depth_pt2ds_num);
	cudaFree(dev_sigmas_s_jbu);
	cudaFree(dev_depths_ret);

	return 0;
}

//int main() 
//{
//	int deviceCount;
//	cudaGetDeviceCount(&deviceCount);
//
//	int dev;
//	for (dev = 0; dev < deviceCount; dev++)
//	{
//		int driver_version(0), runtime_version(0);
//		cudaDeviceProp deviceProp;
//		cudaGetDeviceProperties(&deviceProp, dev);
//		if (dev == 0)
//			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
//				printf("\n");
//		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
//		cudaDriverGetVersion(&driver_version);
//		printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
//		cudaRuntimeGetVersion(&runtime_version);
//		printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
//		printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
//		printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
//		printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
//		printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
//		printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
//		printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
//		printf("Warp size:                                      %d\n", deviceProp.warpSize);
//		printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
//		printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
//		printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
//			deviceProp.maxThreadsDim[1],
//			deviceProp.maxThreadsDim[2]);
//		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
//		printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
//		printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
//		printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
//		printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
//		printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
//	}
//
//	return 0;
//}
