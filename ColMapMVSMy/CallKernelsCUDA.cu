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


/* ����MRF�ĺ˺���...
�Ȳ����κ��Ż�(ȫ��global memory������)
����ÿһ��Neighbor
*/
__global__ void MRFKernel(
	int* d_labels,
	int* d_pts2d_size,
	int* d_sp_labels,  // ���к�ѡƽ��label����
	float* d_sp_depths,
	cv::Point2f* d_no_depth_pts2d,
	float* d_sp_label_plane_arrs,  // label��Ӧ��ƽ�淽������(����K_inv_arr)
	int radius, float beta,
	int WIDTH, int HEIGHT,
	int Num_Pts2d, int Num_Labels,  // pt2d��ĸ�����label�ĸ���
	int* d_sp_labels_ret)
{
	// ������pt2d���� 
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < Num_Pts2d)
	{
		// ��ʼ����С�����������label
		float energy_min = FLT_MAX;
		int best_label_id = -1;

		// ��ʼ�����ĵ�ָ��
		cv::Point2f* ptr_pt2d = d_no_depth_pts2d + tid;

		// �����Ԫ�Χ
		int y_begin = int((*ptr_pt2d).y) - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = int((*ptr_pt2d).y) + radius;
		y_end = y_end <= HEIGHT - 1 ? y_end : HEIGHT - 1;

		int x_begin = int((*ptr_pt2d).x) - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = int((*ptr_pt2d).x) + radius;
		x_end = x_end <= WIDTH - 1 ? x_end : WIDTH - 1;

		// ��������label(plane), �ҳ�������С��plane
		for (int label_i = 0; label_i < Num_Labels; ++label_i)
		{
			// ----- �����Ԫ������
			float energy_binary = 0.0f;  // ��ʼ����Ԫ������
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

			// ----- ����һԪ��
			// �����ȡlabel_iʱ, ���ĵ�����ֵ
			float* K_inv_arr = d_sp_label_plane_arrs + (Num_Labels << 2);
			float* plane_arr = d_sp_label_plane_arrs + (label_i << 2);
			float the_depth = -plane_arr[3] / 
				(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
				+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
				+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
			//printf("The depth of center point: %.3f\n", the_depth);

			// ����depths��offset
			int offset = 0;
			for (int j = 0; j < label_i; ++j)
			{
				offset += d_pts2d_size[j];
			}

			// ��label��Ӧ�����ֵ���λ��
			const float* ptr_depths = d_sp_depths + offset;

			// --- �����label��Ӧ��depth�ľ�ֵ�ͱ�׼��
			float depth_mean = 0.0f, depth_std = 0.0f;

			// �����label��Ӧ����Ⱦ�ֵ
			for (int k = 0; k < d_pts2d_size[label_i]; ++k)
			{
				depth_mean += ptr_depths[k];
			}
			depth_mean /= float(d_pts2d_size[label_i]);

			// �����label��Ӧ��depth��׼��
			for (int k = 0; k < d_pts2d_size[label_i]; ++k)
			{
				depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
			}
			depth_std /= float(d_pts2d_size[label_i]);
			depth_std = sqrtf(depth_std);

			// ����һԪ����
			float energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
				+ 0.5f * (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std);

			// �������յ�����
			const float energy = energy_binary + energy_unary;
			if (energy < energy_min)
			{
				energy_min = energy;
				best_label_id = label_i;
			}
		}

		// д���µ�label
		d_sp_labels_ret[tid] = d_sp_labels[best_label_id];
	}
}

int MRFGPU(const cv::Mat& labels,  // ÿ��pt2d���label
	const std::vector<int>& SPLabels,  // ��ѡ�ĵ�labels
	const std::vector<float>& SPLabelDepths,  // ����label���е�����ֵ
	const std::vector<int>& pts2d_size, // ����label���е��pt2d�����
	const std::vector<cv::Point2f>& NoDepthPts2d,  // �����ֵpt2d��
	const std::vector<float>& sp_label_plane_arrs,  // ����label���е��ƽ�淽��
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet)
{
	assert(HEIGHT == labels.rows && WIDTH == labels.cols);
	assert(std::accumulate(pts2d_size.begin(), pts2d_size.end(), 0) == HEIGHT * WIDTH);
	assert(SPLabels.size() == pts2d_size.size());

	// ��GPU�Ϸ����ڴ�
	int* dev_labels, *dev_pts2d_size, *dev_sp_labels, *dev_sp_labels_ret;
	float *dev_sp_depths, *dev_sp_label_plane_arrs;
	cv::Point2f* dev_no_depth_pts2d;

	cudaMalloc((int**)&dev_labels, sizeof(int) * HEIGHT*WIDTH);
	cudaMalloc((int**)&dev_pts2d_size, sizeof(int) * pts2d_size.size());
	cudaMalloc((int**)&dev_sp_labels, sizeof(int) * SPLabels.size());  // ��ѡlabel����
	cudaMalloc((int**)&dev_sp_labels_ret, sizeof(int) * NoDepthPts2d.size());  // ��������
	cudaMalloc((float**)&dev_sp_depths, sizeof(float) * SPLabelDepths.size());
	cudaMalloc((float**)&dev_sp_label_plane_arrs, sizeof(float) * sp_label_plane_arrs.size());
	cudaMalloc((cv::Point2f**)&dev_no_depth_pts2d, sizeof(cv::Point2f) * NoDepthPts2d.size());

	// �����ݿ�����GPU��
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

	// Kernel�������������
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
	cudaDeviceSynchronize();  // host�ȴ�deviceͬ��

	// GPU�����ݷ���
	cudaMemcpy(NoDepthPt2DSPLabelsRet.data(),
		dev_sp_labels_ret,
		sizeof(int) * NoDepthPts2d.size(),
		cudaMemcpyDeviceToHost);

	// �ͷ�GPU���ڴ�
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
	ֻ����ÿ��Neighbor��label
*/
__global__ void MRFKernel2(
	int* d_labels,
	int* d_pts2d_size,
	int* d_sp_labels,  // ���к�ѡƽ��label����
	const float* d_sp_depths,
	cv::Point2f* d_no_depth_pts2d,  // �������pt2d������
	int* d_NoDepthPts2dLabelIdx,  // ÿ��pt2d���Ӧ��label idx
	float* d_sp_label_plane_arrs,  // label��Ӧ��ƽ�淽������(����K_inv_arr)
	int* d_sp_label_neighs_idx,  // label_idx��Ӧ��neighbors(label idx)
	int* d_sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
	int radius, float beta,
	int WIDTH, int HEIGHT,  // ͼ����
	int Num_Pts2d, int Num_Labels,  // pt2d��ĸ�����label�ĸ���
	int* d_sp_labels_ret)
{
	// ������pt2d���� 
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < Num_Pts2d)
	{
		// ��ʼ�����ĵ�ָ��
		cv::Point2f* ptr_pt2d = d_no_depth_pts2d + tid;

		// �����Ԫ�Χ
		int y_begin = int((*ptr_pt2d).y) - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = int((*ptr_pt2d).y) + radius;
		y_end = y_end <= HEIGHT - 1 ? y_end : HEIGHT - 1;

		int x_begin = int((*ptr_pt2d).x) - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = int((*ptr_pt2d).x) + radius;
		x_end = x_end <= WIDTH - 1 ? x_end : WIDTH - 1;

		// ȡ����ڲξ������
		const float* K_inv_arr = d_sp_label_plane_arrs + (Num_Labels << 2);

		// ���ĵ��label idx
		int the_label_idx = d_NoDepthPts2dLabelIdx[tid];

		// ----- �����ʼ������(���ĵ�ȡԭlabel������, ��Neighbors�Ƚ�)
		// ��ʼ����С�����������label(��ʼ��Ϊ���ĵ��������label)
		int label_idx = the_label_idx;
		int best_label_idx = the_label_idx;

		// ----- �����Ԫ������
		float energy_binary = 0.0f;  // ��ʼ����Ԫ������
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

		// ----- ����һԪ��
		// �����ȡlabel_iʱ, ���ĵ�����ֵ
		float* plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
		float the_depth = -plane_arr[3] /
			(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
				+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
				+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
		//printf("The depth of center point: %.3f\n", the_depth);

		// ����depths��offset
		int offset = 0;
		for (int j = 0; j < label_idx; ++j)
		{
			offset += d_pts2d_size[j];
		}

		// ��label��Ӧ�����ֵ���λ��
		const float* ptr_depths = d_sp_depths + offset;

		// --- �����label��Ӧ��depth�ľ�ֵ�ͱ�׼��
		float depth_mean = 0.0f, depth_std = 0.0f;

		// �����label��Ӧ����Ⱦ�ֵ
		for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
		{
			depth_mean += ptr_depths[k];
		}
		depth_mean /= float(d_pts2d_size[label_idx]);

		// �����label��Ӧ��depth��׼��
		for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
		{
			depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
		}
		depth_std /= float(d_pts2d_size[label_idx]);
		depth_std = sqrtf(depth_std);

		// ����һԪ����
		float energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
			+ 0.5f * (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std);

		// ��ʼ����С����Ϊ���ĵ�ȡԭlabelʱ������
		float energy_min = energy_binary + energy_unary;

		// ---- ����Neighbors������
		// ����neighbor offset
		offset = 0;
		for (int i = 0; i < the_label_idx; ++i)
		{
			offset += d_sp_label_neigh_num[i];
		}

		// �������ĵ��neigh label idx��ʼ��ָ��
		const int* ptr_neigh = d_sp_label_neighs_idx + offset;

		// ����ÿһ��Neighbor
		for (int i = 0; i < d_sp_label_neigh_num[the_label_idx]; ++i)
		{
			label_idx = ptr_neigh[i];

			// ----- �����Ԫ������
			energy_binary = 0.0f;  // ��ʼ����Ԫ������
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

			// ----- ����һԪ��
			// �����ȡlabel_iʱ, ���ĵ�����ֵ
			plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
			the_depth = -plane_arr[3] /
				(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
					+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
					+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
			//printf("The depth of center point: %.3f\n", the_depth);

			// ����depths��offset
			offset = 0;
			for (int j = 0; j < label_idx; ++j)
			{
				offset += d_pts2d_size[j];
			}

			// ��label��Ӧ�����ֵ���λ��
			const float* ptr_depths = d_sp_depths + offset;

			// --- �����label��Ӧ��depth�ľ�ֵ�ͱ�׼��
			depth_mean = 0.0f;
			depth_std = 0.0f;

			// �����label��Ӧ����Ⱦ�ֵ
			for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
			{
				depth_mean += ptr_depths[k];
			}
			depth_mean /= float(d_pts2d_size[label_idx]);

			// �����label��Ӧ��depth��׼��
			for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
			{
				depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
			}
			depth_std /= float(d_pts2d_size[label_idx]);
			depth_std = sqrtf(depth_std);

			// ����һԪ����
			energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
				+ 0.5f * (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std);

			// �������յ�����
			const float energy = energy_binary + energy_unary;
			if (energy < energy_min)
			{
				energy_min = energy;
				best_label_idx = label_idx;
			}
		}

		// д���µ�label
		d_sp_labels_ret[tid] = d_sp_labels[best_label_idx];
	}
}

int MRFGPU2(const cv::Mat& labels,  // ÿ��pt2d���label
	const std::vector<int>& SPLabels,  // ��ѡ�ĵ�labels
	const std::vector<float>& SPLabelDepths,  // ����label���е�����ֵ
	const std::vector<int>& pts2d_size, // ����label���е��pt2d�����
	const std::vector<cv::Point2f>& NoDepthPts2d,  // �����ֵpt2d��
	const std::vector<int>& NoDepthPts2dLabelIdx,  // ÿ��pt2d���label idx
	const std::vector<float>& sp_label_plane_arrs,  // ����label���е��ƽ�淽��
	const std::vector<int>& sp_label_neighs_idx,  // ÿ��label_idx��Ӧ��sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet)
{
	assert(HEIGHT == labels.rows && WIDTH == labels.cols);
	assert(std::accumulate(pts2d_size.begin(), pts2d_size.end(), 0) == HEIGHT * WIDTH);
	assert(SPLabels.size() == pts2d_size.size() == sp_label_neigh_num.size());
	assert(NoDepthPts2d.size() == NoDepthPts2dLabelIdx.size());

	// ��GPU�Ϸ����ڴ�
	int* dev_labels, *dev_pts2d_size,
		*dev_sp_labels, *dev_sp_labels_ret,
		*dev_sp_label_neighs_idx, *dev_sp_label_neigh_num;
	float*dev_sp_depths, *dev_sp_label_plane_arrs;
	cv::Point2f* dev_no_depth_pts2d;
	int* dev_NoDepthPts2dLabelIdx;

	cudaMalloc((int**)&dev_labels, sizeof(int) * HEIGHT*WIDTH);
	cudaMalloc((int**)&dev_pts2d_size, sizeof(int) * pts2d_size.size());
	cudaMalloc((int**)&dev_sp_labels, sizeof(int) * SPLabels.size());  // ��ѡlabel����
	cudaMalloc((int**)&dev_sp_labels_ret, sizeof(int) * NoDepthPts2d.size());  // ��������
	cudaMalloc((int**)&dev_sp_label_neighs_idx, sizeof(int) * sp_label_neighs_idx.size());
	cudaMalloc((int**)&dev_sp_label_neigh_num, sizeof(int) * sp_label_neigh_num.size());
	cudaMalloc((int**)&dev_NoDepthPts2dLabelIdx, sizeof(int) * NoDepthPts2dLabelIdx.size());
	cudaMalloc((float**)&dev_sp_depths, sizeof(float) * SPLabelDepths.size());
	cudaMalloc((float**)&dev_sp_label_plane_arrs, sizeof(float) * sp_label_plane_arrs.size());
	cudaMalloc((cv::Point2f**)&dev_no_depth_pts2d, sizeof(cv::Point2f) * NoDepthPts2d.size());

	// �����ݿ�����GPU��
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

	// Kernel�������������
	int threads_per_block = 128;
	int blocks_per_grid = NoDepthPts2d.size() / threads_per_block + 1;
	MRFKernel2 << <blocks_per_grid, threads_per_block >> > (
		dev_labels,
		dev_pts2d_size,
		dev_sp_labels,
		dev_sp_depths,
		dev_no_depth_pts2d,  // �������pt2d������
		dev_NoDepthPts2dLabelIdx,  // ÿ��pt2d���Ӧ��label idx
		dev_sp_label_plane_arrs,
		dev_sp_label_neighs_idx,  // label idx��Ӧ��neighbor label idx���鴮��
		dev_sp_label_neigh_num,  // label idx���õ�neighbor��������
		Radius, Beta,
		WIDTH, HEIGHT,
		(int)NoDepthPts2d.size(), (int)SPLabels.size(),
		dev_sp_labels_ret);
	std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
	cudaDeviceSynchronize();  // host�ȴ�deviceͬ��

	// GPU�����ݷ���
	cudaMemcpy(NoDepthPt2DSPLabelsRet.data(),
		dev_sp_labels_ret,
		sizeof(int) * NoDepthPts2d.size(),
		cudaMemcpyDeviceToHost);

	// �ͷ�GPU���ڴ�
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
	int NProcBlk, int NLabels,  // ������block����, Labels����
	int WIDTH, int HEIGHT,
	int num_x, int num_y,
	int radius, float beta, float depth_range,
	int* labels_ret)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < NProcBlk)
	{
		// ȡ������block��blk_id
		const int proc_blk_id = d_proc_blk_ids[tid];

		// ȡ������block��������
		const int proc_blk_y = proc_blk_id / num_x;
		const int proc_blk_x = proc_blk_id % num_x;

		// ȡ����ڲξ������
		const float* K_inv_arr = d_pl_euqa_K_inv_arr + (NLabels << 2);

		// �����Ԫ�Χ
		int y_begin = proc_blk_y - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = proc_blk_y + radius;
		y_end = y_end <= num_y - 1 ? y_end : num_y - 1;

		int x_begin = proc_blk_x - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = proc_blk_x + radius;
		x_end = x_end <= num_x - 1 ? x_end : num_x - 1;

		// ��ʼ����С�����������label
		float energy_min = FLT_MAX;
		int best_label_i = -1;

		// ����ÿһ����ѡlabel
		for (int label_i = 0; label_i < NLabels; ++label_i)
		{
			// ѡ��ǰlabel_i��3D�ռ�ƽ�淽��
			const float* plane_arr = d_pl_euqa_K_inv_arr + (label_i << 2);

			// ----- �����Ԫ������
			// ��ʼ����Ԫ������
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

			// ----- ����һԪ������
			// ��ʼ��һԪ������
			float energy_unary = 0.0f;

			// ����������block����ȵ����
			int proc_blk_pt2d_num_non = d_proc_blks_pts2d_non_num[tid];  // ������block����ȵ���
			if (proc_blk_pt2d_num_non == d_blks_pt_cnt[proc_blk_id])  // ��blockȫ����Ҫ�������
			{
				// ����pt2d�㼯����offset
				int offset = 0;
				for (int idx = 0; idx < proc_blk_id; ++idx)
				{
					offset += d_blks_pt_cnt[idx];
				}

				// �����block�㼯��ʼָ��
				const cv::Point2f* ptr_blks_pts2d = d_blks_pts2d + offset;

				// --- �����blockƽ��Լ������Ⱦ�ֵ
				float the_depth_mean = 0.0f;

				for (int i = 0; i < d_blks_pt_cnt[proc_blk_id]; ++i)
				{
					// ����ÿ��������ֵ(���ڵ�ǰѡ���label: plane array)
					float depth = -plane_arr[3] /
						(plane_arr[0] * (K_inv_arr[0] * ptr_blks_pts2d[i].x + K_inv_arr[1] * ptr_blks_pts2d[i].y + K_inv_arr[2])
							+ plane_arr[1] * (K_inv_arr[3] * ptr_blks_pts2d[i].x + K_inv_arr[4] * ptr_blks_pts2d[i].y + K_inv_arr[5])
							+ plane_arr[2] * (K_inv_arr[6] * ptr_blks_pts2d[i].x + K_inv_arr[7] * ptr_blks_pts2d[i].y + K_inv_arr[8]));

					the_depth_mean += depth;
				}
				the_depth_mean /= float(d_blks_pt_cnt[proc_blk_id]);

				// --- �����blockԭʼ��Ⱦ�ֵ
				float orig_depth_mean = 0.0f;
				for (int i = 0; i < proc_blk_pt2d_num_non; ++i)
				{
					// ����ÿ��������ֵ(ȡԭʼ���ֵ)
					orig_depth_mean += d_depth_mat[int(ptr_blks_pts2d[i].y)*WIDTH + int(ptr_blks_pts2d[i].x)];
				}
				orig_depth_mean /= float(d_blks_pt_cnt[proc_blk_id]);

				// ����һԪ����
				energy_unary = fabsf(the_depth_mean - orig_depth_mean) / depth_range;
			}
			else if (proc_blk_pt2d_num_non < d_blks_pt_cnt[proc_blk_id])  // ��blk�в�����֪���ֵ
			{
				// ----- �����blockƽ��Լ������Ⱦ�ֵ
				float the_depth_mean = 0.0f;

				// --- �����blk���Ѿ����ڵ����ֵ
				// ���������ֵ�����offset
				int offset = 0;
				for (int i = 0; i < tid; ++i)
				{
					offset += d_proc_blks_pts2d_has_num[i];
				}

				// ���������ֵ���鿪ʼָ��
				const float* ptr_d_proc_blks_depths_has = d_proc_blks_depths_has + offset;

				// ͳ����������鲿�ֵ����ֵ
				for (int i = 0; i < d_proc_blks_pts2d_has_num[tid]; ++i)
				{
					the_depth_mean += ptr_d_proc_blks_depths_has[i];
				}

				// --- ���������ֵ���ֵ����ֵ(���ڵ�ǰѡ���label: plane array)
				// ��������ȵ������offset
				offset = 0;  // offset����
				for (int i = 0; i < tid; ++i)
				{
					offset += d_proc_blks_pts2d_non_num[i];
				}

				// ��������ȵ����鿪ʼָ��
				const cv::Point2f* ptr_d_proc_blks_pt2d_non = d_proc_blks_pt2d_non + offset;

				// ͳ������ȵ�����ֵ
				for (int i = 0; i < d_proc_blks_pts2d_non_num[tid]; ++i)
				{
					float depth = -plane_arr[3] /
						(plane_arr[0] * (K_inv_arr[0] * ptr_d_proc_blks_pt2d_non[i].x + K_inv_arr[1] * ptr_d_proc_blks_pt2d_non[i].y + K_inv_arr[2])
							+ plane_arr[1] * (K_inv_arr[3] * ptr_d_proc_blks_pt2d_non[i].x + K_inv_arr[4] * ptr_d_proc_blks_pt2d_non[i].y + K_inv_arr[5])
							+ plane_arr[2] * (K_inv_arr[6] * ptr_d_proc_blks_pt2d_non[i].x + K_inv_arr[7] * ptr_d_proc_blks_pt2d_non[i].y + K_inv_arr[8]));
					the_depth_mean += depth;
				}
				the_depth_mean /= float(d_blks_pt_cnt[proc_blk_id]);

				// ----- �����blockԭʼ��Ⱦ�ֵ
				float orig_depth_mean = 0.0f;

				// ��������block��pt2d�㼯����offset
				offset = 0;  // offset����
				for (int idx = 0; idx < proc_blk_id; ++idx)
				{
					offset += d_blks_pt_cnt[idx];
				}

				// �����block�㼯��ʼָ��
				const cv::Point2f* ptr_blks_pts2d = d_blks_pts2d + offset;

				for (int i = 0; i < d_blks_pt_cnt[proc_blk_id]; ++i)
				{
					// ����ÿ��������ֵ(ȡԭʼ���ֵ)
					orig_depth_mean += d_depth_mat[int(ptr_blks_pts2d[i].y)*WIDTH + int(ptr_blks_pts2d[i].x)];
				}
				orig_depth_mean /= float(d_blks_pt_cnt[proc_blk_id]);

				// ����һԪ����
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

			// �������յ�����
			const float energy = energy_binary + energy_unary;

			// �Ƚ�, ����
			if (energy < energy_min)
			{
				energy_min = energy;
				best_label_i = label_i;
			}
		}

		// ��䷵������
		//labels_ret[tid] = d_label_blk_ids[best_label_i];  // label��block id
		labels_ret[tid] = best_label_i;  // ����label idx
	}
}

int BlockMRF(const cv::Mat& depth_mat,
	const int blk_size,  // block size
	const float* K_inv_arr,  // ����ڲξ������
	const std::vector<int>& blks_pt_cnt,  // ��¼����block��pt2d����
	const std::vector<cv::Point2f>& blks_pts2d,  // ��¼����block��pt2d������
	const std::vector<int>& blks_pt_cnt_has,  // ��¼������block�������ֵ�����
	const std::vector<int>& blks_pt_cnt_non,  // ��¼������block�������ֵ�����
	const std::vector<std::vector<float>>& plane_equa_arr,  // ��¼��Ϊlabel��blk_id��Ӧ��ƽ�淽��
	const std::vector<int>& label_blk_ids,  // ��¼���㹻�����ֵ���blk_id: �ɵ���label
	const std::vector<int>& process_blk_ids,  // ��¼��(MRF)�����blk_id
	const std::vector<float>& process_blks_depths_has,  // ��¼������block�������ֵ(��ɵ�����)
	const std::vector<int>& process_blks_pts2d_has_num,  // ��¼������block�������ֵ�����
	const std::vector<int>& process_blks_pts2d_non_num,  // ��¼������block����ȵ����
	const std::vector<cv::Point2f>& process_blks_pt2d_non,  // ��¼������block�������ֵ������
	const std::vector<int>& all_blks_labels,  // ��¼ÿ��block��Ӧ��label(blk_id): ��ʼlabel����
	const int num_x, const int num_y,  // y����block����, x����block����
	const int radius, const float beta, const float depth_range,
	std::vector<int>& labels_ret)  
{
	const int& HEIGHT = depth_mat.rows;
	const int& WIDTH = depth_mat.cols;

	// ����plane_equa_arr+K_inv_arr��������
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

	// ��GPU�Ϸ����ڴ�
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

	// �����ݿ�����GPU��
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

	// Kernel�������������
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
	cudaDeviceSynchronize();  // host�ȴ�deviceͬ��

	// GPU�����ݷ���
	cudaMemcpy(labels_ret.data(),
		dev_labels_ret, 
		sizeof(int)*process_blk_ids.size(), 
		cudaMemcpyDeviceToHost);

	// GPU���ڴ��ͷ�1
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
	ֻ����ÿ��Neighbor��label
	һԪ������abs(ԭʼ���-��label�µ����)
*/
__global__ void MRFKernel3(
	float* d_depth_mat,
	int* d_labels,
	int* d_sp_labels,  // ���к�ѡƽ��label����
	cv::Point2f* d_no_depth_pts2d,  // �������pt2d������
	int* d_NoDepthPts2dLabelIdx,  // ÿ��pt2d���Ӧ��label idx
	float* d_sp_label_plane_arrs,  // label��Ӧ��ƽ�淽������(����K_inv_arr)
	int* d_sp_label_neighs_idx,  // label_idx��Ӧ��neighbors(label idx)
	int* d_sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
	int radius, float beta,
	int WIDTH, int HEIGHT,  // ͼ����
	int Num_Pts2d, int Num_Labels,  // pt2d��ĸ�����label�ĸ���
	int* d_sp_labels_ret)
{
	// ������pt2d���� 
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < Num_Pts2d)
	{
		// ��ʼ�����ĵ�ָ��
		cv::Point2f* ptr_pt2d = d_no_depth_pts2d + tid;

		// �����Ԫ�Χ
		int y_begin = int((*ptr_pt2d).y) - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = int((*ptr_pt2d).y) + radius;
		y_end = y_end <= HEIGHT - 1 ? y_end : HEIGHT - 1;

		int x_begin = int((*ptr_pt2d).x) - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = int((*ptr_pt2d).x) + radius;
		x_end = x_end <= WIDTH - 1 ? x_end : WIDTH - 1;

		// ȡ����ڲξ������
		const float* K_inv_arr = d_sp_label_plane_arrs + (Num_Labels << 2);

		// ���ĵ��label idx
		int the_label_idx = d_NoDepthPts2dLabelIdx[tid];

		// ----- �����ʼ������(���ĵ�ȡԭlabel������, ��Neighbors�Ƚ�)
		// ��ʼ����С�����������label(��ʼ��Ϊ���ĵ��������label)
		int label_idx = the_label_idx;
		int Best_Label_Idx = the_label_idx;

		// ----- �����Ԫ������
		float energy_binary = 0.0f;  // ��ʼ����Ԫ������
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

		// ----- ����һԪ��
		// �����ȡlabel_iʱ, ���ĵ�����ֵ
		float* plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
		float the_depth = -plane_arr[3] /
			(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
				+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
				+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
		//printf("The depth of center point: %.3f\n", the_depth);

		const float orig_depth = d_depth_mat[int(ptr_pt2d[0].y)*WIDTH + int(ptr_pt2d[0].x)];

		// ����һԪ����
		float energy_unary = fabsf(the_depth - orig_depth);

		// ��ʼ����С����Ϊ���ĵ�ȡԭlabelʱ������
		float Energy_Min = energy_binary + energy_unary;

		// ---- ����Neighbors������
		// ����neighbor offset
		int offset = 0;
		for (int i = 0; i < the_label_idx; ++i)
		{
			offset += d_sp_label_neigh_num[i];
		}

		// �������ĵ��neigh label idx��ʼ��ָ��
		const int* ptr_neigh = d_sp_label_neighs_idx + offset;

		// ����ÿһ��Neighbor
		for (int i = 0; i < d_sp_label_neigh_num[the_label_idx]; ++i)
		{
			label_idx = ptr_neigh[i];

			// ----- �����Ԫ������
			energy_binary = 0.0f;  // ��ʼ����Ԫ������
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

			// ----- ����һԪ��
			// �����ȡlabel_iʱ, ���ĵ�����ֵ
			plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
			the_depth = -plane_arr[3] /
				(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
					+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y + K_inv_arr[5])
					+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y + K_inv_arr[8]));
			//printf("The depth of center point: %.3f\n", the_depth);

			// ����һԪ����
			energy_unary = fabsf(the_depth - orig_depth);

			//// for debug...
			//if (tid % 100 == 0)
			//{
			//	std::printf("energy_unary: %.3f, energy_binary: %.3f\n",
			//		energy_unary, energy_binary);
			//}

			// �������յ�����
			const float energy = energy_binary + energy_unary;
			if (energy < Energy_Min)
			{
				Energy_Min = energy;
				Best_Label_Idx = label_idx;
			}
		}

		// д���µ�label
		d_sp_labels_ret[tid] = d_sp_labels[Best_Label_Idx];
	}
}

int MRFGPU3(const cv::Mat& depth_mat,
	const cv::Mat& labels,  // ÿ��pt2d���label
	const std::vector<int>& SPLabels,  // ��ѡ�ĵ�labels
	const std::vector<cv::Point2f>& NoDepthPts2d,  // �����ֵpt2d��
	const std::vector<int>& NoDepthPts2dLabelIdx,  // ÿ��pt2d���label idx
	const std::vector<float>& sp_label_plane_arrs,  // ����label���е��ƽ�淽��
	const std::vector<int>& sp_label_neighs_idx,  // ÿ��label_idx��Ӧ��sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet)
{
	assert(HEIGHT == labels.rows && WIDTH == labels.cols);
	assert(std::accumulate(pts2d_size.begin(), pts2d_size.end(), 0) == HEIGHT * WIDTH);
	assert(SPLabels.size() == pts2d_size.size() == sp_label_neigh_num.size());
	assert(NoDepthPts2d.size() == NoDepthPts2dLabelIdx.size());

	// ��GPU�Ϸ����ڴ�
	int* dev_labels, *dev_sp_labels, *dev_sp_labels_ret,
		*dev_sp_label_neighs_idx, *dev_sp_label_neigh_num;
	float *dev_depth_mat, *dev_sp_label_plane_arrs;
	cv::Point2f* dev_no_depth_pts2d;
	int* dev_NoDepthPts2dLabelIdx;

	cudaMalloc((int**)&dev_labels, sizeof(int) * HEIGHT*WIDTH);
	cudaMalloc((int**)&dev_sp_labels, sizeof(int) * SPLabels.size());  // ��ѡlabel����
	cudaMalloc((int**)&dev_sp_labels_ret, sizeof(int) * NoDepthPts2d.size());  // ��������
	cudaMalloc((int**)&dev_sp_label_neighs_idx, sizeof(int) * sp_label_neighs_idx.size());
	cudaMalloc((int**)&dev_sp_label_neigh_num, sizeof(int) * sp_label_neigh_num.size());
	cudaMalloc((int**)&dev_NoDepthPts2dLabelIdx, sizeof(int) * NoDepthPts2dLabelIdx.size());
	cudaMalloc((float**)&dev_depth_mat, sizeof(float) * HEIGHT*WIDTH);
	cudaMalloc((float**)&dev_sp_label_plane_arrs, sizeof(float) * sp_label_plane_arrs.size());
	cudaMalloc((cv::Point2f**)&dev_no_depth_pts2d, sizeof(cv::Point2f) * NoDepthPts2d.size());

	// �����ݿ�����GPU��
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

	// Kernel�������������
	int threads_per_block = 128;
	int blocks_per_grid = NoDepthPts2d.size() / threads_per_block + 1;
	MRFKernel3 << <blocks_per_grid, threads_per_block >> > (
		dev_depth_mat,
		dev_labels,
		dev_sp_labels,
		dev_no_depth_pts2d,  // �������pt2d������
		dev_NoDepthPts2dLabelIdx,  // ÿ��pt2d���Ӧ��label idx
		dev_sp_label_plane_arrs,
		dev_sp_label_neighs_idx,  // label idx��Ӧ��neighbor label idx���鴮��
		dev_sp_label_neigh_num,  // label idx���õ�neighbor��������
		Radius, Beta,
		WIDTH, HEIGHT,
		(int)NoDepthPts2d.size(), (int)SPLabels.size(),
		dev_sp_labels_ret);
	std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
	cudaDeviceSynchronize();  // host�ȴ�deviceͬ��

	// GPU�����ݷ���
	cudaMemcpy(NoDepthPt2DSPLabelsRet.data(),
		dev_sp_labels_ret,
		sizeof(int) * NoDepthPts2d.size(),
		cudaMemcpyDeviceToHost);

	// �ͷ�GPU���ڴ�
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
	int* d_sp_labels,  // ���к�ѡƽ��label����
	const float* d_sp_depths,
	cv::Point2f* d_no_depth_pts2d,  // �������pt2d������
	int* d_NoDepthPts2dLabelIdx,  // ÿ��pt2d���Ӧ��label idx
	float* d_sp_label_plane_arrs,  // label��Ӧ��ƽ�淽������(����K_inv_arr)
	int* d_sp_label_neighs_idx,  // label_idx��Ӧ��neighbors(label idx)
	int* d_sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
	int radius, float beta,
	int WIDTH, int HEIGHT,  // ͼ����
	int Num_Pts2d, int Num_Labels,  // pt2d��ĸ�����label�ĸ���
	int* d_sp_labels_ret)
{
	// ������pt2d���� 
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < Num_Pts2d)
	{
		// ��ʼ�����ĵ�ָ��
		cv::Point2f* ptr_pt2d = d_no_depth_pts2d + tid;

		// ����ԭʼ���ĵ����ֵ
		const float orig_depth = d_depth_mat[int(ptr_pt2d[0].y)*WIDTH + int(ptr_pt2d[0].x)];

		// �����Ԫ�Χ
		int y_begin = int((*ptr_pt2d).y) - radius;
		y_begin = y_begin >= 0 ? y_begin : 0;
		int y_end = int((*ptr_pt2d).y) + radius;
		y_end = y_end <= HEIGHT - 1 ? y_end : HEIGHT - 1;

		int x_begin = int((*ptr_pt2d).x) - radius;
		x_begin = x_begin >= 0 ? x_begin : 0;
		int x_end = int((*ptr_pt2d).x) + radius;
		x_end = x_end <= WIDTH - 1 ? x_end : WIDTH - 1;

		// ȡ����ڲξ������
		const float* K_inv_arr = d_sp_label_plane_arrs + (Num_Labels << 2);

		// ���ĵ��label idx
		int the_label_idx = d_NoDepthPts2dLabelIdx[tid];

		// ----- �����ʼ������(���ĵ�ȡԭlabel������, ��Neighbors�Ƚ�)
		// ��ʼ����С�����������label(��ʼ��Ϊ���ĵ��������label)
		int label_idx = the_label_idx;
		int best_label_idx = the_label_idx;

		// ----- �����Ԫ������
		float energy_binary = 0.0f;  // ��ʼ����Ԫ������
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

		// ----- ����һԪ������
		// �����ȡlabel_iʱ, ���ĵ�����ֵ
		float* plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
		float the_depth = -plane_arr[3] /
			(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y + K_inv_arr[2])
				+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y +K_inv_arr[5])
				+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y +K_inv_arr[8]));
		//printf("The depth of center point: %.3f\n", the_depth);

		// ����depths��offset
		int offset = 0;
		for (int j = 0; j < label_idx; ++j)
		{
			offset += d_pts2d_size[j];
		}

		// ��label��Ӧ�����ֵ���λ��
		const float* ptr_depths = d_sp_depths + offset;

		// --- �����label��Ӧ��depth�ľ�ֵ�ͱ�׼��
		float depth_mean = 0.0f, depth_std = 0.0f;

		// �����label��Ӧ����Ⱦ�ֵ
		for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
		{
			depth_mean += ptr_depths[k];
		}
		depth_mean /= float(d_pts2d_size[label_idx]);

		// �����label��Ӧ��depth��׼��
		for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
		{
			depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
		}
		depth_std /= float(d_pts2d_size[label_idx]);
		depth_std = sqrtf(depth_std);

		// ����һԪ����
		float energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
			+ (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std)
			+ fabsf(the_depth - orig_depth);

		// ��ʼ����С����Ϊ���ĵ�ȡԭlabelʱ������
		float energy_min = 2.0f*energy_binary + energy_unary;

		// ---- ����Neighbors������
		// ����neighbor offset
		offset = 0;
		for (int i = 0; i < the_label_idx; ++i)
		{
			offset += d_sp_label_neigh_num[i];
		}

		// �������ĵ��neigh label idx��ʼ��ָ��
		const int* ptr_neigh = d_sp_label_neighs_idx + offset;

		// ����ÿһ��Neighbor
		for (int i = 0; i < d_sp_label_neigh_num[the_label_idx]; ++i)
		{
			label_idx = ptr_neigh[i];

			// ----- �����Ԫ������
			energy_binary = 0.0f;  // ��ʼ����Ԫ������
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

			// ----- ����һԪ������
			// �����ȡlabel_iʱ, ���ĵ�����ֵ
			plane_arr = d_sp_label_plane_arrs + (label_idx << 2);
			the_depth = -plane_arr[3] /
				(plane_arr[0] * (K_inv_arr[0] * ptr_pt2d[0].x + K_inv_arr[1] * ptr_pt2d[0].y +K_inv_arr[2])
					+ plane_arr[1] * (K_inv_arr[3] * ptr_pt2d[0].x + K_inv_arr[4] * ptr_pt2d[0].y +K_inv_arr[5])
					+ plane_arr[2] * (K_inv_arr[6] * ptr_pt2d[0].x + K_inv_arr[7] * ptr_pt2d[0].y +K_inv_arr[8]));
			//printf("The depth of center point: %.3f\n", the_depth);

			// ����depths��offset
			offset = 0;
			for (int j = 0; j < label_idx; ++j)
			{
				offset += d_pts2d_size[j];
			}

			// ��label��Ӧ�����ֵ���λ��
			const float* ptr_depths = d_sp_depths + offset;

			// --- �����label��Ӧ��depth�ľ�ֵ�ͱ�׼��
			depth_mean = 0.0f;
			depth_std = 0.0f;

			// �����label��Ӧ����Ⱦ�ֵ
			for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
			{
				depth_mean += ptr_depths[k];
			}
			depth_mean /= float(d_pts2d_size[label_idx]);

			// �����label��Ӧ��depth��׼��
			for (int k = 0; k < d_pts2d_size[label_idx]; ++k)
			{
				depth_std += (ptr_depths[k] - depth_mean) * (ptr_depths[k] - depth_mean);
			}
			depth_std /= float(d_pts2d_size[label_idx]);
			depth_std = sqrtf(depth_std);

			// ����һԪ����
			energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
				+ (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std)
				+ fabsf(the_depth - orig_depth);

			//// for debug...
			//if (tid % 1000 == 0)
			//{
			//	std::printf("energy unary: %.3f, energy_binary: %.3f\n",
			//		energy_unary, energy_binary);
			//}

			// �������յ�����
			const float energy = 2.0f*energy_binary + energy_unary;
			if (energy < energy_min)
			{
				energy_min = energy;
				best_label_idx = label_idx;
			}
		}

		// д���µ�label
		d_sp_labels_ret[tid] = d_sp_labels[best_label_idx];
	}
}

int MRFGPU4(const cv::Mat& depth_mat,
	const cv::Mat& labels,  // ÿ��pt2d���label
		const std::vector<int>& SPLabels,  // ��ѡ�ĵ�labels
		const std::vector<float>& SPLabelDepths,  // ����label���е�����ֵ
		const std::vector<int>& pts2d_size, // ����label���е��pt2d�����
		const std::vector<cv::Point2f>& NoDepthPts2d,  // �����ֵpt2d��
		const std::vector<int>& NoDepthPts2dLabelIdx,  // ÿ��pt2d���label idx
		const std::vector<float>& sp_label_plane_arrs,  // ����label���е��ƽ�淽��
		const std::vector<int>& sp_label_neighs_idx,  // ÿ��label_idx��Ӧ��sp_label idx
		const std::vector<int>& sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
		const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
		std::vector<int>& NoDepthPt2DSPLabelsRet)
	{
		assert(HEIGHT == labels.rows && WIDTH == labels.cols);
		assert(std::accumulate(pts2d_size.begin(), pts2d_size.end(), 0) == HEIGHT * WIDTH);
		assert(SPLabels.size() == pts2d_size.size() == sp_label_neigh_num.size());
		assert(NoDepthPts2d.size() == NoDepthPts2dLabelIdx.size());

		// ��GPU�Ϸ����ڴ�
		int* dev_labels, *dev_pts2d_size,
			*dev_sp_labels, *dev_sp_labels_ret,
			*dev_sp_label_neighs_idx, *dev_sp_label_neigh_num;
		float* dev_depth_mat, *dev_sp_depths, *dev_sp_label_plane_arrs;
		cv::Point2f* dev_no_depth_pts2d;
		int* dev_NoDepthPts2dLabelIdx;

		cudaMalloc((int**)&dev_labels, sizeof(int) * HEIGHT*WIDTH);
		cudaMalloc((int**)&dev_pts2d_size, sizeof(int) * pts2d_size.size());
		cudaMalloc((int**)&dev_sp_labels, sizeof(int) * SPLabels.size());  // ��ѡlabel����
		cudaMalloc((int**)&dev_sp_labels_ret, sizeof(int) * NoDepthPts2d.size());  // ��������
		cudaMalloc((int**)&dev_sp_label_neighs_idx, sizeof(int) * sp_label_neighs_idx.size());
		cudaMalloc((int**)&dev_sp_label_neigh_num, sizeof(int) * sp_label_neigh_num.size());
		cudaMalloc((int**)&dev_NoDepthPts2dLabelIdx, sizeof(int) * NoDepthPts2dLabelIdx.size());
		cudaMalloc((float**)&dev_depth_mat, sizeof(float) * HEIGHT*WIDTH);
		cudaMalloc((float**)&dev_sp_depths, sizeof(float) * SPLabelDepths.size());
		cudaMalloc((float**)&dev_sp_label_plane_arrs, sizeof(float) * sp_label_plane_arrs.size());
		cudaMalloc((cv::Point2f**)&dev_no_depth_pts2d, sizeof(cv::Point2f) * NoDepthPts2d.size());

		// �����ݿ�����GPU��
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

		// Kernel�������������
		int threads_per_block = 128;
		int blocks_per_grid = NoDepthPts2d.size() / threads_per_block + 1;
		MRFKernel4 << <blocks_per_grid, threads_per_block >> > (
			dev_depth_mat,
			dev_labels,
			dev_pts2d_size,
			dev_sp_labels,
			dev_sp_depths,
			dev_no_depth_pts2d,  // �������pt2d������
			dev_NoDepthPts2dLabelIdx,  // ÿ��pt2d���Ӧ��label idx
			dev_sp_label_plane_arrs,
			dev_sp_label_neighs_idx,  // label idx��Ӧ��neighbor label idx���鴮��
			dev_sp_label_neigh_num,  // label idx���õ�neighbor��������
			Radius, Beta,
			WIDTH, HEIGHT,
			(int)NoDepthPts2d.size(), (int)SPLabels.size(),
			dev_sp_labels_ret);
		std::printf("Starting %d threads\n", threads_per_block*blocks_per_grid);
		cudaDeviceSynchronize();  // host�ȴ�deviceͬ��

		// GPU�����ݷ���
		cudaMemcpy(NoDepthPt2DSPLabelsRet.data(),
			dev_sp_labels_ret,
			sizeof(int) * NoDepthPts2d.size(),
			cudaMemcpyDeviceToHost);

		// �ͷ�GPU���ڴ�
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
	cv::Point2f* d_pts2d_has_no_depth_jbu,  // ��Ҫ�����pt2d������
	int* d_sp_labels_idx_jbu,  // ÿ�������pt2d���Ӧ��label idx
	cv::Point2f* d_pts2d_has_depth_jbu,
	int* d_sp_has_depth_pt2ds_num,
	float* d_sigmas_s_jbu,  // ÿ��label��Ӧһ��sigma_s
	int WIDTH, int N_pts2d_no_depth,  // ���������ݸ���
	float* d_depths_ret)
{
	// ������pt2d���� 
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N_pts2d_no_depth)
	{
		// ��ʼ�����ֵ��Ȩ��
		float depth = 0.0f, sum_depth = 0.0f, sum_weight = 0.0f;

		// ������pt2d��ָ��
		cv::Point2f* ptr_pt2d_no_depth = d_pts2d_has_no_depth_jbu + tid;

		// ----- ��ȡ�����ֵ���Ӧ�������ֵ�㼯��ָ��
		// ����ָ��ƫ����
		int offset = 0, label_idx = d_sp_labels_idx_jbu[tid];
		for (int i = 0; i < label_idx; ++i)
		{
			offset += d_sp_has_depth_pt2ds_num[i];
		}
		cv::Point2f* ptr_pts2d_depth = d_pts2d_has_depth_jbu + offset;

		// ----- ����ÿһ����depth��pt2d���Ȩ��
		// ȡ�����ֵpt2d���srcָ��
		uchar* ptr_color_no_depth = d_src +
			(int(ptr_pt2d_no_depth[0].y)*WIDTH + int(ptr_pt2d_no_depth[0].x)) * 3;

		for (int i = 0; i < d_sp_has_depth_pt2ds_num[label_idx]; ++i)
		{
			// --- ����2D�ռ����Ȩ��
			float delta_dist = sqrtf(
				  (ptr_pt2d_no_depth[0].x - ptr_pts2d_depth[i].x)
				* (ptr_pt2d_no_depth[0].x - ptr_pts2d_depth[i].x)
				+ (ptr_pt2d_no_depth[0].y - ptr_pts2d_depth[i].y)
				* (ptr_pt2d_no_depth[0].y - ptr_pts2d_depth[i].y)
			);
			float space_weight = expf(-0.5f*delta_dist*delta_dist
				/ (d_sigmas_s_jbu[label_idx] * d_sigmas_s_jbu[label_idx]));

			// --- ����ɫ��Ȩ��
			// ���㵱ǰ�����ֵ��pt2d���srcָ��
			uchar* ptr_color_depth = d_src +
				(int(ptr_pts2d_depth[i].y)*WIDTH + int(ptr_pts2d_depth[i].x)) * 3;

			// ����delta_color: L1 norm of color difference
			float delta_color = fabsf(ptr_color_no_depth[0] - ptr_color_depth[0]) +
				fabsf(ptr_color_no_depth[1] - ptr_color_depth[1]) +
				fabsf(ptr_color_no_depth[2] - ptr_color_depth[2]);
			float color_weight = expf(-0.5f*delta_color*delta_color / 16384.0f);  //128*128(color��sigmaֵ)
			float weight = space_weight * color_weight;

			// ͳ��Ȩ��ֵ
			float depth_neighbor = d_depth_mat[int(ptr_pts2d_depth[i].y)*WIDTH 
				+ int(ptr_pts2d_depth[i].x)];

			sum_depth += weight * depth_neighbor;
			sum_weight += weight;
		}

		// to prevent overflow
		depth = sum_depth / (0.00001f+sum_weight);

		// д��������
		d_depths_ret[tid] = depth;
	}
}

int JBUSPGPU(const cv::Mat& src,
	const cv::Mat& depth_mat,
	const std::vector<cv::Point2f>& pts2d_has_no_depth_jbu,  // �������pt2d��
	const std::vector<int>& sp_labels_idx_jbu,  // ÿ���������pt2d���Ӧ��label_idx
	const std::vector<cv::Point2f>& pts2d_has_depth_jbu,  // ��������JBU�����ֵ��pt2d��
	const std::vector<int>& sp_has_depth_pt2ds_num,  // ÿ��label_idx��Ӧ�������ֵpt2d����
	const std::vector<float>& sigmas_s_jbu, // // ÿ��label_idx��Ӧ��sigma_s
	std::vector<float>& depths_ret)
{
	assert(pts2d_has_no_depth_jbu.size() == sp_labels_idx_jbu.size()
		== depths_ret.size());
	assert(sp_has_depth_pt2ds_num.size() == sigmas_s_jbu.size());

	const int WIDTH = depth_mat.cols;
	const int HEIGHT = depth_mat.rows;

	// ��GPU�Ϸ����ڴ�
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

	// �����ݿ�����GPU��
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

	// Kernel�������������
	int threads_per_block = 128;  // ��������Ϊblock����߳���
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
	cudaDeviceSynchronize();  // host�ȴ�deviceͬ��

	// GPU�����ݷ���
	cudaMemcpy(depths_ret.data(),
		dev_depths_ret,
		sizeof(float) * depths_ret.size(),
		cudaMemcpyDeviceToHost);

	// �ͷ�GPU���ڴ�
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
//		printf("CUDA�����汾:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
//		cudaRuntimeGetVersion(&runtime_version);
//		printf("CUDA����ʱ�汾:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
//		printf("�豸��������:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
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
