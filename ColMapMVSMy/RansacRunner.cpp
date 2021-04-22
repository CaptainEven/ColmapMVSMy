#include "RansacRunner.h"
#include <iostream>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


RansacRunner::RansacRunner(const float tolerance,
	const int num_sample,
	const int num_sub_sample,
	const int num_iter,
	const int total_iter_num,
	const float inlier_rate)
{
	this->m_tolerance = tolerance;

	this->m_num_iter = num_iter;  // 初始化的迭代次数
	this->m_num_iter_orig = num_iter;
	this->m_total_iter_count = 0;  // 总的迭代次数
	this->m_total_iter_num = total_iter_num;

	this->m_num_sample = num_sample;
	this->m_num_sub_sample = num_sub_sample;

	assert(this->m_num_sample >= 3);
	assert(this->m_num_sample >= this->m_num_sub_sample);

	// 重设迭代次数
	if (-1 == this->m_num_sub_sample)
	{
		this->m_num_sub_sample = int(0.1f * (float)this->m_num_sample + 0.5f);  // using 10%
		this->m_num_sub_sample = m_num_sub_sample >= 3 ? m_num_sub_sample : 3;
		this->m_num_sub_sample = m_num_sub_sample <= m_num_sample ? m_num_sub_sample : m_num_sample;
	}
	if (this->m_num_sub_sample == this->m_num_sample)
	{
		this->m_num_iter = 1;
		this->m_total_iter_num = 1;
	}

	// 为3D点子集预先开辟内存
	if (this->m_num_sub_sample > 3)
	{
		this->m_subsets.reserve(this->m_num_sub_sample);
		this->m_subsets.resize(this->m_num_sub_sample);
	}

	this->m_inlier_rate = inlier_rate;
	this->m_num_inliers = 0;
	this->m_cost = FLT_MAX;

	// 是否达到指定inlier rate的标志
	this->m_done = false;

	memset(this->m_plane, 0, sizeof(float) * 4);
	memset(this->m_eigen_vals, 0, sizeof(float) * 3);  // 初始化特征值
	memset(this->m_eigen_vect, 0, sizeof(float) * 9);  // 初始化特征向量
}

int RansacRunner::GetMinSubSets(const std::vector<cv::Point3f>& Pts3D)
{
	int id_0 = 0, id_1 = 0, id_2 = 0;

	// make sure to fetch 3 different 3D pts
	while (id_0 == id_1 || id_0 == id_2 || id_1 == id_2)
	{
		id_0 = rand() % int(this->m_num_sample);
		id_1 = rand() % int(this->m_num_sample);
		id_2 = rand() % int(this->m_num_sample);
	}

	this->m_min_subsets[0] = Pts3D[id_0];
	this->m_min_subsets[1] = Pts3D[id_1];
	this->m_min_subsets[2] = Pts3D[id_2];

	return 0;
}

int RansacRunner::GetSubSets(const std::vector<cv::Point3f>& Pts3D)
{
	for (int i = 0; i < this->m_num_sub_sample; ++i)
	{
		int id = rand() % int(this->m_num_sample);
		this->m_subsets[i] = Pts3D[id];
	}

	return 0;
}

int RansacRunner::UpdateNumIters(double p, double ep, int modelPoints, int maxIters)
{
	if (modelPoints <= 0)
	{
		printf("[Err]: the number of model points should be positive");
	}

	p = MAX(p, 0.);
	p = MIN(p, 1.);
	ep = MAX(ep, 0.);
	ep = MIN(ep, 1.);

	// avoid inf's & nan's
	double num = MAX(1. - p, DBL_MIN);
	double denom = 1. - std::pow(1. - ep, modelPoints);
	if (denom < DBL_MIN)
	{
		return 0;
	}

	num = std::log(num);
	denom = std::log(denom);

	return denom >= 0 || -num >= maxIters * (-denom) ? maxIters : cvRound(num / denom);
}

int RansacRunner::PlaneFitBy3Pts(const cv::Point3f* pts, float* plane_arr)
{
	const float& x1 = pts[0].x;
	const float& y1 = pts[0].y;
	const float& z1 = pts[0].z;

	const float& x2 = pts[1].x;
	const float& y2 = pts[1].y;
	const float& z2 = pts[1].z;

	const float& x3 = pts[2].x;
	const float& y3 = pts[2].y;
	const float& z3 = pts[2].z;

	const float center_x = (x1 + x2 + x3) / 3.0f;
	const float center_y = (y1 + y2 + y3) / 3.0f;
	const float center_z = (z1 + z2 + z3) / 3.0f;

	float A = (y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1);
	float B = (z2 - z1)*(x3 - x1) - (z3 - z1)*(x2 - x1);
	float C = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);

	const float DENOM = std::sqrtf(A * A + B * B + C * C);

	// 判断三点是否共线
	if (DENOM < 1e-12)
	{
		//printf("[Warning]: 3 Points may near colinear\n");
		return -1;
	}

	A /= DENOM;
	B /= DENOM;
	C /= DENOM;
	float D = -(A*center_x + B * center_y + C * center_z);

	if (_isnanf(A) || _isnanf(B) || _isnanf(C) || _isnanf(D))
	{
		printf("[Err]: nan plane parameters\n");
		return -1;
	}

	plane_arr[0] = A;
	plane_arr[1] = B;
	plane_arr[2] = C;
	plane_arr[3] = D;

	return 0;
}

int RansacRunner::PlaneFitBy3Pts(float* plane_arr)
{
	const float& x1 = this->m_min_subsets[0].x;
	const float& y1 = this->m_min_subsets[0].y;
	const float& z1 = this->m_min_subsets[0].z;

	const float& x2 = this->m_min_subsets[1].x;
	const float& y2 = this->m_min_subsets[1].y;
	const float& z2 = this->m_min_subsets[1].z;

	const float& x3 = this->m_min_subsets[2].x;
	const float& y3 = this->m_min_subsets[2].y;
	const float& z3 = this->m_min_subsets[2].z;

	const float center_x = (x1 + x2 + x3) / 3.0f;
	const float center_y = (y1 + y2 + y3) / 3.0f;
	const float center_z = (z1 + z2 + z3) / 3.0f;

	float A = (y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1);
	float B = (z2 - z1)*(x3 - x1) - (z3 - z1)*(x2 - x1);
	float C = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);

	const float DENOM = std::sqrtf(A * A + B * B + C * C);

	// 判断三点是否共线
	if (DENOM < 1e-12)
	{
		//printf("[Warning]: 3 Points may near colinear\n");
		return -1;
	}

	A /= DENOM;
	B /= DENOM;
	C /= DENOM;
	float D = -(A*center_x + B * center_y + C * center_z);

	//if (_isnanf(A) || _isnanf(B) || _isnanf(C) || _isnanf(D))
	//{
	//	printf("[Err]: nan plane parameters\n");
	//	return -1;
	//}

	plane_arr[0] = A;
	plane_arr[1] = B;
	plane_arr[2] = C;
	plane_arr[3] = D;

	return 0;
}


// https://blog.csdn.net/qq_29912325/article/details/106917141
int RansacRunner::PlaneFitBy3PtsEi(float* plane_arr, float * ei_vals, float * ei_vects)
{
	const float& x1 = this->m_min_subsets[0].x;
	const float& y1 = this->m_min_subsets[0].y;
	const float& z1 = this->m_min_subsets[0].z;

	const float& x2 = this->m_min_subsets[1].x;
	const float& y2 = this->m_min_subsets[1].y;
	const float& z2 = this->m_min_subsets[1].z;

	const float& x3 = this->m_min_subsets[2].x;
	const float& y3 = this->m_min_subsets[2].y;
	const float& z3 = this->m_min_subsets[2].z;

	const float center_x = (x1 + x2 + x3) / 3.0f;
	const float center_y = (y1 + y2 + y3) / 3.0f;
	const float center_z = (z1 + z2 + z3) / 3.0f;

	float A = (y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1);
	float B = (z2 - z1)*(x3 - x1) - (z3 - z1)*(x2 - x1);
	float C = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);

	const float DENOM = std::sqrtf(A * A + B * B + C * C);

	//// 判断三点是否共线
	//if (DENOM < 1e-12)
	//{
	//	//printf("[Warning]: 3 Points may near colinear\n");
	//	return -1;
	//}

	A /= DENOM;
	B /= DENOM;
	C /= DENOM;

	// 确保法向量指向相机
	if (A*center_x + B * center_y + C * center_z > 0.0f)
	{
		A = -A;
		B = -B;
		C = -C;
	}

	float D = -(A*center_x + B * center_y + C * center_z);

	if (_isnanf(A) || _isnanf(B) || _isnanf(C) || _isnanf(D))
	{
		printf("[Err]: nan plane parameters\n");
		return -1;
	}
	if (0.0f == A && 0.0f == B && 0.0f == C && 0.0f == D)
	{
		printf("[Err]: All zeros plane parameters\n");
		return -1;
	}

	plane_arr[0] = A;
	plane_arr[1] = B;
	plane_arr[2] = C;
	plane_arr[3] = D;

	// PCA分解获取特征值特征向量
	this->PlaneFitPCAEi(std::vector<cv::Point3f>(m_min_subsets, m_min_subsets + 3),
		ei_vals, ei_vects);
	memcpy(ei_vects, plane_arr, sizeof(float) * 3);

	return 0;
}

// 3D平面方程拟合(最小二乘法)写成Ax=B的形式: aX + bY + Z + c = 0 or aX + bY + c = -Z
// https://blog.csdn.net/qq_45427038/article/details/100139330
int RansacRunner::PlaneFitOLS1(float* plane_arr)
{// 输出结果向量X: a, b, c
	cv::Mat A(this->m_num_sub_sample, 3, CV_32F);
	cv::Mat B(this->m_num_sub_sample, 1, CV_32F);

	// 系数矩阵A和结果向量b初始化
	float sum_x = 0.0f;
	float sum_y = 0.0f;
	float sum_z = 0.0f;
	for (int i = 0; i < this->m_num_sub_sample; ++i)
	{
		A.at<float>((int)i, 0) = this->m_subsets[i].x;
		A.at<float>((int)i, 1) = this->m_subsets[i].y;
		A.at<float>((int)i, 2) = 1.0f;

		B.at<float>((int)i, 0) = -this->m_subsets[i].z;

		sum_x += this->m_subsets[i].x;
		sum_y += this->m_subsets[i].y;
		sum_z += this->m_subsets[i].z;
	}

	// 计算空间点均值
	const float ave_x = sum_x / float(this->m_num_sub_sample);
	const float ave_y = sum_y / float(this->m_num_sub_sample);
	const float ave_z = sum_z / float(this->m_num_sub_sample);

	// 解线性方程组AX=B: x = (A' * A)^-1 * A' * B
	cv::Mat X = (A.t() * A).inv() * A.t() * B;  // 3×1
	float a = X.at<float>(0, 0);
	float b = X.at<float>(1, 0);
	float c = X.at<float>(2, 0);

	// 确定正确的法向方向: 确保normal指向camera
	if (a * ave_x + b * ave_y + -1.0f * ave_z > 0.0f)
	{
		a = -a;
		b = -b;
		//c = -c;
	}

	const float DENOM = std::sqrtf(a * a + b * b + 1.0f);  // sqrt(a^2+b^2+1^2)

	X.at<float>(0, 0) /= DENOM;  // a
	X.at<float>(1, 0) /= DENOM;  // b
	X.at<float>(2, 0) /= DENOM;  // c

	plane_arr[0] = X.at<float>(0, 0);  // a
	plane_arr[1] = X.at<float>(1, 0);  // b
	plane_arr[2] = 1.0f / DENOM;       // 1
	plane_arr[3] = X.at<float>(2, 0);  // c

	return 0;
}

// 超定方程最小二乘解
// 写成Ax=B的形式: aX + bY + c = Z <=> aX + bY - Z + c = 0
// 推导 https://blog.csdn.net/konglingshneg/article/details/82585868
int RansacRunner::PlaneFitOLS2(float* plane_arr)
{
	cv::Mat A(3, 3, CV_32F);
	cv::Mat B(3, 1, CV_32F);

	// 构建系数矩阵A和结果向量B
	float sum_xx = 0.0f, sum_yy = 0.0f, sum_xy = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
	float sum_xz = 0.0f, sum_yz = 0.0f, sum_z = 0.0f;
	for (auto pt : this->m_subsets)
	{
		sum_xx += pt.x * pt.x;
		sum_yy += pt.y * pt.y;
		sum_xy += pt.x * pt.y;
		sum_x += pt.x;
		sum_y += pt.y;

		sum_xz += pt.x * pt.z;
		sum_yz += pt.y * pt.z;
		sum_z += pt.z;
	}
	A.at<float>(0, 0) = sum_xx;
	A.at<float>(0, 1) = sum_xy;
	A.at<float>(0, 2) = sum_x;
	A.at<float>(1, 0) = sum_xy;
	A.at<float>(1, 1) = sum_yy;
	A.at<float>(1, 2) = sum_y;
	A.at<float>(2, 0) = sum_x;
	A.at<float>(2, 1) = sum_y;
	A.at<float>(2, 2) = (float)this->m_num_sub_sample;

	B.at<float>(0, 0) = sum_xz;
	B.at<float>(1, 0) = sum_yz;
	B.at<float>(2, 0) = sum_z;

	// 计算空间点均值
	const float ave_x = sum_x / float(this->m_subsets.size());
	const float ave_y = sum_y / float(this->m_subsets.size());
	const float ave_z = sum_z / float(this->m_subsets.size());

	// 解线性方程组AX=B: x = (A' * A)^-1 * A' * B
	cv::Mat X = (A.t() * A).inv() * A.t() * B;  // 3×1
	float a = X.at<float>(0, 0);
	float b = X.at<float>(1, 0);
	float c = X.at<float>(2, 0);

	// 确定正确的法向方向: 确保normal指向camera
	if (a * ave_x + b * ave_y + c * ave_z > 0.0f)
	{
		a = -a;
		b = -b;
		c = -c;
	}
	const float DENOM = std::sqrtf(a * a + b * b + 1.0f);  // sqrt(a^2+b^2+(-1)^2)

	X.at<float>(0, 0) /= DENOM;  // a
	X.at<float>(1, 0) /= DENOM;  // b
	X.at<float>(2, 0) /= DENOM;  // c

	plane_arr[0] = X.at<float>(0, 0);  // a
	plane_arr[1] = X.at<float>(1, 0);  // b
	plane_arr[2] = -1.0f / DENOM;      // -1
	plane_arr[3] = X.at<float>(2, 0);  // c

	return 0;
}

// 3D点云空间平面拟合SVD分解方法
// 推导 https://blog.csdn.net/oChenWen/article/details/84373582
int RansacRunner::PlaneFitSVD(float* plane_arr)
{
	float ave_x = 0.0f, ave_y = 0.0f, ave_z = 0.0f;
	for (auto pt : this->m_subsets)
	{
		ave_x += pt.x;
		ave_y += pt.y;
		ave_z += pt.z;
	}
	ave_x /= float(this->m_num_sub_sample);
	ave_y /= float(this->m_num_sub_sample);
	ave_z /= float(this->m_num_sub_sample);

	Eigen::MatrixXf A(this->m_num_sub_sample, 3);  // m×3的系数矩阵
	for (int i = 0; i < this->m_num_sub_sample; ++i)
	{
		A(i, 0) = this->m_subsets[i].x - ave_x;
		A(i, 1) = this->m_subsets[i].y - ave_y;
		A(i, 2) = this->m_subsets[i].z - ave_z;
	}
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3f V = svd.matrixV();

	// 取V矩阵最后一列
	float& a = V(0, 2);
	float& b = V(1, 2);
	float& c = V(2, 2);

	// 确定正确的法向方向: 确保normal指向camera
	if (a * ave_x + b * ave_y + c * ave_z > 0.0f)
	{
		a = -a;
		b = -b;
		c = -c;
	}
	float d = -(a*ave_x + b * ave_y + c * ave_z);

	float DENOM = sqrtf(a*a + b * b + c * c);
	a /= DENOM;
	b /= DENOM;
	c /= DENOM;
	d /= DENOM;

	plane_arr[0] = a;
	plane_arr[1] = b;
	plane_arr[2] = c;
	plane_arr[3] = d;

	return 0;
}

// 3D点云空间平面拟合PCA分解方法
// 推导 https://www.jianshu.com/p/faa9953213dd
int RansacRunner::PlaneFitPCA(float* plane_arr,
	float* ei_vals, float* ei_vects)
{
	float ave_x = 0.0f, ave_y = 0.0f, ave_z = 0.0f;
	for (auto pt : this->m_subsets)
	{
		ave_x += pt.x;
		ave_y += pt.y;
		ave_z += pt.z;
	}
	ave_x /= float(this->m_num_sub_sample);
	ave_y /= float(this->m_num_sub_sample);
	ave_z /= float(this->m_num_sub_sample);

	// 求协方差矩阵A
	Eigen::Matrix3f A;
	float sum_xx = 0.0f, sum_yy = 0.0f, sum_zz = 0.0f,
		sum_xy = 0.0f, sum_xz = 0.0f, sum_yz = 0.0f;
	for (auto pt : this->m_subsets)
	{
		sum_xx += (pt.x - ave_x) * (pt.x - ave_x);
		sum_yy += (pt.y - ave_y) * (pt.y - ave_y);
		sum_zz += (pt.z - ave_z) * (pt.z - ave_z);

		sum_xy += (pt.x - ave_x) * (pt.y - ave_y);
		sum_xz += (pt.x - ave_x) * (pt.z - ave_z);
		sum_yz += (pt.y - ave_y) * (pt.z - ave_z);
	}
	A(0, 0) = sum_xx / float(this->m_num_sub_sample);  // 其实, 没必要求均值
	A(0, 1) = sum_xy / float(this->m_num_sub_sample);
	A(0, 2) = sum_xz / float(this->m_num_sub_sample);
	A(1, 0) = sum_xy / float(this->m_num_sub_sample);
	A(1, 1) = sum_yy / float(this->m_num_sub_sample);
	A(1, 2) = sum_yz / float(this->m_num_sub_sample);
	A(2, 0) = sum_xz / float(this->m_num_sub_sample);
	A(2, 1) = sum_yz / float(this->m_num_sub_sample);
	A(2, 2) = sum_zz / float(this->m_num_sub_sample);

	// 求协方差矩阵A的特征值和特征向量
	Eigen::EigenSolver<Eigen::Matrix3f> ES(A);
	Eigen::MatrixXcf eigen_vals = ES.eigenvalues();
	Eigen::MatrixXcf eigen_vects = ES.eigenvectors();
	Eigen::MatrixXf eis = eigen_vals.real();
	Eigen::MatrixXf vects = eigen_vects.real();

	//std::cout << eigen_vals << std::endl;
	//std::cout << eigens << std::endl << std::endl;
	//std::cout << vects << std::endl;
	//std::cout << "Eigen value sum: " << eigens.sum() << std::endl;

	// 求最小特征值对应的特征向量
	Eigen::MatrixXf::Index min_idx, max_idx;
	eis.rowwise().sum().minCoeff(&min_idx);
	eis.rowwise().sum().maxCoeff(&max_idx);

	//int row_idx, col_idx;
	//eigens.maxCoeff(&row_idx, &col_idx);
	//std::cout << row_idx << " " << col_idx << std::endl;
	//eigens.minCoeff(&row_idx, &col_idx);
	//std::cout << row_idx << " " << col_idx << std::endl;
	//auto min_eigen_vect = ES.eigenvectors().col(min_idx).real();
	//std::cout << "min eigen vector:\n" << min_eigen_vect << std::endl;

	// ----- 对特征值(特征向量)排序：从小到大
	int mid_idx = 0;
	if (0 == (int)min_idx)
	{
		mid_idx = max_idx == 1 ? 2 : 1;
	}
	else if (1 == (int)min_idx)
	{
		mid_idx = max_idx == 0 ? 2 : 0;
	}
	else
	{
		mid_idx = max_idx == 0 ? 1 : 0;
	}

	// 最小特征值对用的特征向量
	float& a = vects(0, min_idx);
	float& b = vects(1, min_idx);
	float& c = vects(2, min_idx);

	// 确定正确的法向方向: 确保normal指向camera
	if (a * ave_x + b * ave_y + c * ave_z > 0.0f)
	{
		a = -a;
		b = -b;
		c = -c;
	}

	// 最小特征向量(法向量)L2归一化
	const float DENOM = sqrtf(a*a + b * b + c * c);
	a /= DENOM;
	b /= DENOM;
	c /= DENOM;

	// 计算参数d
	float d = -(a*ave_x + b * ave_y + c * ave_z);

	// 返回计算的平面方程
	plane_arr[0] = a;
	plane_arr[1] = b;
	plane_arr[2] = c;
	plane_arr[3] = d;

	// 返回特征值: 从小到大排列
	ei_vals[0] = eis(min_idx);
	ei_vals[1] = eis(mid_idx);
	ei_vals[2] = eis(max_idx);

	ei_vects[0] = vects(0, min_idx);  // a
	ei_vects[1] = vects(1, min_idx);  // b
	ei_vects[2] = vects(2, min_idx);  // c

	ei_vects[3] = vects(0, mid_idx);  // 切平面特征向量, 切向分量 1
	ei_vects[4] = vects(1, mid_idx);
	ei_vects[5] = vects(2, mid_idx);

	ei_vects[6] = vects(0, max_idx);  // 切平面特征向量, 切向分量 2
	ei_vects[7] = vects(1, max_idx);
	ei_vects[8] = vects(2, max_idx);

	return 0;
}

int RansacRunner::PlaneFitPCAEi(const std::vector<cv::Point3f>& pts3d,
	float* ei_vals, float* ei_vects)
{
	float ave_x = 0.0f, ave_y = 0.0f, ave_z = 0.0f;
	for (auto pt : pts3d)
	{
		ave_x += pt.x;
		ave_y += pt.y;
		ave_z += pt.z;
	}
	ave_x /= float(pts3d.size());
	ave_y /= float(pts3d.size());
	ave_z /= float(pts3d.size());

	// 求协方差矩阵A
	Eigen::Matrix3f A;
	float sum_xx = 0.0f, sum_yy = 0.0f, sum_zz = 0.0f,
		sum_xy = 0.0f, sum_xz = 0.0f, sum_yz = 0.0f;
	for (auto pt : pts3d)
	{
		sum_xx += (pt.x - ave_x) * (pt.x - ave_x);
		sum_yy += (pt.y - ave_y) * (pt.y - ave_y);
		sum_zz += (pt.z - ave_z) * (pt.z - ave_z);

		sum_xy += (pt.x - ave_x) * (pt.y - ave_y);
		sum_xz += (pt.x - ave_x) * (pt.z - ave_z);
		sum_yz += (pt.y - ave_y) * (pt.z - ave_z);
	}
	A(0, 0) = sum_xx / float(pts3d.size());  // 其实, 没必要求均值
	A(0, 1) = sum_xy / float(pts3d.size());
	A(0, 2) = sum_xz / float(pts3d.size());
	A(1, 0) = sum_xy / float(pts3d.size());
	A(1, 1) = sum_yy / float(pts3d.size());
	A(1, 2) = sum_yz / float(pts3d.size());
	A(2, 0) = sum_xz / float(pts3d.size());
	A(2, 1) = sum_yz / float(pts3d.size());
	A(2, 2) = sum_zz / float(pts3d.size());

	// 求协方差矩阵A的特征值和特征向量
	Eigen::EigenSolver<Eigen::Matrix3f> ES(A);
	Eigen::MatrixXcf eigen_vals = ES.eigenvalues();
	Eigen::MatrixXcf eigen_vects = ES.eigenvectors();
	Eigen::MatrixXf eis = eigen_vals.real();  // 特征值
	Eigen::MatrixXf vects = eigen_vects.real();  // 特征向量

	// 求最小特征值对应的特征向量
	Eigen::MatrixXf::Index min_idx, max_idx;
	eis.rowwise().sum().minCoeff(&min_idx);
	eis.rowwise().sum().maxCoeff(&max_idx);

	// ----- 对特征值(特征向量)排序：从小到大
	int mid_idx = 0;
	if (0 == (int)min_idx)
	{
		mid_idx = max_idx == 1 ? 2 : 1;
	}
	else if (1 == (int)min_idx)
	{
		mid_idx = max_idx == 0 ? 2 : 0;
	}
	else
	{
		mid_idx = max_idx == 0 ? 1 : 0;
	}

	// 最小的特征值对应的特征向量(法向量)
	float& a = vects(0, min_idx);
	float& b = vects(1, min_idx);
	float& c = vects(2, min_idx);

	// 确定正确的法向方向: 确保normal指向camera
	if (a * ave_x + b * ave_y + c * ave_z > 0.0f)
	{
		a = -a;
		b = -b;
		c = -c;
	}

	// 最小特征向量(法向量)L2归一化
	const float DENOM = sqrtf(a*a + b * b + c * c);
	a /= DENOM;
	b /= DENOM;
	c /= DENOM;

	// 计算参数d
	//float d = -(a*ave_x + b * ave_y + c * ave_z);

	// 返回特征值: 从小到大排列
	ei_vals[0] = eis(min_idx);
	ei_vals[1] = eis(mid_idx);
	ei_vals[2] = eis(max_idx);

	// 返回特征向量
	ei_vects[0] = a;  // a
	ei_vects[1] = b;  // b
	ei_vects[2] = c;  // c               法向向量

	ei_vects[3] = vects(0, mid_idx);  // 切平面特征向量, 切向分量 1
	ei_vects[4] = vects(1, mid_idx);
	ei_vects[5] = vects(2, mid_idx);

	ei_vects[6] = vects(0, max_idx);  // 切平面特征向量, 切向分量 2
	ei_vects[7] = vects(1, max_idx);
	ei_vects[8] = vects(2, max_idx);

	return 0;
}

// RANSAC与MSAC 参考 https://blog.csdn.net/weixin_44558898/article/details/88986497
// 一般来讲, MSAC要优于RANSAC(对阈值更不敏感)
int RansacRunner::CountInliersRansac(const float* plane_arr,
	const std::vector<cv::Point3f>& Pts3D)
{
	cv::Mat plane_mat(4, 1, CV_32F);
	plane_mat.at<float>(0, 0) = plane_arr[0];
	plane_mat.at<float>(1, 0) = plane_arr[1];
	plane_mat.at<float>(2, 0) = plane_arr[2];
	plane_mat.at<float>(3, 0) = plane_arr[3];

	int count = 0;
	cv::Mat point_mat(4, 1, CV_32F);
	for (auto pt : Pts3D)
	{
		point_mat.at<float>(0, 0) = pt.x;
		point_mat.at<float>(1, 0) = pt.y;
		point_mat.at<float>(2, 0) = pt.z;
		point_mat.at<float>(3, 0) = 1.0f;

		float dist = fabs((float)plane_mat.dot(point_mat)) \
			/ sqrtf(plane_arr[0] * plane_arr[0] + \
				plane_arr[1] * plane_arr[1] + plane_arr[2] * plane_arr[2]);
		if (dist < m_tolerance)
		{
			count++;
		}
	}

	if (count > this->m_num_inliers)
	{
		// update inlier number
		this->m_num_inliers = count;

		// update plane 's 4 parameters
		memcpy(m_plane, plane_arr, sizeof(float) * 4);
	}

	return 0;
}

int RansacRunner::CountInliersMsac(const float* plane_arr,
	const std::vector<cv::Point3f>& Pts3D,
	const float baseline,
	const float focus)
{
	cv::Mat plane_mat(4, 1, CV_32F);
	plane_mat.at<float>(0, 0) = plane_arr[0];
	plane_mat.at<float>(1, 0) = plane_arr[1];
	plane_mat.at<float>(2, 0) = plane_arr[2];
	plane_mat.at<float>(3, 0) = plane_arr[3];

	int count = 0;
	float cost = 0.0f;
	cv::Mat point_mat(4, 1, CV_32F);
	for (auto pt : Pts3D)
	{
		point_mat.at<float>(0, 0) = pt.x;
		point_mat.at<float>(1, 0) = pt.y;
		point_mat.at<float>(2, 0) = pt.z;
		point_mat.at<float>(3, 0) = 1.0f;

		// 计算绝对距离
		float dist = fabs((float)plane_mat.dot(point_mat)) \
			/ sqrtf(plane_arr[0] * plane_arr[0] + \
				plane_arr[1] * plane_arr[1] + plane_arr[2] * plane_arr[2]);

		// 计算相对(尺度)距离...
		if (baseline > 0.0f && focus > 0.0f)
		{
			float DeltaP = pt.z * pt.z / (focus * baseline) * sqrtf(2.0f);  // delta_p
			dist /= DeltaP;

			if (dist < this->m_tolerance / DeltaP)
			{
				cost += dist;
				count += 1;
			}
			else
			{
				cost += this->m_tolerance;
			}
		}
		else
		{
			if (dist < this->m_tolerance)
			{
				cost += dist;
				count += 1;
			}
			else
			{
				cost += this->m_tolerance;
			}
		}
	}

	if (cost < this->m_cost)
	{
		// update cost
		this->m_cost = cost;

		// update inlier number
		this->m_num_inliers = count;

		// update plane's 4 parameters
		memcpy(m_plane, plane_arr, sizeof(float) * 4);
	}

	return 0;
}

// 运行RANSAC迭代
int RansacRunner::RunRansac(const std::vector<cv::Point3f>& Pts3D,
	const float baseline,
	const float focus)
{
	if (this->m_num_sub_sample < 3 || (int)Pts3D.size() < 3)
	{
		printf("[Err]: not enough 3D points\n");
		return -1;
	}

	float plane_arr[4];  // 存放平面方程, 4个参数
	float ei_vals[3];    // 3个特征值
	float ei_vects[9];   // 9个特征向量值

	// 指定最多迭代次数
	while (!this->m_done && this->m_total_iter_count <= this->m_total_iter_num)
	{
		for (int i = 0; i < this->m_num_iter; ++i)
		{
			// 取样本子集
			if (3 == this->m_num_sub_sample)
			{
				if ((int)Pts3D.size() > 3)
				{
					this->GetMinSubSets(Pts3D);
				}
				else if (3 == (int)Pts3D.size())
				{
					memcpy(this->m_min_subsets, Pts3D.data(), sizeof(cv::Point3f) * 3);
					this->m_subsets = Pts3D;
				}
				else
				{
					return -1;
				}
			}
			else
			{
				this->GetSubSets(Pts3D);
			}

			// 计算平面方程存于plane_arr
			if (3 == this->m_num_sub_sample)
			{
				//int ret = this->PlaneFitBy3Pts(plane_arr);
				int ret = this->PlaneFitBy3PtsEi(plane_arr, ei_vals, ei_vects);
				if (-1 == ret)
				{
					continue;
				}
			}
			else
			{
				//this->PlaneFitOLS1(plane_arr);
				//this->PlaneFitOLS2(plane_arr);
				//this->PlaneFitSVD(plane_arr);
				this->PlaneFitPCA(plane_arr, ei_vals, ei_vects);
			}

			// 统计Inliers
			//this->CountInliersRansac(plane_arr, Pts3Dk);
			this->CountInliersMsac(plane_arr, Pts3D, baseline, focus);  // 采用MSAC方法

			// 根据Outlier ratio更新总的迭代次数
			//int pre_num_iter = this->m_num_iter;
			this->m_num_iter = this->UpdateNumIters(0.9999f,   // 99.99%
				(double)((int)Pts3D.size() - this->m_num_inliers) / (double)Pts3D.size(),
				this->m_num_sub_sample,
				this->m_num_iter);
			//printf("Update num_iter from %d to %d\n", pre_num_iter, this->m_num_iter);

			// 更新总的迭代次数
			this->m_total_iter_count += 1;
		}

		// 测试inlier rate
		if ((float)this->m_num_inliers / (float)this->m_num_sample > this->m_inlier_rate)
		{
			this->m_done = true;
			//printf("Inlier rate reached\n");
		}
		else
		{
			// update iter number
			this->m_num_iter = this->m_num_iter_orig;
		}
	}

	// TODO: 保存特征值与特征向量
	if (ei_vals[0] > ei_vals[1] || ei_vals[0] > ei_vals[2] || ei_vals[1] > ei_vals[2])
	{
		printf("[Err]: wrong order of eigen values\n");
		return -1;
	}
	memcpy(this->m_eigen_vals, ei_vals, sizeof(float) * 3);
	memcpy(this->m_eigen_vect, ei_vects, sizeof(float) * 9);

	//printf("Ransac done\n");
	return 0;
}

RansacRunner::~RansacRunner()
{

}

//// ----- verify the 3 points are on the plane
//float& p0 = plane_arr[0];
//float& p1 = plane_arr[1];
//float& p2 = plane_arr[2];
//float& p3 = plane_arr[3];

//float Norm = std::sqrtf(p0 * p0 + p1 * p1 + p2 * p2);
//printf("Norm: %.5f\n", Norm);

//cv::Mat plane_mat(4, 1, CV_32F);
//cv::Mat point_mat(4, 1, CV_32F);
//plane_mat.at<float>(0, 0) = p0;
//plane_mat.at<float>(1, 0) = p1;
//plane_mat.at<float>(2, 0) = p2;
//plane_mat.at<float>(3, 0) = p3;

//for (int i = 0; i < this->m_num_sub_sample; i++)
//{
//	point_mat.at<float>(0, 0) = this->m_subsets[i].x;
//	point_mat.at<float>(1, 0) = this->m_subsets[i].y;
//	point_mat.at<float>(2, 0) = this->m_subsets[i].z;
//	point_mat.at<float>(3, 0) = 1.0f;

//	float res = fabs(plane_mat.dot(point_mat));
//	if (res > 1e-3)
//		printf("Res: %.8f\n", res);
//}