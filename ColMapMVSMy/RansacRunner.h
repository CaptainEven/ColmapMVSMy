#ifndef RANSAC_RUNNER
#define RANSAC_RUNNER

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/base.hpp>

class RansacRunner
{
public:
	// 初始化
	RansacRunner(const float tolerance,
		const int num_sample,
		const int num_sub_sample=-1,
		const int num_iter=60,
		const int total_iter_num = 120,
		const float inlier_rate=0.3f);

	// 取样本子集: 3个样本点
	int GetMinSubSets(const std::vector<cv::Point3f>& Pts3D);

	// 取样本子集: 多于3个样本点
	int GetSubSets(const std::vector<cv::Point3f>& Pts3D);

	// 根据outlier ratio更新总的迭代次数
	int UpdateNumIters(double p, double ep, int modelPoints, int maxIters);

	// 3个空间点(非共线)确定一个空间平面
	int PlaneFitBy3Pts(const cv::Point3f* pts, float* plane_arr);
	int PlaneFitBy3Pts(float* plane_arr);
	int PlaneFitBy3PtsEi(float* plane_arr, float* ei_vals, float* ei_vects);  // 分解特征值和特征向量

	// 3D平面方程拟合(最小二乘法)写成Ax=B的形式: aX + bY + Z + c = 0(aX + bY + c = -Z)
	int PlaneFitOLS1(float* plane_arr);

	// 3D平面方程拟合(最小二乘法)写成Ax=B的形式: aX + bY + c = Z
	int PlaneFitOLS2(float* plane_arr);

	// 3D点云空间平面拟合SVD分解方法
	int PlaneFitSVD(float* plane_arr);

	// 3D点云空间平面拟合PCA分解方法
	int PlaneFitPCA(float* plane_arr, float* ei_vals, float* ei_vects);
	int PlaneFitPCAEi(const std::vector<cv::Point3f>& pts3d,
		float* ei_vals, float* ei_vects);  // 只返回特征值和特征向量

	// 统计内点(inlier)个数: Ransac方法
	int CountInliersRansac(const float* plane_arr, const std::vector<cv::Point3f>& Pts3D);

	// 统计内点(inlier)个数: Msac方法
	int CountInliersMsac(const float* plane_arr,
		const std::vector<cv::Point3f>& Pts3D,
		const float baseline = 0.0f,
		const float focus = 0.0f);

	// 运行RANSAC
	int RunRansac(const std::vector<cv::Point3f>& Pts3D, 
		const float baseline=0.0f,
		const float focus=0.0f);

	~RansacRunner();

//private:
	float m_tolerance;
	int m_num_sample;
	int m_num_sub_sample;

	int m_num_iter;
	int m_num_iter_orig;
	int m_total_iter_count;
	int m_total_iter_num;

	int m_num_inliers;
	float m_inlier_rate;
	float m_cost;

	bool m_done;

	float m_plane[4];  // 平面方程的前三个值即平面法向量

	float m_eigen_vals[3];  // 3个特征值
	float m_eigen_vect[9];  // 9个特征向量值

	cv::Point3f m_min_subsets[3];
	std::vector<cv::Point3f> m_subsets;
};

#endif // !1