#ifndef RANSAC_RUNNER
#define RANSAC_RUNNER

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/base.hpp>

class RansacRunner
{
public:
	// ��ʼ��
	RansacRunner(const float tolerance,
		const int num_sample,
		const int num_sub_sample=-1,
		const int num_iter=60,
		const int total_iter_num = 120,
		const float inlier_rate=0.3f);

	// ȡ�����Ӽ�: 3��������
	int GetMinSubSets(const std::vector<cv::Point3f>& Pts3D);

	// ȡ�����Ӽ�: ����3��������
	int GetSubSets(const std::vector<cv::Point3f>& Pts3D);

	// ����outlier ratio�����ܵĵ�������
	int UpdateNumIters(double p, double ep, int modelPoints, int maxIters);

	// 3���ռ��(�ǹ���)ȷ��һ���ռ�ƽ��
	int PlaneFitBy3Pts(const cv::Point3f* pts, float* plane_arr);
	int PlaneFitBy3Pts(float* plane_arr);
	int PlaneFitBy3PtsEi(float* plane_arr, float* ei_vals, float* ei_vects);  // �ֽ�����ֵ����������

	// 3Dƽ�淽�����(��С���˷�)д��Ax=B����ʽ: aX + bY + Z + c = 0(aX + bY + c = -Z)
	int PlaneFitOLS1(float* plane_arr);

	// 3Dƽ�淽�����(��С���˷�)д��Ax=B����ʽ: aX + bY + c = Z
	int PlaneFitOLS2(float* plane_arr);

	// 3D���ƿռ�ƽ�����SVD�ֽⷽ��
	int PlaneFitSVD(float* plane_arr);

	// 3D���ƿռ�ƽ�����PCA�ֽⷽ��
	int PlaneFitPCA(float* plane_arr, float* ei_vals, float* ei_vects);
	int PlaneFitPCAEi(const std::vector<cv::Point3f>& pts3d,
		float* ei_vals, float* ei_vects);  // ֻ��������ֵ����������

	// ͳ���ڵ�(inlier)����: Ransac����
	int CountInliersRansac(const float* plane_arr, const std::vector<cv::Point3f>& Pts3D);

	// ͳ���ڵ�(inlier)����: Msac����
	int CountInliersMsac(const float* plane_arr,
		const std::vector<cv::Point3f>& Pts3D,
		const float baseline = 0.0f,
		const float focus = 0.0f);

	// ����RANSAC
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

	float m_plane[4];  // ƽ�淽�̵�ǰ����ֵ��ƽ�淨����

	float m_eigen_vals[3];  // 3������ֵ
	float m_eigen_vect[9];  // 9����������ֵ

	cv::Point3f m_min_subsets[3];
	std::vector<cv::Point3f> m_subsets;
};

#endif // !1