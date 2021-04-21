#ifndef MYPATCHMATCH_H
#define MYPATCHMATCH_H

#include <iostream>
#include <memory>
#include <vector>
#include <queue>
#include <string>
#include <random>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <numeric>


#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

//#include "cuda_array_wrapper.h"
#include "gpu_mat.h"
#include "gpu_mat_ref_image.h"
#include "mat.h"
#include "math.h"


#include "depth_map.h"
#include "image.h"
#include "model.h"
#include "normal_map.h"
#include "consistency_graph.h"
#include "workspace.h"
#include "patch_match.h"


namespace colmap
{
namespace mvs
{

	inline float DotProduct3(const float vec1[3], const float vec2[3]) {
		return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
	}
	inline void NormVec3(float vec[3])
	{
		const float norm = sqrt(DotProduct3(vec, vec));
		for (int i = 0; i < 3; i++)
		{
			vec[i] /= norm;
		}
	}
	inline void Mat33DotVec3(const float mat[9], const float vec[3], float result[3]) {
		result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
		result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
		result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
	}
	inline void Mat33DotVec3Homogeneous(const float mat[9],const float vec[2],float result[2])
	{
		const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
		result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
		result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
	}

	inline void ComposeHomography(const Image *refImage, const Image *srcImage, const int row, const int col,
		const float depth, const float normal[3], float H[9])
	{
		// Calibration of source image.
		const float K[4] = { srcImage->GetK()[0], srcImage->GetK()[2], srcImage->GetK()[4], srcImage->GetK()[5] };
		//invCalibration of reference image
		const float ref_inv_K[4] = { 1.0f / refImage->GetK()[0], -refImage->GetK()[2] / refImage->GetK()[0],
			1.0f / refImage->GetK()[4], -refImage->GetK()[5] / refImage->GetK()[4] };

		// Relative rotation and translation between reference and source image.
		float R[9], T[3];
		ComputeRelativePose(refImage->GetR(), refImage->GetT(), srcImage->GetR(), srcImage->GetT(), R, T);

		// Distance to the plane.
		const float dist =
			depth * (normal[0] * (ref_inv_K[0] * col + ref_inv_K[1]) +
			normal[1] * (ref_inv_K[2] * row + ref_inv_K[3]) + normal[2]);
		const float inv_dist = 1.0f / dist;

		const float inv_dist_N0 = inv_dist * normal[0];
		const float inv_dist_N1 = inv_dist * normal[1];
		const float inv_dist_N2 = inv_dist * normal[2];

		// Homography as H = K * (R - T * n' / d) * Kref^-1.
		H[0] = ref_inv_K[0] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
			K[1] * (R[6] + inv_dist_N0 * T[2]));
		H[1] = ref_inv_K[2] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
			K[1] * (R[7] + inv_dist_N1 * T[2]));
		H[2] = K[0] * (R[2] + inv_dist_N2 * T[0]) +
			K[1] * (R[8] + inv_dist_N2 * T[2]) +
			ref_inv_K[1] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
			K[1] * (R[6] + inv_dist_N0 * T[2])) +
			ref_inv_K[3] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
			K[1] * (R[7] + inv_dist_N1 * T[2]));
		H[3] = ref_inv_K[0] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
			K[3] * (R[6] + inv_dist_N0 * T[2]));
		H[4] = ref_inv_K[2] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
			K[3] * (R[7] + inv_dist_N1 * T[2]));
		H[5] = K[2] * (R[5] + inv_dist_N2 * T[1]) +
			K[3] * (R[8] + inv_dist_N2 * T[2]) +
			ref_inv_K[1] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
			K[3] * (R[6] + inv_dist_N0 * T[2])) +
			ref_inv_K[3] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
			K[3] * (R[7] + inv_dist_N1 * T[2]));
		H[6] = ref_inv_K[0] * (R[6] + inv_dist_N0 * T[2]);
		H[7] = ref_inv_K[2] * (R[7] + inv_dist_N1 * T[2]);
		H[8] = R[8] + ref_inv_K[1] * (R[6] + inv_dist_N0 * T[2]) +
			ref_inv_K[3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
	}

//计算似然函数的类
class LikelihoodComputer {
	public:
		LikelihoodComputer(const float ncc_sigma,
			const float min_triangulation_angle,
			const float incident_angle_sigma)
			: cos_min_triangulation_angle_(cos(min_triangulation_angle)),
			inv_incident_angle_sigma_square_(-0.5f / (incident_angle_sigma * incident_angle_sigma)),
			inv_ncc_sigma_square_(-0.5f / (ncc_sigma * ncc_sigma)),
			ncc_norm_factor_(ComputeNCCCostNormFactor(ncc_sigma)) {}

		// Compute forward message from current cost and forward message of
		// previous / neighboring pixel.
		float ComputeForwardMessage(const float cost,
			const float prev) const {
			return ComputeMessage<true>(cost, prev);
		}

		// Compute backward message from current cost and backward message of
		// previous / neighboring pixel.
		float ComputeBackwardMessage(const float cost,
			const float prev) const {
			return ComputeMessage<false>(cost, prev);
		}

		// Compute the selection probability from the forward and backward message.
		inline float ComputeSelProb(const float alpha, const float beta,
			const float prev,
			const float prev_weight) const {
			const float zn0 = (1.0f - alpha) * (1.0f - beta);
			const float zn1 = alpha * beta;
			const float curr = zn1 / (zn0 + zn1);
			return prev_weight * prev + (1.0f - prev_weight) * curr;
		}

		// Compute NCC probability. Note that cost = 1 - NCC.
		inline float ComputeNCCProb(const float cost) const {
			return exp(cost * cost * inv_ncc_sigma_square_) * ncc_norm_factor_;
		}

		// Compute the triangulation angle probability.
		inline float ComputeTriProb(const float cos_triangulation_angle) const 
		{
			const float abs_cos_triangulation_angle = abs(cos_triangulation_angle);
			if (abs_cos_triangulation_angle > cos_min_triangulation_angle_) 
			{
				const float scaled = 1.0f -
					(1.0f - abs_cos_triangulation_angle) /
					(1.0f - cos_min_triangulation_angle_);
				const float likelihood = 1.0f - scaled * scaled;
				return min(1.0f, max(0.0f, likelihood));
			}
			else 
			{
				return 1.0f;
			}
		}

		// Compute the incident angle probability.
		inline float ComputeIncProb(const float cos_incident_angle) const
		{
			const float x = 1.0f - max(0.0f, cos_incident_angle);
			return exp(x * x * inv_incident_angle_sigma_square_);
		}

		// Compute the warping/resolution prior probability.
		inline float ComputeResolutionProb(const float H[9],const float row,const float col ,const int kWindowSize) const 
		{
			const int kWindowRadius = kWindowSize / 2;

			// Warp corners of patch in reference image to source image.
			float src1[2];
			const float ref1[2] = { row - kWindowRadius, col - kWindowRadius };
			Mat33DotVec3Homogeneous(H, ref1, src1);
			float src2[2];
			const float ref2[2] = { row - kWindowRadius, col + kWindowRadius };
			Mat33DotVec3Homogeneous(H, ref2, src2);
			float src3[2];
			const float ref3[2] = { row + kWindowRadius, col + kWindowRadius };
			Mat33DotVec3Homogeneous(H, ref3, src3);
			float src4[2];
			const float ref4[2] = { row + kWindowRadius, col - kWindowRadius };
			Mat33DotVec3Homogeneous(H, ref4, src4);

			// Compute area of patches in reference and source image.
			const float ref_area = float(kWindowSize * kWindowSize);
			const float src_area =
				abs(0.5f * (src1[0] * src2[1] - src2[0] * src1[1] - src1[0] * src4[1] +
				src2[0] * src3[1] - src3[0] * src2[1] + src4[0] * src1[1] +
				src3[0] * src4[1] - src4[0] * src3[1]));

			if (ref_area > src_area) 
			{
				return src_area / ref_area;
			}
			else 
			{
				return ref_area / src_area;
			}
		}

	private:
		// The normalization for the likelihood function, i.e. the normalization for
		// the prior on the matching cost.
		static inline float ComputeNCCCostNormFactor(
			const float ncc_sigma) {
			// A = sqrt(2pi)*sigma/2*erf(sqrt(2)/sigma)
			// erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t^2) dt
			return float(2.0f / (sqrt(2.0f * M_PI) * ncc_sigma *
				erff(2.0f / (ncc_sigma * 1.414213562f))));
		}

		// Compute the forward or backward message.
		template <bool kForward>
		inline float ComputeMessage(const float cost,
			const float prev) const {
			const float kUniformProb = 0.5f;
			const float kNoChangeProb = 0.99999f;
			const float kChangeProb = 1.0f - kNoChangeProb;
			const float emission = ComputeNCCProb(cost);

			float zn0;  // Message for selection probability = 0.
			float zn1;  // Message for selection probability = 1.
			if (kForward) {
				zn0 = (prev * kChangeProb + (1.0f - prev) * kNoChangeProb) * kUniformProb;
				zn1 = (prev * kNoChangeProb + (1.0f - prev) * kChangeProb) * emission;
			}
			else {
				zn0 = prev * emission * kChangeProb +
					(1.0f - prev) * kUniformProb * kNoChangeProb;
				zn1 = prev * emission * kNoChangeProb +
					(1.0f - prev) * kUniformProb * kChangeProb;
			}

			return zn1 / (zn0 + zn1);
		}

		float cos_min_triangulation_angle_;
		float inv_incident_angle_sigma_square_;
		float inv_ncc_sigma_square_;
		float ncc_norm_factor_;
	};



class MyPatchMatch
{
public:
	struct Options
	{
		//稀疏点云最大的pach损失值
		float sparse_max_patch_cost = 1.5f;

		//稀疏点云最大的迭代次数
		int sparse_max_patch_iterator = 3;

		//传播过程中，最大的patch损失值
		float radiant_max_patch_cost = 0.2f;

		//传播过程中，最大迭代次数
		int radiant_max_patch_iterator = 4;

		//传播过程中，较小patch个数阈值
		int samll_cost_num = 3;

		//蒙特卡洛采样个数
		int num_smaples = 10;

		//每个像素点被传播的次数
		int num_pixel_propagation = 1;
	};

	MyPatchMatch(const MyPatchMatch::Options &options, const PatchMatch::Options &pmOptions,
	           std::unique_ptr<Workspace> workspace)
		:options_(options), pmOptions_(pmOptions)
	{
		workspace_ = std::move(workspace);
	}
	virtual ~MyPatchMatch(){}

	void Run();
	std::unique_ptr<Workspace> ReturnValue();
private:
	void getProblems();
	void computeAllSelfPatch();
	void initSparsePoints();  // 初始化稀疏点云，计算每个点的法向量
	void radiantPropagation();  // 以稀疏点云为中心，向外辐射状传播

	void seedExtend(const int row, const int col, const PatchMatch::Problem &problem);


	inline float GenerateRandomDepth(const float min_depth,const float max_depth) const;
	inline void GenerateRandomNormal(const int row, const int col, const float *refK, float normal[3]) const;
	inline void PerturbNormal(const int row, const int col, const float *refK, const float perturbation, const float normal[3], float perturbed_normal[3]) const;
	inline void PerturbDepth(const float srcDepth, const float perturbation, float *pertDepth)const;
	inline int  FindMinCost(const float *costs, const int kNumCosts)const;
	inline int FindGivenValue(const std::vector<int> &value, const int num, const int given)const;
	inline float PropagateDepth(const float *refK, const float depth1,const float normal1[3], 
		const float row1, const float col1,const float row2, const float col2) const;
	inline void ComputeViewingAngles(const float point[3], const float normal[3], const Image &refImage, const Image &srcImage,
		float* cos_triangulation_angle, float* cos_incident_angle) const;
	inline void TransformPDFToCDF(std::vector<float> &probs, const int num_probs) const;

	inline void SortWithIds(std::vector<float> &value, std::vector<int> &ids)const;

	Options options_;
	const PatchMatch::Options pmOptions_;
	std::vector<PatchMatch::Problem> problems_;

	std::unique_ptr<Workspace> workspace_;
	std::unique_ptr<GpuMatRefImage> ref_image_;//用于计算各自图像的patch
	std::vector<Mat<float>> image_sums_;
	std::vector<Mat<float>> image_squared_sums_;
	std::vector<cv::Point3f> sparse_normals_;

	std::vector<DepthMap> depthMaps_;
	std::vector<NormalMap> normalMaps_;

	std::vector<Mat<float>> selProbMaps_;
	std::vector<Mat<float>> prevSelProMaps_;
	std::vector<Mat<float>> costMaps_;
	Mat<int> mask_;

	std::queue<int> Rows_;
	std::queue<int> Cols_;

};


}//namespace mvs
}//namespace colmap
#endif//MYPATCHMATCH_H