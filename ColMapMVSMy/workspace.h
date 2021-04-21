#ifndef COLMAP_SRC_MVS_WORKSPACE_H_
#define COLMAP_SRC_MVS_WORKSPACE_H_

#include <algorithm>
#include <numeric>

#include <thread>
#include <mutex>

#include "consistency_graph.h"
#include "depth_map.h"
#include "model.h"
#include "normal_map.h"

#include "SLICSuperpixels.h"
#include "plane_detection.h"


namespace colmap {
	namespace mvs {

		class Workspace {
		public:
			struct Options {
				// The maximum cache size in gigabytes.
				double cache_size = 32.0;

				// Maximum image size in either dimension.
				int max_image_size = 400;  // -1

				// Whether to read image as RGB or gray scale.
				bool image_as_rgb = true;

				////�Ƿ�����ϸ����ǿ��ͼ��
				bool bDetailEnhance = true;

				////�Ƿ����ýṹ��ǿ��ͼ��
				bool bStructureEnhance = false;

				////�Ƿ���н���������(��ͼ��߶Ⱥ�������������޸ģ�ϡ����άģ�Ͳ���)
				bool bDown_sampling = false;

				//�������ĳ߶�
				float fDown_scale = 4.0f;

				bool bOurs = false;  // ����˫�ߴ����ϲ���
				bool bOursFast = false;  // ��������˫�ߴ����ϲ�������

				bool bBilinearInterpolation = 0;  // ˫���Բ�ֵ�ϲ���

				bool bFastBilateralSolver = 0;  // ����˫�������

				bool bJointBilateralUpsampling = 0;  // ����˫���ϲ���

				bool bBilateralGuidedUpsampling = 0;  // ˫�������ϲ���

				// Location and type of workspace.
				std::string workspace_path;
				std::string workspace_format;
				std::string input_type;
				std::string input_type_geom;
				std::string newPath;
				std::string src_img_dir;

				//ȥŤ����Ŀ¼
				std::string undistorte_path;

				// �����طָ�Ŀ¼
				std::string slic_path;
			};

			// ���캯��: ��bundler�����ж�ȡϡ�������Ϣ
			Workspace(const Options& options);

			const Model& GetModel() const;
			const cv::Mat& GetBitmap(const int image_id);
			const DepthMap& GetDepthMap(const int image_id) const;
			const NormalMap& GetNormalMap(const int image_id) const;
			const ConsistencyGraph& GetConsistencyGraph(const int image_id) const;

			const void ReadDepthAndNormalMaps(const bool isGeometric);

			const std::vector<DepthMap>& GetAllDepthMaps() const;
			const std::vector<NormalMap>& GetAllNormalMaps() const;

			//��������Ľ����д��workspace��Ӧ�ı�������
			void WriteDepthMap(const int image_id, const DepthMap &depthmap);
			void WriteNormalMap(const int image_id, const NormalMap &normalmap);
			void WriteConsistencyGraph(const int image_id, const ConsistencyGraph &consistencyGraph);

			//ִ�г����طָ�
			void runSLIC(const std::string& path);

			//����ά��ͶӰ��ͼ����
			void showImgPointToSlicImage(const std::string &path);

			//�����ͼ�ͷ���ͼ�����ϲ�����ͬʱ�޸�model�е�ͼ����Ϣ
			void UpSampleMapAndModel();

			// Get paths to bitmap, depth map, normal map and consistency graph.
			std::string GetBitmapPath(const int image_id) const;
			std::string GetDepthMapPath(const int image_id, const bool isGeom) const;
			std::string GetNormalMapPath(const int image_id, const bool isGeom) const;
			std::string GetConsistencyGaphPath(const int image_id) const;

			// Return whether bitmap, depth map, normal map, and consistency graph exist.
			bool HasBitmap(const int image_id) const;
			bool HasDepthMap(const int image_id, const bool isGeom) const;
			bool HasNormalMap(const int image_id, const bool isGeom) const;

			float GetDepthRange(const int image_id, bool isMax) const;

			void jointBilateralUpsampling(const cv::Mat &joint, const cv::Mat &lowin, const float upscale,
				const double sigma_color, const double sigma_space, int radius, cv::Mat &highout) const;

			void jointBilateralPropagationUpsampling(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
				const float upscale, const double sigma_color, const double sigma_space, const int radius, cv::Mat &highDepthMat) const;

			void jointBilateralDepthMapFilter1(const cv::Mat &srcDepthMap, const cv::Mat &srcNormalMap, const cv::Mat &srcImage, const float *refK,
				const int radius, const double sigma_color, const double sigma_space, DepthMap &desDepMap, NormalMap &desNorMap, const bool DoNormal)const;

			float PropagateDepth(const float *refK, const float depth1, const float normal1[3],
				const float row1, const float col1, const float row2, const float col2) const;

			void SuitNormal(const int row, const int col, const float *refK, float normal[3]) const;

			//�Է�����ͼ��������ֵ�˲�
			void NormalMapMediaFilter(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat, const int windowRadis)const;
			void NormalMapMediaFilter1(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat, const int windowRadis)const;
			void NormalMapMediaFilterWithDepth(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat,
				const cv::Mat &InDepthMapMat, cv::Mat &OutDepthMapMat, int windowRadis)const;

			std::vector<cv::Point3f> sparse_normals_;

			std::string GetFileName(const int image_id, const bool isGeom) const;

			void newPropagation(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
				const float upscale, const double sigma_color, const double sigma_space, int radius, const int maxSrcPoint,
				cv::Mat &highDepthMat, cv::Mat &highNormalMat) const;

			void newPropagationFast(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
				const double sigma_color, const double sigma_space, int radius, const int maxSrcPoint,
				cv::Mat &outDepthMat, cv::Mat &outNormalMat) const;

			// @even:ΪCV_32F���ݽ��аߵ��˲�����
			template<typename T>
			void FilterSpeckles(cv::Mat & img, T newVal, int maxSpeckleSize, T maxDiff);

			// @even: ����ƽ��Լ�������ֵ: ��������ϵ
			float GetDepthCoPlane(const float* K_inv_arr,
				const float* R_inv_arr,
				const float* T_arr,
				const float* plane_arr,
				const cv::Point2f& pt2D);

			// @even: ����ƽ��Լ�������ֵ: �������ϵ
			float GetDepthCoPlaneCam(const float* K_inv_arr,
				const float* plane_arr,
				const cv::Point2f& pt2D);

			inline bool Is2RectIntersect(const float* x_range_1,
				const float* y_range_1,
				const float* x_range_2,
				const float* y_range_2)
			{
				if (y_range_1[1] < y_range_2[0] || y_range_2[1] < y_range_1[0]
					|| x_range_1[1] < x_range_2[0] || x_range_2[1] < x_range_1[0])
				{
					return false;
				}

				return true;
			}

			// @even: ������ƽ������: ��������ϵ
			int CorrectPCPlane(const float* P_arr,
				const float* K_inv_arr,
				const float* R_inv_arr,
				const float* T_arr,
				const int IMG_WIDTH,
				const int IMG_HEIGHT,
				const float DIST_THRESH,
				const float fold,
				const std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				std::unordered_map<int, std::vector<float>>& plane_map,
				std::unordered_map<int, cv::Point3f>& center_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, std::vector<float>>& eigen_vals_map,
				std::unordered_map<int, std::vector<float>>& eigen_vects_map);

			// @even: ������ƽ������: �������ϵ
			int CorrectPlaneCam(const float* K_arr,
				const float* K_inv_arr,
				const int IMG_WIDTH,
				const int IMG_HEIGHT,
				const float DIST_THRESH,
				const float fold, const int TH_Num_Neigh,
				const std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				std::unordered_map<int, std::vector<float>>& plane_map,
				std::unordered_map<int, cv::Point3f>& center_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, std::vector<float>>& eigen_vals_map,
				std::unordered_map<int, std::vector<float>>& eigen_vects_map);

			// ���ͼ��Ӧ��3D����
			struct OrganizedImage3D
			{
				const cv::Mat_<cv::Vec3f>& cloud;

				//note: ahc::PlaneFitter assumes mm as unit!!!
				OrganizedImage3D(const cv::Mat_<cv::Vec3f>& c) : cloud(c) {}

				inline int width() const { return cloud.cols; }

				inline int height() const { return cloud.rows; }

				inline bool get(const int row, const int col,
					double& x, double& y, double& z) const
				{
					const cv::Vec3f& p = cloud.at<cv::Vec3f>(row, col);
					x = p[0];
					y = p[1];
					z = p[2];
					return z > 0 && isnan(z) == 0;  // return false if current depth is NaN
				}
			};

			// @even: ����3D�����������(R, T)��ȡ���ֵ: ��������ϵ
			inline float GetDepthBy3dPtRT(const float* R_arr,
				const float* T_arr,
				const cv::Point3f& pt3d)
			{
				// 3D�ռ����������
				const Eigen::Vector3f X(pt3d.x, pt3d.y, pt3d.z);
				return Eigen::Map<const Eigen::Vector3f>(&R_arr[6]).dot(X) + T_arr[2];
			}

			// @even: ����3D����(�������ϵ)������ڲ�K��ȡ���ֵ: �������ϵ
			inline float GetDepthBy3dPtK(const float* K_arr,
				const cv::Point3f& pt3d)
			{
				// 3D�ռ������:�������ϵ
				const Eigen::Vector3f X(pt3d.x, pt3d.y, pt3d.z);
				Eigen::Vector3f pt2d_h = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * X;
				return pt2d_h[2];
			}

			// @even: ������Ʒ���
			inline int GetPointCloudVariance(const std::vector<cv::Point3f>& pts3D,
				float* variance)
			{
				// �ֱ����x,y,z�ķ���
				float mean_x = 0.0f, mean_y = 0.0f, mean_z = 0.0f;
				for (auto pt3d : pts3D)
				{
					mean_x += pt3d.x;
					mean_y += pt3d.y;
					mean_z += pt3d.z;
				}
				mean_x /= float(pts3D.size());
				mean_y /= float(pts3D.size());
				mean_z /= float(pts3D.size());

				// ���㷽��
				float var_x = 0.0f, var_y = 0.0f, var_z = 0.0f;
				for (auto pt3d : pts3D)
				{
					var_x += (pt3d.x - mean_x) * (pt3d.x - mean_x);
					var_y += (pt3d.y - mean_y) * (pt3d.y - mean_y);
					var_z += (pt3d.z - mean_z) * (pt3d.z - mean_z);
				}
				var_x /= float(pts3D.size());
				var_y /= float(pts3D.size());
				var_z /= float(pts3D.size());

				variance[0] = var_x;
				variance[1] = var_y;
				variance[2] = var_z;

				return 0;
			}

			// @even: ������ƽ��Ե��ƽ�������ƽ��: ��������ϵ
			int SmoothPointCloud(const float* R_arr,
				const float* T_arr,
				const float* K_inv_arr,
				const float* R_inv_arr,
				DepthMap& depth_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, std::vector<float>>& ei_vects_map);

			// @even: ������ƽ��Ե��ƽ�������ƽ��: ��������ϵ
			int SmoothPointCloudCam(const float* K_arr,
				const float* K_inv_arr,
				const std::vector<cv::Point2f>& pts2D,
				const float* normal,
				const float* tangent_1,
				const float* tangent_2,
				DepthMap& depth_map);

			// @even: �жϸ���ƽ���ཻ��3D���Ƿ�����ƽ�淶Χ��
			bool IsPt3DInPlaneRange(const cv::Point3f& center, const cv::Point3f& pt3D,
				const float ei_val_tagent_1, const float ei_val_tagent_2,
				const float* ei_vect_tagent_1, const float* ei_vect_tagent_2,  // ��ƽ�������������2���������
				const float fold)
			{
				// ��������
				cv::Point3f vector = pt3D - center;  // vector: center -> pt3D

				// �жϵ�һ������������
				float component_1 = fabs((vector.x * ei_vect_tagent_1[0]
										+ vector.y * ei_vect_tagent_1[1]
										+ vector.z * ei_vect_tagent_1[2]) / \
					sqrtf(ei_vect_tagent_1[0] * ei_vect_tagent_1[0]
						+ ei_vect_tagent_1[1] * ei_vect_tagent_1[1]
						+ ei_vect_tagent_1[2] * ei_vect_tagent_1[2]));
				if (component_1 > fold * sqrtf(ei_val_tagent_1))
				{
					return false;
				}

				// �жϵڶ�������������
				float component_2 = fabs((vector.x * ei_vect_tagent_2[0]
										+ vector.y * ei_vect_tagent_2[1]
										+ vector.z * ei_vect_tagent_2[2]) / \
					sqrtf(ei_vect_tagent_2[0] * ei_vect_tagent_2[0]
						+ ei_vect_tagent_2[1] * ei_vect_tagent_2[1]
						+ ei_vect_tagent_2[2] * ei_vect_tagent_2[2]));
				if (component_2 > fold * sqrtf(ei_val_tagent_2))
				{
					return false;
				}

				return true;
			}

			// @even: ��ƽ�淶Χ�ڵ��ĸ�������3D(����)����ϵ�е�����
			// Ȼ��3DͶӰ��2D, ����2D������Χ
			int GetSearchRange(const cv::Point3f& center, 
				const float* P, 
				const float* ei_vals,
				const float* ei_vects,
				const float fold,
				float* x_range, float* y_range);

			// @even: ��ƽ�淶Χ�ڵ��ĸ�������3D(���)����ϵ�е�����
			// Ȼ��3DͶӰ��2D, ����2D������Χ
			int GetSearchRangeCam(const cv::Point3f& center,
				const float* K_arr,
				const float* ei_vals,
				const float* ei_vects,
				const float fold,
				float* x_range, float* y_range);

			// @even: ����image(view)��Ա�
			inline std::unordered_map<int, std::set<int>> GetImgPairMap()
			{
				std::unordered_map<int, std::set<int>> pair_map;

				for (auto pt3D : this->m_model.m_points)
				{
					// ����ÿ�����track
					for (int i = 0; i < (int)pt3D.track.size(); ++i)
					{
						for (int j = i + 1; j < (int)pt3D.track.size(); ++j)
						{
							pair_map[pt3D.track[i]].insert(pt3D.track[j]);
							pair_map[pt3D.track[j]].insert(pt3D.track[i]);
						}
					}
				}

				return pair_map;
			}

			// @even: ������Ա�, ����image(view)��ƽ�����߾���
			inline float GetMeanBaseline(const int img_id, 
				std::unordered_map<int, std::set<int>>& pair_map)
			{
				float b_sum = 0.0f;
				for (int id : pair_map[img_id])
				{
					const float* center1 = this->m_model.m_images[img_id].GetCenter();
					const float* center2 = this->m_model.m_images[id].GetCenter();

					b_sum += sqrtf((center1[0] - center2[0]) * (center1[0] - center2[0]) +
						(center1[1] - center2[1]) * (center1[1] - center2[1]) +
						(center1[2] - center2[2]) * (center1[2] - center2[2]));
				}

				return b_sum / float(pair_map[img_id].size());
			}

			// @even: ��ȡConsistency graph
			int ReadConsistencyGraph(const std::string& view_name);

			// @even: ����superpixel��Χ(2D����)��̬ȷ��sigma
			inline float GetSigmaOfPts2D(const std::vector<cv::Point2f>& pts2d)
			{
				/*
				1. ȡx, y��Χ�Ľ�Сֵ(or��ֵ)
				2. ��˹�ֲ�, ��6*sigma == ��С��ȡֵ��Χ(+-3sigma)
				*/

				std::vector<float> x_arr(pts2d.size());
				std::vector<float> y_arr(pts2d.size());

				for (int i = 0; i < (int)pts2d.size(); ++i)
				{
					x_arr[i] = pts2d[i].x;
					y_arr[i] = pts2d[i].y;
				}

				auto x_minmax = std::minmax_element(x_arr.begin(), x_arr.end());
				auto y_minmax = std::minmax_element(y_arr.begin(), y_arr.end());
				float x_range = *x_minmax.second - *x_minmax.first;
				float y_range = *y_minmax.second - *y_minmax.first;

				//return (x_range + y_range) / 12.0f;
				//return x_range <= y_range ? x_range/6.0f : y_range/6.0f;  // ȡ��С��range
				return x_range >= y_range ? x_range / 6.0f : y_range / 6.0f;  // ȡ�ϴ��range
			}

			inline float GetSigmaOfPts3D(const std::vector<cv::Point3f>& pts3D)
			{
				/*
				1. ȡx, y, z�ֲ���Χ�ľ�ֵ
				2. ��˹�ֲ�, ��6*sigma == ��С��ȡֵ��Χ(+-3sigma)
				*/
				std::vector<float> x_arr(pts3D.size());
				std::vector<float> y_arr(pts3D.size());
				std::vector<float> z_arr(pts3D.size());

				for (int i = 0; i < (int)pts3D.size(); ++i)
				{
					x_arr[i] = pts3D[i].x;
					y_arr[i] = pts3D[i].y;
					z_arr[i] = pts3D[i].z;
				}

				auto x_minmax = std::minmax_element(x_arr.begin(), x_arr.end());
				auto y_minmax = std::minmax_element(y_arr.begin(), y_arr.end());
				auto z_minmax = std::minmax_element(z_arr.begin(), z_arr.end());
				float x_range = *x_minmax.second - *x_minmax.first;
				float y_range = *y_minmax.second - *y_minmax.first;
				float z_range = *z_minmax.second - *z_minmax.first;

				return (x_range + y_range + z_range) / 18.0f;
			}

			// @even: �������ͼ(���ڳ����طָ�): ��������ϵ
			void TestDepth1();

			// @even: �������ͼ(���ڳ����طָ�): �������ϵ
			void TestDepth5();

			// ----- @even: ����Speckle����, 3Dƽ����Ϻ�MRF�Ż�
			struct HashFuncPt2f  // �Զ����ϣ����
			{
				std::size_t operator()(const cv::Point2f& pt2d) const
				{
					using std::size_t;
					using std::hash;

					// ʹ����Ϊ+�������õ���ϣֵ(������ײ)
					return hash<float>()(pt2d.x) ^ (hash<float>()(pt2d.y) << 1);
				}
			};
			struct EuqualPt2f  // �Զ�������ж�����
			{
				bool operator () (const cv::Point2f& pt2f_1, const cv::Point2f& pt2f_2) const
				{
					return pt2f_1.x == pt2f_2.x && pt2f_1.y == pt2f_2.y;
				}
			};

			// @even: TODO ����MRF����
			float GetEnergyOfPt2d(const cv::Point2f& Pt2D,
				const int SP_Label,
				std::unordered_map<cv::Point2f, int, HashFuncPt2f, EuqualPt2f>& pt2d2SPLabel,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map,
				std::unordered_map<int, std::vector<float>>& plane_map,
				const DepthMap& depth_map,
				const float* K_inv_arr,
				const float beta,
				const int radius);

			// @even: ����MRF�������̺߳���(ÿ���̼߳���һ����2D��)
			void ThFuncPts2dEnergy(const std::vector<cv::Point2f>& Pts2D,
				const int Start, const int End, const int plane_count,
				const std::unordered_map<int, int>& plane_id2SPLabel,
				const std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				const cv::Mat& labels,
				const DepthMap& depth_map,
				const float beta,
				const int radius,
				std::vector<int>& Pt2DSPLabelsRet);

			// ����Speckle����, 3Dƽ����Ϻ�MRF�Ż�
			void TestDepth6();

			// ����ָ����ͼ,�ֿ����3d�㼯,ƽ�����,���ɺ�ѡlabel_array��labels
			int SplitDepthMat(const cv::Mat& depth_mat,
				const int blk_size,  // block size
				const float* K_inv_arr,  // ����ڲξ������
				std::vector<int>& blks_pt_cnt,  // ��¼����block��pt2d����
				std::vector<cv::Point2f>& blks_pts2d,  // ��¼����block��pt2d������
				std::vector<int>& blks_pt_cnt_has,  // ��¼������block�������ֵ�����
				std::vector<int>& blks_pt_cnt_non,  // ��¼������block�������ֵ�����
				std::vector<std::vector<float>>& plane_equa_arr,  // ��¼��Ϊlabel��blk_id��Ӧ��ƽ�淽��
				std::vector<int>& label_blk_ids,  // ��¼���㹻�����ֵ���blk_id: �ɵ���label
				std::vector<int>& proc_blk_ids,  // ��¼��(MRF)�����blk_id
				std::vector<float>& proc_blks_depths_has,  // ��¼������block�������ֵ(��ɵ�����)
				std::vector<int>& proc_blks_pts2d_has_num,  // ��¼������block�������ֵ�����
				std::vector<int>& proc_blks_pts2d_non_num,  // ��¼������block����ȵ����
				std::vector<cv::Point2f>& proc_blks_pt2d_non,  // ��¼������block�������ֵ������
				std::vector<int>& all_blks_labels,  // ��¼ÿ��block��Ӧ��label(blk_id): ��ʼlabel����
				int& num_y, int& num_x);

			// ����Speckle����, 3Dƽ����Ϻ�MRF�Ż�: �ֿ�˼������ƽ������(΢��)
			void TestDepth7();

			// @even: �������ͼ(���ڿ���ƽ������ȡ)opencv����
			//void TestDepth2();

			// @even: 
			void runMRFOptimization();

			// @even: ����ƽ����ȡ, ��������ϵ(K[R|t])
			void TestDepth3();

			// ���*1000��uint16���ͼ
			void OutputU16Depthmap();

			// @even: ����ƽ����ȡ, �������ϵ(K)
			void TestDepth4();

			// @even: ĳ��superpixel����PCA���: ��������ϵ
			int FitPlaneForSuperpixel(const DepthMap& depth_map,
				const float* K_inv_arr,
				const float* R_inv_arr,
				const float* T_arr,
				const std::vector<cv::Point2f>& Pts2D,
				std::vector<float>& plane_normal,
				std::vector<float>& eigen_vals,
				std::vector<float>& center_arr);

			// @even: ĳ��superpixel����PCA���: �������ϵ
			int FitPlaneForSuperpixelCam(const DepthMap& depth_map,
				const float* K_inv_arr,
				const std::vector<cv::Point2f>& Pts2D,
				std::vector<float>& plane_arr,
				std::vector<float>& plane_normal,
				std::vector<float>& eigen_vals,
				std::vector<float>& center_arr);

			// @even: ����FitPlaneForSuper 1
			int FitPlaneForSuperpixel(const std::vector<cv::Point3f>& Pts3D,
				float* plane_normal,
				float* eigen_vals,
				float* eigen_vects,
				float* center_arr);

			// @even: ����FitPlaneForSuper 2
			int FitPlaneForSuperpixel(const std::vector<cv::Point3f>& Pts3D,
				std::vector<float>& plane_arr,
				std::vector<float>& eigen_vals,
				std::vector<float>& eigen_vects,
				cv::Point3f& center_pt);

			// @even: ÿ��superpixel����PCA���ƽ��: ��������ϵ
			int FitPlaneForSuperpixels(const DepthMap& depth_map,
				const float* K_inv_arr,
				const float* R_inv_arr,
				const float* T_arr,
				const std::unordered_map<int, std::vector<cv::Point2f>>& labels_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
				std::unordered_map<int, cv::Point3f>& center_map,
				std::unordered_map<int, std::vector<float>>& eigen_vals_map,
				std::unordered_map<int, std::vector<float>>& eigen_vects_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, std::vector<float>>& plane_map);

			// @even: ÿ��superpixel����PCA���ƽ��: �������ϵ
			int FitPlaneForSPsCam(const DepthMap& depth_map,
				const float* K_inv_arr,
				const std::unordered_map<int, std::vector<cv::Point2f>>& labels_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
				std::unordered_map<int, cv::Point3f>& center_map,
				std::unordered_map<int, std::vector<float>>& eigen_vals_map,
				std::unordered_map<int, std::vector<float>>& eigen_vects_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, std::vector<float>>& plane_map);

			// �ж�һ��superpixel��Ӧ�ĵ����Ƿ���ƽ�� 
			inline bool IsPlaneSuperpixelCloud(const std::vector<cv::Point3f>& pts3D,
				const float* plane_arr,
				const std::vector<float>& eigen_vals,
				const float Ei_Val_TH,
				const float Dist_TH)
			{
				int plane_dim = GetPlaneDimForSuperpixel(eigen_vals, Ei_Val_TH);
				if (plane_dim > 2)
				{
					return false;
				}

				// 1. ȡ(10%~90%)��ƽ���ƽ������
				// 2. ��ƽ��ƽ������ķ���
				std::vector<float> dists(pts3D.size(), 0.0f);
				for (int i = 0; i < (int)pts3D.size(); ++i)
				{
					dists[i] = fabsf(plane_arr[0] * pts3D[i].x
						+ plane_arr[1] * pts3D[i].y
						+ plane_arr[2] * pts3D[i].z + plane_arr[3]);
				}
				std::sort(dists.begin(), dists.end());
				int ten_pct = int(dists.size() * 0.1f);
				int ninety_pct = int(dists.size() * 0.9f);

				std::vector<float> new_dists(ninety_pct - ten_pct + 1, 0.0f);
				memcpy(new_dists.data(), dists.data() + ten_pct, sizeof(float) * new_dists.size());

				float mean = std::accumulate(new_dists.begin(), new_dists.end(), 0.0f)
					/ float(new_dists.size());
				if (mean > Dist_TH)
				{
					return false;
				}

				//for (auto pt3d : pts3D)
				//{
				//	float dist = fabsf(plane_arr[0] * pt3d.x
				//		+ plane_arr[1] * pt3d.y
				//		+ plane_arr[2] * pt3d.z);
				//	std += (dist - mean) * (dist - mean);
				//}
				//std /= float(pts3D.size());
				//std = sqrtf(std);
				//if (std > Dist_TH)
				//{
				//	return false;
				//}   // ����ǲ�����

				return true;
			}

			// ��һ��superpixel�ڲ�����JBU���ֵ
			inline float JBUSP(const cv::Point2f pt_1,
				const std::vector<cv::Point2f>& pts2d_has_depth,
				const cv::Mat& src,
				const DepthMap& depth_map,
				const float& sigma_s)
			{
				const cv::Vec3b& color_0 = src.at<cv::Vec3b>((int)pt_1.y, (int)pt_1.x);

				float depth = 0.0f, sum_depth = 0.0f, sum_weight = 0.0f;
				for (cv::Point2f pt_2 : pts2d_has_depth)
				{
					float delta_dist = sqrtf((pt_1.x - pt_2.x) * (pt_1.x - pt_2.x)
						+ (pt_1.y - pt_2.y) * (pt_1.y - pt_2.y));
					float space_weight = expf(-0.5f*delta_dist*delta_dist / (sigma_s*sigma_s));

					const cv::Vec3b& color_1 = src.at<cv::Vec3b>((int)pt_2.y, (int)pt_2.x);
					float delta_color = fabsf(color_0[0] - color_1[0])
						+ fabsf(color_0[1] - color_1[1])
						+ fabsf(color_0[2] - color_1[2]);  // L1 norm of color difference
					float color_weight = expf(-0.5f*delta_color*delta_color / 16384.0f);  //128*128
					float weight = space_weight * color_weight;

					sum_depth += weight * depth_map.GetDepth((int)pt_2.y, (int)pt_2.x);
					sum_weight += weight;
				}

				// to prevent overflow
				if (sum_weight > 1e-10)
				{
					depth = sum_depth / sum_weight;
				}
				//else
				//{
				//	printf("[Warning]: empty depth @[%d, %d]\n",
				//		(int)pt_1.x, (int)pt_1.y);  // need to debug here...
				//}

				return depth;
			}

			// @even: ����ÿ��superpixel��Ӧ��plane������ֵȷ��ƽ��ά��
			inline int GetPlaneDimForSuperpixel(const std::vector<float>& eigen_vals,
				const float THRESHOLD)
			{
				auto W_n = std::min_element(std::begin(eigen_vals), std::end(eigen_vals));
				auto W_1 = std::max_element(std::begin(eigen_vals), std::end(eigen_vals));
				float W_2 = 0.0f;
				for (float eigen_val : eigen_vals)
				{
					if (eigen_val != float(*W_1) && eigen_val != float(*W_n))
					{
						W_2 = eigen_val;
					}
				}

				if (float(*W_1) <= THRESHOLD)
				{
					return 0;
				}
				else if (W_2 <= THRESHOLD && THRESHOLD < float(*W_1))
				{
					return 1;
				}
				else if (float(*W_n) <= THRESHOLD && THRESHOLD < W_2)
				{
					return 2;
				}
				else if (THRESHOLD < float(*W_n))
				{
					return 3;
				}
				else
				{
					printf("[Err]: wrong eigen values\n");
					return -1;
				}
			}

			// @even: ���ӿ����ӵ�superpixel: ��������ϵ
			int ConnectSuperpixels(const float THRESH_1,
				const float THRESH_2,
				const DepthMap& depth_map,
				const float* K_inv_arr,
				const float* R_inv_arr,
				const float* T_arr,
				std::unordered_map<int, std::vector<float>>& eigen_vals_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, cv::Point3f>& center_map,
				cv::Mat& labels,
				std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map);

			// @even: ���ӿ����ӵ�superpixel: �������ϵ
			int ConnectSuperpixelsCam(const float THRESH_1,
				const float THRESH_2,
				const DepthMap& depth_map,
				const float* K_inv_arr,
				std::unordered_map<int, std::vector<float>>& plane_map,
				std::unordered_map<int, std::vector<float>>& eigen_vals_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, cv::Point3f>& center_map,
				cv::Mat& labels,
				std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map);

			// @even: ����filtered depth map�ϲ�superpixel
			int MergeSuperpixels(const cv::Mat& src,
				const int min_num,
				cv::Mat& labels,
				std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map);

			// @even: �����ڽӱ�, ����mask
			int DrawMaskOfSuperpixels(const cv::Mat& labels, cv::Mat& Input);

			// @even: ����labels��ȡ�ڽӱ�
			std::unordered_map<int, std::set<int>> GetNeighborMap(const cv::Mat & labels);

			// @even: ��������superpixel�İ��Ͼ���
			float BaDistOf2Superpixel(const cv::Mat& src,
				const std::vector<cv::Point2f>& superpix1,
				const std::vector<cv::Point2f>& superpix2,
				const int num_bins=15);

			//��src��enhance�Ľ���ϲ���һ��
			void MergeDepthNormalMaps(const bool haveMerged = false, const bool selectiveJBPF = false);

			//����Ⱥͷ�����ͼ������ѡ���Ե�����˫��(����)�˲���ֵ
			void selJointBilateralPropagateFilter(const cv::Mat& joint, 
				const DepthMap& depthMap, const NormalMap& normalMap,
				const float *refK,
				const double sigma_color, const double sigma_space,
				int radius, const int maxSrcPoint,
				DepthMap& outDepthMap, NormalMap& outNormalMap) const;

			// ������֪�˲�(˫���˲��ı���)Noise-aware filter
			void NoiseAwareFilter(const cv::Mat& joint, 
				DepthMap& depthMap, const NormalMap& normalMap,
				const float* refK,
				const double& sigma_space, const double& sigma_color, const double& sigma_depth,
				const float& THRESH,
				const float& eps, const float& tau,
				const bool is_propagate,
				int radius, 
				DepthMap& outDepthMap, NormalMap& outNormalMap) const;

			//  joint-trilateral-upsampling: JTU
			void JTU(const cv::Mat& joint,
				DepthMap& depthMap, const NormalMap& normalMap,
				const float* refK,
				const double& sigma_space, const double& sigma_color, double& sigma_depth,
				const float& THRESH,
				const bool is_propagate,
				int radius,
				DepthMap& outDepthMap, NormalMap& outNormalMap) const;

			//�����ͼ�ͷ�����ͼ��������˫���˲�
			void jointBilateralFilter_depth_normal_maps(const cv::Mat &joint, const DepthMap &depthMap, const NormalMap &normalMap,
				const float *refK, const double sigma_color, const double sigma_space, int radius,
				DepthMap &outDepthMap, NormalMap &outNormalMap) const;

			void distanceWeightFilter(const DepthMap &depthMap, const NormalMap &normalMap,
				const float *refK, const double sigma_color, const double sigma_space, int radius,
				DepthMap &outDepthMap, NormalMap &outNormalMap) const;

		private:
			std::mutex m_lock;

			Options options_;
			Model m_model;
			bool hasReadMapsPhoto_;  // �Ƿ����ͼ��һ����mapͼ
			bool hasReadMapsGeom_;  // �Ƿ���뼸��һ����mapͼ�����߲���ͬʱΪtrue��
			std::vector<bool> hasBitMaps_;
			std::vector<cv::Mat> bitMaps_;
			std::vector<DepthMap> m_depth_maps;
			std::vector<NormalMap> m_normal_maps;
			std::vector<std::pair<float, float>> depth_ranges_;
			std::vector<cv::Mat> slicLabels_;

		};

	}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_WORKSPACE_H_
