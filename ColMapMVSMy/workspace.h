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

				////是否利用细节增强的图像
				bool bDetailEnhance = true;

				////是否利用结构增强的图像
				bool bStructureEnhance = false;

				////是否进行降采样处理(对图像尺度和摄像机参数做修改，稀疏三维模型不变)
				bool bDown_sampling = false;

				//降采样的尺度
				float fDown_scale = 4.0f;

				bool bOurs = false;  // 联合双边传播上采样
				bool bOursFast = false;  // 快速联合双边传播上采样方法

				bool bBilinearInterpolation = 0;  // 双线性插值上采样

				bool bFastBilateralSolver = 0;  // 快速双边求解器

				bool bJointBilateralUpsampling = 0;  // 联合双边上采样

				bool bBilateralGuidedUpsampling = 0;  // 双边引导上采样

				// Location and type of workspace.
				std::string workspace_path;
				std::string workspace_format;
				std::string input_type;
				std::string input_type_geom;
				std::string newPath;
				std::string src_img_dir;

				//去扭曲的目录
				std::string undistorte_path;

				// 超像素分割目录
				std::string slic_path;
			};

			// 构造函数: 从bundler数据中读取稀疏点云信息
			Workspace(const Options& options);

			const Model& GetModel() const;
			const cv::Mat& GetBitmap(const int image_id);
			const DepthMap& GetDepthMap(const int image_id) const;
			const NormalMap& GetNormalMap(const int image_id) const;
			const ConsistencyGraph& GetConsistencyGraph(const int image_id) const;

			const void ReadDepthAndNormalMaps(const bool isGeometric);

			const std::vector<DepthMap>& GetAllDepthMaps() const;
			const std::vector<NormalMap>& GetAllNormalMaps() const;

			//将算出来的结果，写到workspace对应的变量里面
			void WriteDepthMap(const int image_id, const DepthMap &depthmap);
			void WriteNormalMap(const int image_id, const NormalMap &normalmap);
			void WriteConsistencyGraph(const int image_id, const ConsistencyGraph &consistencyGraph);

			//执行超像素分割
			void runSLIC(const std::string& path);

			//将三维点投影到图像中
			void showImgPointToSlicImage(const std::string &path);

			//对深度图和法向图进行上采样，同时修改model中的图像信息
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

			//对法向量图进行类中值滤波
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

			// @even:为CV_32F数据进行斑点滤波操作
			template<typename T>
			void FilterSpeckles(cv::Mat & img, T newVal, int maxSpeckleSize, T maxDiff);

			// @even: 依据平面约束求深度值: 世界坐标系
			float GetDepthCoPlane(const float* K_inv_arr,
				const float* R_inv_arr,
				const float* T_arr,
				const float* plane_arr,
				const cv::Point2f& pt2D);

			// @even: 依据平面约束求深度值: 相机坐标系
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

			// @even: 点云切平面修正: 世界坐标系
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

			// @even: 点云切平面修正: 相机坐标系
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

			// 深度图对应的3D点云
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

			// @even: 根据3D坐标和相机外参(R, T)获取深度值: 世界坐标系
			inline float GetDepthBy3dPtRT(const float* R_arr,
				const float* T_arr,
				const cv::Point3f& pt3d)
			{
				// 3D空间点世界坐标
				const Eigen::Vector3f X(pt3d.x, pt3d.y, pt3d.z);
				return Eigen::Map<const Eigen::Vector3f>(&R_arr[6]).dot(X) + T_arr[2];
			}

			// @even: 根据3D坐标(相机坐标系)和相机内参K获取深度值: 相机坐标系
			inline float GetDepthBy3dPtK(const float* K_arr,
				const cv::Point3f& pt3d)
			{
				// 3D空间点坐标:相机坐标系
				const Eigen::Vector3f X(pt3d.x, pt3d.y, pt3d.z);
				Eigen::Vector3f pt2d_h = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * X;
				return pt2d_h[2];
			}

			// @even: 计算点云方差
			inline int GetPointCloudVariance(const std::vector<cv::Point3f>& pts3D,
				float* variance)
			{
				// 分别计算x,y,z的方差
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

				// 计算方差
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

			// @even: 基于切平面对点云进行网格平滑: 世界坐标系
			int SmoothPointCloud(const float* R_arr,
				const float* T_arr,
				const float* K_inv_arr,
				const float* R_inv_arr,
				DepthMap& depth_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, std::vector<float>>& ei_vects_map);

			// @even: 基于切平面对点云进行网格平滑: 世界坐标系
			int SmoothPointCloudCam(const float* K_arr,
				const float* K_inv_arr,
				const std::vector<cv::Point2f>& pts2D,
				const float* normal,
				const float* tangent_1,
				const float* tangent_2,
				DepthMap& depth_map);

			// @even: 判断跟切平面相交的3D点是否在切平面范围内
			bool IsPt3DInPlaneRange(const cv::Point3f& center, const cv::Point3f& pt3D,
				const float ei_val_tagent_1, const float ei_val_tagent_2,
				const float* ei_vect_tagent_1, const float* ei_vect_tagent_2,  // 切平面的特征向量的2个切向分量
				const float fold)
			{
				// 方向向量
				cv::Point3f vector = pt3D - center;  // vector: center -> pt3D

				// 判断第一个切向量方向
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

				// 判断第二个切向量方向
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

			// @even: 切平面范围内的四个顶点在3D(世界)坐标系中的坐标
			// 然后3D投影到2D, 计算2D搜索范围
			int GetSearchRange(const cv::Point3f& center, 
				const float* P, 
				const float* ei_vals,
				const float* ei_vects,
				const float fold,
				float* x_range, float* y_range);

			// @even: 切平面范围内的四个顶点在3D(相机)坐标系中的坐标
			// 然后3D投影到2D, 计算2D搜索范围
			int GetSearchRangeCam(const cv::Point3f& center,
				const float* K_arr,
				const float* ei_vals,
				const float* ei_vects,
				const float fold,
				float* x_range, float* y_range);

			// @even: 计算image(view)配对表
			inline std::unordered_map<int, std::set<int>> GetImgPairMap()
			{
				std::unordered_map<int, std::set<int>> pair_map;

				for (auto pt3D : this->m_model.m_points)
				{
					// 遍历每个点的track
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

			// @even: 根据配对表, 计算image(view)的平均基线距离
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

			// @even: 读取Consistency graph
			int ReadConsistencyGraph(const std::string& view_name);

			// @even: 根据superpixel范围(2D坐标)动态确定sigma
			inline float GetSigmaOfPts2D(const std::vector<cv::Point2f>& pts2d)
			{
				/*
				1. 取x, y范围的较小值(or均值)
				2. 高斯分布, 令6*sigma == 较小的取值范围(+-3sigma)
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
				//return x_range <= y_range ? x_range/6.0f : y_range/6.0f;  // 取较小的range
				return x_range >= y_range ? x_range / 6.0f : y_range / 6.0f;  // 取较大的range
			}

			inline float GetSigmaOfPts3D(const std::vector<cv::Point3f>& pts3D)
			{
				/*
				1. 取x, y, z分布范围的均值
				2. 高斯分布, 令6*sigma == 较小的取值范围(+-3sigma)
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

			// @even: 测试深度图(基于超像素分割): 世界坐标系
			void TestDepth1();

			// @even: 测试深度图(基于超像素分割): 相机坐标系
			void TestDepth5();

			// ----- @even: 基于Speckle过滤, 3D平面拟合和MRF优化
			struct HashFuncPt2f  // 自定义哈希函数
			{
				std::size_t operator()(const cv::Point2f& pt2d) const
				{
					using std::size_t;
					using std::hash;

					// 使用以为+异或运算得到哈希值(较少碰撞)
					return hash<float>()(pt2d.x) ^ (hash<float>()(pt2d.y) << 1);
				}
			};
			struct EuqualPt2f  // 自定义相等判断条件
			{
				bool operator () (const cv::Point2f& pt2f_1, const cv::Point2f& pt2f_2) const
				{
					return pt2f_1.x == pt2f_2.x && pt2f_1.y == pt2f_2.y;
				}
			};

			// @even: TODO 计算MRF能量
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

			// @even: 计算MRF能量的线程函数(每个线程计算一部分2D点)
			void ThFuncPts2dEnergy(const std::vector<cv::Point2f>& Pts2D,
				const int Start, const int End, const int plane_count,
				const std::unordered_map<int, int>& plane_id2SPLabel,
				const std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				const cv::Mat& labels,
				const DepthMap& depth_map,
				const float beta,
				const int radius,
				std::vector<int>& Pt2DSPLabelsRet);

			// 基于Speckle过滤, 3D平面拟合和MRF优化
			void TestDepth6();

			// 按块分割深度图,分块计算3d点集,平面拟合,生成候选label_array和labels
			int SplitDepthMat(const cv::Mat& depth_mat,
				const int blk_size,  // block size
				const float* K_inv_arr,  // 相机内参矩阵的逆
				std::vector<int>& blks_pt_cnt,  // 记录所有block的pt2d数量
				std::vector<cv::Point2f>& blks_pts2d,  // 记录所有block的pt2d点坐标
				std::vector<int>& blks_pt_cnt_has,  // 记录待处理block的有深度值点个数
				std::vector<int>& blks_pt_cnt_non,  // 记录待处理block的无深度值点个数
				std::vector<std::vector<float>>& plane_equa_arr,  // 记录作为label的blk_id对应的平面方程
				std::vector<int>& label_blk_ids,  // 记录有足够多深度值点的blk_id: 可当作label
				std::vector<int>& proc_blk_ids,  // 记录待(MRF)处理的blk_id
				std::vector<float>& proc_blks_depths_has,  // 记录待处理block的有深度值(组成的数组)
				std::vector<int>& proc_blks_pts2d_has_num,  // 记录待处理block的有深度值点个数
				std::vector<int>& proc_blks_pts2d_non_num,  // 记录待处理block无深度点个数
				std::vector<cv::Point2f>& proc_blks_pt2d_non,  // 记录待处理block的无深度值点坐标
				std::vector<int>& all_blks_labels,  // 记录每个block对应的label(blk_id): 初始label数组
				int& num_y, int& num_x);

			// 基于Speckle过滤, 3D平面拟合和MRF优化: 分块思想解决非平面问题(微分)
			void TestDepth7();

			// @even: 测试深度图(基于快速平面检测提取)opencv点云
			//void TestDepth2();

			// @even: 
			void runMRFOptimization();

			// @even: 快速平面提取, 世界坐标系(K[R|t])
			void TestDepth3();

			// 输出*1000的uint16深度图
			void OutputU16Depthmap();

			// @even: 快速平面提取, 相机坐标系(K)
			void TestDepth4();

			// @even: 某个superpixel进行PCA拟合: 世界坐标系
			int FitPlaneForSuperpixel(const DepthMap& depth_map,
				const float* K_inv_arr,
				const float* R_inv_arr,
				const float* T_arr,
				const std::vector<cv::Point2f>& Pts2D,
				std::vector<float>& plane_normal,
				std::vector<float>& eigen_vals,
				std::vector<float>& center_arr);

			// @even: 某个superpixel进行PCA拟合: 相机坐标系
			int FitPlaneForSuperpixelCam(const DepthMap& depth_map,
				const float* K_inv_arr,
				const std::vector<cv::Point2f>& Pts2D,
				std::vector<float>& plane_arr,
				std::vector<float>& plane_normal,
				std::vector<float>& eigen_vals,
				std::vector<float>& center_arr);

			// @even: 重载FitPlaneForSuper 1
			int FitPlaneForSuperpixel(const std::vector<cv::Point3f>& Pts3D,
				float* plane_normal,
				float* eigen_vals,
				float* eigen_vects,
				float* center_arr);

			// @even: 重载FitPlaneForSuper 2
			int FitPlaneForSuperpixel(const std::vector<cv::Point3f>& Pts3D,
				std::vector<float>& plane_arr,
				std::vector<float>& eigen_vals,
				std::vector<float>& eigen_vects,
				cv::Point3f& center_pt);

			// @even: 每个superpixel进行PCA拟合平面: 世界坐标系
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

			// @even: 每个superpixel进行PCA拟合平面: 相机坐标系
			int FitPlaneForSPsCam(const DepthMap& depth_map,
				const float* K_inv_arr,
				const std::unordered_map<int, std::vector<cv::Point2f>>& labels_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
				std::unordered_map<int, cv::Point3f>& center_map,
				std::unordered_map<int, std::vector<float>>& eigen_vals_map,
				std::unordered_map<int, std::vector<float>>& eigen_vects_map,
				std::unordered_map<int, std::vector<float>>& plane_normal_map,
				std::unordered_map<int, std::vector<float>>& plane_map);

			// 判断一个superpixel对应的点云是否是平面 
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

				// 1. 取(10%~90%)到平面的平均距离
				// 2. 到平面平均距离的方差
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
				//}   // 方差还是不靠谱

				return true;
			}

			// 在一个superpixel内部计算JBU深度值
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

			// @even: 根据每个superpixel对应的plane的特征值确定平面维度
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

			// @even: 连接可连接的superpixel: 世界坐标系
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

			// @even: 连接可连接的superpixel: 相机坐标系
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

			// @even: 依据filtered depth map合并superpixel
			int MergeSuperpixels(const cv::Mat& src,
				const int min_num,
				cv::Mat& labels,
				std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
				std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map);

			// @even: 依据邻接表, 绘制mask
			int DrawMaskOfSuperpixels(const cv::Mat& labels, cv::Mat& Input);

			// @even: 根据labels获取邻接表
			std::unordered_map<int, std::set<int>> GetNeighborMap(const cv::Mat & labels);

			// @even: 计算两个superpixel的巴氏距离
			float BaDistOf2Superpixel(const cv::Mat& src,
				const std::vector<cv::Point2f>& superpix1,
				const std::vector<cv::Point2f>& superpix2,
				const int num_bins=15);

			//将src和enhance的结果合并在一起
			void MergeDepthNormalMaps(const bool haveMerged = false, const bool selectiveJBPF = false);

			//对深度和法向量图进行有选择性的联合双边(传播)滤波插值
			void selJointBilateralPropagateFilter(const cv::Mat& joint, 
				const DepthMap& depthMap, const NormalMap& normalMap,
				const float *refK,
				const double sigma_color, const double sigma_space,
				int radius, const int maxSrcPoint,
				DepthMap& outDepthMap, NormalMap& outNormalMap) const;

			// 噪声感知滤波(双边滤波的变种)Noise-aware filter
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

			//对深度图和法向量图进行联合双边滤波
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
			bool hasReadMapsPhoto_;  // 是否读入图像一致性map图
			bool hasReadMapsGeom_;  // 是否读入几何一致性map图，两者不能同时为true；
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
