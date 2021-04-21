#ifndef COLMAP_SRC_MVS_MODEL_H_
#define COLMAP_SRC_MVS_MODEL_H_

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "depth_map.h"
#include "image.h"
#include "normal_map.h"


namespace colmap {
	namespace mvs {

		// Simple sparse model class.
		struct Model
		{
			struct Point
			{
				float x = 0;
				float y = 0;
				float z = 0;
				std::vector<int> track;
			};

			std::string m_src_img_rel_dir;

			// Read the model from different data formats.
			void Read(const std::string& path, const std::string& format, const std::string &newPath);
			void ReadFromCOLMAP(const std::string& path, const std::string &newPath);
			void ReadFromBundlerOfColMap(const std::string& path, const std::string &newPath);

			// @even ��ȡimages.txt
			std::unordered_map<std::string, int> ReadImagesTxt(const std::string& root);

			void ReadFromPMVS(const std::string& path, const std::string &newPath);

			void RunUndistortion(const std::string& outPutpath);

			// Get the image identifier for the given image name.
			int GetImageId(const std::string& name) const;
			std::string GetImageName(const int image_id) const;

			// For each image, determine the maximally overlapping images, sorted based on
			// the number of shared points subject to a minimum robust average
			// triangulation angle of the points.
			std::vector<std::vector<int>> GetMaxOverlappingImgs(
				const size_t num_images, const double min_triangulation_angle) const;

			// Compute the robust minimum and maximum depths from the sparse point cloud.
			std::vector<std::pair<float, float>> ComputeDepthRanges() const;

			// Compute the number of shared points between all overlapping images.
			std::vector<std::map<int, int>> ComputeSharedPoints() const;

			// Compute the median triangulation angles between all overlapping images.
			std::vector<std::map<int, float>> ComputeTriangulationAngles(
				const float percentile = 50) const;

			// 3D��ά��ͶӰ��ͼ����2D
			void ProjectToImage();

			// 2D��㷴ͶӰ��3D��������ϵ
			// ���������������...ֱ��ʹ�������, �����ظ����������
			inline cv::Point3f Model::BackProjTo3D_1(const float* K_arr,
				const float* R_arr,
				const float* T_arr,
				const float& depth,
				const cv::Point2f pt2D)
			{
				const Eigen::Vector3f pt2D_H(depth * pt2D.x, depth * pt2D.y, depth);  // ��X_2D

				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr);
				const Eigen::Vector3f X_cam = K.inverse() * pt2D_H;
				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R_arr);
				const Eigen::Vector3f pt3D = R.inverse() \
					* (X_cam - Eigen::Map <const Eigen::Vector3f>(T_arr));

				cv::Point3f pt3D_CV(pt3D(0), pt3D(1), pt3D(2));
				return pt3D_CV;
			}

			// ��ͶӰ��3D�ռ�(��������ϵ)
			inline cv::Point3f Model::BackProjTo3D(const float* K_inv_arr,
				const float* R_inv_arr,
				const float* T_arr,
				const float& depth,
				const cv::Point2f pt2D)
			{
				const Eigen::Vector3f pt2D_H(depth * pt2D.x, depth * pt2D.y, depth);  // ��X_2D

				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_inv = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_inv_arr);
				const Eigen::Vector3f X_cam = K_inv * pt2D_H;
				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_inv = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R_inv_arr);
				const Eigen::Vector3f pt3D = R_inv \
					* (X_cam - Eigen::Map <const Eigen::Vector3f>(T_arr));

				cv::Point3f pt3D_CV(pt3D(0), pt3D(1), pt3D(2));
				return pt3D_CV;
			}

			// ��ͶӰ��3D�ռ�(�������ϵ)
			inline cv::Point3f Model::BackProjTo3DCam(const float* K_inv_arr,
				const float& depth,
				const cv::Point2f& pt2D)
			{
				const Eigen::Vector3f pt2D_H(depth * pt2D.x, depth * pt2D.y, depth);  // ��X_2D

				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_inv = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_inv_arr);
				const Eigen::Vector3f X_cam = K_inv * pt2D_H;

				return cv::Point3f(X_cam(0), X_cam(1), X_cam(2));
			}

			// ��С����3D�ռ�ƽ�����OLS
			cv::Mat PlaneFitOLS(const std::vector<cv::Point3f>& Pts3D);

			// 3���ռ��(�ǹ���)ȷ��һ���ռ�ƽ��
			int PlaneFitBy3Pts(const cv::Point3f* pts, float* plane_arr);

			// ͳ��ƽ����ϲв�
			float MeanDistOfPtToPlane(const std::vector<cv::Point3f>& Pts3D, const cv::Mat& plane);

			// ����ԭͼ����Թ����ռ��·��
			void SetSrcImgRelDir(const std::string& img_dir);

			// @even: ͨ��imga_id�õ�image name
			inline void InitImageID2ImgName()
			{
				if (0 != this->m_img_name2image_id.size())
				{
					for (auto it = m_img_name2image_id.begin();
						it != m_img_name2image_id.end();
						++it)
					{
						this->m_image_id2img_name[it->second] = it->first;
					}
					printf("[Note]: m_image_id2img_name initialized\n");
				}
				else
				{
					printf("[Err]: m_img_name2image_id not initialized\n");
				}
			}

			// ---------------------------------
			std::vector<Image> m_images;
			std::vector<Point> m_points;

			//��ά����ͼ�������ͶӰ���꣨��Σ�z�����ʾ���ֵ��
			std::vector<std::vector<cv::Point3f>> m_img_pts;

		private:
			std::vector<std::string> m_img_names;
			std::unordered_map<std::string, int> m_img_name_to_id;
			std::unordered_map<std::string, int> m_img_name2image_id;
			std::unordered_map<int, std::string> m_image_id2img_name;
		};

	}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_MODEL_H_
