#ifndef COLMAP_SRC_MVS_DEPTH_MAP_H_
#define COLMAP_SRC_MVS_DEPTH_MAP_H_

#include <string>
#include <vector>
#include <algorithm>

#include "mat.h"
#include <opencv2/highgui.hpp>

namespace colmap {
	namespace mvs {

		class DepthMap : public Mat<float>
		{
		public:
			DepthMap();
			DepthMap(const float depth_min, const float depth_max);
			DepthMap(const size_t width, const size_t height,
				const float depth_min, const float depth_max);
			DepthMap(const Mat<float>& mat, const float depth_min, const float depth_max);

			inline float GetDepthMin() const;
			inline float GetDepthMax() const;

			inline float GetDepth(const int row, const int col) const
			{
				return data_.at(row * width_ + col);
			}

			void Rescale(const float factor);
			void Downsize(const size_t max_width, const size_t max_height);

			// Bitmap��opencv�е�Mat�����ʾ
			cv::Mat ToBitmap(const float min_percentile, const float max_percentile) const;
			cv::Mat ToBitmapGray(const float min_percentile, const float max_percentile);

			// ��DepthMapת��ΪOpencv Mat: Ĭ��float32
			inline cv::Mat Depth2Mat()
			{
				cv::Mat mat(height_, width_, CV_32F);

				for (size_t y = 0; y < height_; ++y)
				{
					for (size_t x = 0; x < width_; ++x)
					{
						//if (data_.at(y * width_ + x) < 0.0f)
						//	printf("[%d, %d] depth: %.3f\n", x, y, data_.at(y * width_ + x));
						mat.at<float>(y, x) = data_.at(y * width_ + x);
					}
				}

				return mat;
			}

			// ��Opencv Mat���Depthmap
			void fillDepthWithMat(const cv::Mat& mat);

			void mat2depth(cv::Mat &mat);

			// �����Ϻ���alpha
			inline double CalculateAlpha(const double& eps,
				const double& tau,
				const double& omega_depth)
			{
				return 1.0 / (1.0 + exp(-eps * (omega_depth - tau)));
			}

			float depth_min_ = -1.0f;
			float depth_max_ = -1.0f;

		private:
			//float depth_min_ = -1.0f;
			//float depth_max_ = -1.0f;

			// ³������ȷ�Χ������ת����ͼ����ʽ
			float last_depth_min = 0.0f;
			float last_depth_max = 0.0f;
		};


		float base(const float val);
		inline float interPolate(const float val, const float y0, const float x0,
			const float y1, const float x1);

		////////////////////////////////////////////////////////////////////////////////
		// Implementation
		////////////////////////////////////////////////////////////////////////////////

		inline float DepthMap::GetDepthMin() const { return depth_min_; }

		inline float DepthMap::GetDepthMax() const { return depth_max_; }

	}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_DEPTH_MAP_H_
