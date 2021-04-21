#include "depth_map.h"
#include "math.h"

namespace colmap {
	namespace mvs {

		DepthMap::DepthMap() : DepthMap(0, 0, -1.0f, -1.0f) {}

		DepthMap::DepthMap(const float depth_min, const float depth_max)
			: depth_min_(depth_min), depth_max_(depth_max) {}

		DepthMap::DepthMap(const size_t width, const size_t height,
			const float depth_min, const float depth_max)
			: Mat<float>(width, height, 1),
			depth_min_(depth_min),
			depth_max_(depth_max) {}

		DepthMap::DepthMap(const Mat<float>& mat, const float depth_min,
			const float depth_max)
			: Mat<float>(mat.GetWidth(), mat.GetHeight(), mat.GetDepth()),
			depth_min_(depth_min),
			depth_max_(depth_max)
		{
			assert(mat.GetDepth() == 1);
			data_ = mat.GetData();
		}

		void DepthMap::Rescale(const float factor)
		{
			if (width_ * height_ == 0)
			{
				return;
			}

			const size_t new_width = std::round(width_ * factor);
			const size_t new_height = std::round(height_ * factor);
			std::vector<float> new_data(new_width * new_height);
			//DownsampleImage(data_.data(), height_, width_, new_height, new_width,
			//                new_data.data());

			data_ = new_data;
			width_ = new_width;
			height_ = new_height;

			data_.shrink_to_fit();
		}

		void DepthMap::Downsize(const size_t max_width, const size_t max_height) 
		{
			if (height_ <= max_height && width_ <= max_width) 
			{
				return;
			}
			const float factor_x = static_cast<float>(max_width) / width_;
			const float factor_y = static_cast<float>(max_height) / height_;
			Rescale(std::min(factor_x, factor_y));
		}

		cv::Mat DepthMap::ToBitmap(const float min_percentile,
			const float max_percentile) const
		{
			assert(width_ > 0);
			assert(height_ > 0);

			cv::Mat bitmap(height_, width_, CV_8UC3);

			std::vector<float> valid_depths;
			valid_depths.reserve(data_.size());  // 先分配好内存

			for (const float depth : data_)
			{
				if (depth > 0)
				{
					valid_depths.push_back(depth);
				}
			}

			const float robust_depth_min = Percentile(valid_depths, min_percentile);
			const float robust_depth_max = Percentile(valid_depths, max_percentile);
			const float robust_depth_range = robust_depth_max - robust_depth_min;

			for (size_t y = 0; y < height_; ++y)
			{
				for (size_t x = 0; x < width_; ++x)
				{
					const float depth = GetDepth(y, x);
					if (depth > 0.0f)
					{
						const float robust_depth =
							std::max(robust_depth_min, std::min(robust_depth_max, depth));
						const float gray =
							(robust_depth - robust_depth_min) / robust_depth_range;

						const cv::Vec3f colorF(255 * base(gray - 0.25f),
							255 * base(gray),
							255 * (gray + 0.25f));

						cv::Vec3b color;  // BGR

						color[0] = std::min((float)255, std::max((float)0, round(colorF[2])));
						color[1] = std::min((float)255, std::max((float)0, round(colorF[1])));
						color[2] = std::min((float)255, std::max((float)0, round(colorF[0])));
						bitmap.at<cv::Vec3b>(y, x) = color;
					}
					else
					{
						bitmap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
					}
				}
			}

			return bitmap;
		}

		cv::Mat DepthMap::ToBitmapGray(const float min_percentile, const float max_percentile)
		{
			assert(width_ > 0);
			assert(height_ > 0);

			cv::Mat bitmap(height_, width_, CV_8UC1);  // 用opencv Mat存储

			std::vector<float> valid_depths;
			valid_depths.reserve(data_.size());
			for (const float depth : data_)
			{
				if (depth > 0.0f)
				{
					valid_depths.push_back(depth);
				}
			}

			const float robust_depth_min = Percentile(valid_depths, min_percentile);
			const float robust_depth_max = Percentile(valid_depths, max_percentile);

			// 把鲁棒的深度值记录下来，方便之后转换
			last_depth_min = robust_depth_min;
			last_depth_max = robust_depth_max;

			const float robust_depth_range = robust_depth_max - robust_depth_min;
			for (size_t y = 0; y < height_; ++y)
			{
				for (size_t x = 0; x < width_; ++x)
				{
					const float depth = GetDepth(y, x);
					if (depth > 0.0f)
					{
						const float robust_depth =
							std::max(robust_depth_min, std::min(robust_depth_max, depth));
						const float gray =
							(robust_depth - robust_depth_min) / robust_depth_range;

						//const float colorF = 128 * gray + 64;
						const float colorF = 192.0f * gray + 32.0f;
						//const float colorF = 224 * gray + 16;

						uchar color;
						color = std::min((uchar)255, std::max((uchar)0, uchar(colorF + 0.5f)));
						bitmap.at<uchar>(y, x) = color;
					}
					else
					{
						bitmap.at<uchar>(y, x) = 0;
					}
				}
			}

			return bitmap;
		}

		void DepthMap::fillDepthWithMat(const cv::Mat& mat)
		{
			assert(mat.cols == width_ && mat.rows == height_);

			for (size_t y = 0; y < height_; ++y)
			{
				for (size_t x = 0; x < width_; ++x)
				{
					//printf("depth(%d, %d): %.3f\n", y, x, mat.at<float>(y, x));
					data_.at(y * width_ + x) = mat.at<float>(y, x);
				}
			}
		}

		// 灰色图像转换为真实深度值
		void DepthMap::mat2depth(cv::Mat& mat)
		{
			const int row = mat.rows;
			const int col = mat.cols;

			assert(width_ == col);
			assert(height_ == row);

			//std::fill(data_.begin(), data_.end(), 0.0f); // 全部赋值为0

			const float robust_depth_range = last_depth_max - last_depth_min;
			for (size_t y = 0; y < height_; ++y)
			{
				for (size_t x = 0; x < width_; ++x)
				{
					const float gray = mat.at<uchar>(y, x);
					if (gray == 0)
						continue;

					//const float scale = (gray - 64) / 128;
					const float scale = (gray - 32) / 192;
					//const float scale = (gray - 16) / 224;
					const float depth1 = scale * robust_depth_range + last_depth_min;

					const float depth = std::min(depth_max_, std::max(depth_min_, depth1));

					data_.at(y * width_ + x) = depth;
				}
			}
		}

		float base(const float val)
		{
			if (val <= 0.125f)
			{
				return 0.0f;
			}
			else if (val <= 0.375f)
			{
				return interPolate(2.0f * val - 1.0f, 0.0f, -0.75f, 1.0f, -0.25f);
			}
			else if (val <= 0.625f)
			{
				return 1.0f;
			}
			else if (val <= 0.87f)
			{
				return interPolate(2.0f * val - 1.0f, 1.0f, 0.25f, 0.0f, 0.75f);
			}
			else
			{
				return 0.0f;
			}
		}

		inline float interPolate(const float val, const float y0, const float x0,
			const float y1, const float x1)
		{
			return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
		}

	}  // namespace mvs
}  // namespace colmap
