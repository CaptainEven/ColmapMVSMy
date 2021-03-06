#include "normal_map.h"
#include <Eigen/Core>

namespace colmap {
namespace mvs {

NormalMap::NormalMap() : Mat<float>(0, 0, 3) {}

NormalMap::NormalMap(const size_t width, const size_t height)
    : Mat<float>(width, height, 3) {}

NormalMap::NormalMap(const Mat<float>& mat)
    : Mat<float>(mat.GetWidth(), mat.GetHeight(), mat.GetDepth())
{
  assert(mat.GetDepth()== 3);
  data_ = mat.GetData();
}

void NormalMap::Rescale(const float factor)
{
  if (width_ * height_ == 0) 
  {
    return;
  }

  const size_t new_width = std::round(width_ * factor);
  const size_t new_height = std::round(height_ * factor);
  std::vector<float> new_data(new_width * new_height * 3);

  // Resample the normal map.
  for (size_t d = 0; d < 3; ++d) {
    const size_t offset = d * width_ * height_;
    const size_t new_offset = d * new_width * new_height;
  //  DownsampleImage(data_.data() + offset, height_, width_, new_height,
  //                  new_width, new_data.data() + new_offset);
  }

  data_ = new_data;
  width_ = new_width;
  height_ = new_height;

  data_.shrink_to_fit();

  // Re-normalize the normal vectors.
  for (size_t r = 0; r < height_; ++r) {
    for (size_t c = 0; c < width_; ++c) {
      Eigen::Vector3f normal(Get(r, c, 0), Get(r, c, 1), Get(r, c, 2));
      const float squared_norm = normal.squaredNorm();
      if (squared_norm > 0) {
        normal /= std::sqrt(squared_norm);
      }
      Set(r, c, 0, normal(0));
      Set(r, c, 1, normal(1));
      Set(r, c, 2, normal(2));
    }
  }
}

void NormalMap::Downsize(const size_t max_width, const size_t max_height) {
  if (height_ <= max_height && width_ <= max_width) {
    return;
  }
  const float factor_x = static_cast<float>(max_width) / width_;
  const float factor_y = static_cast<float>(max_height) / height_;
  Rescale(std::min(factor_x, factor_y));
}

cv::Mat NormalMap::ToBitmap() const {
 assert(width_> 0);
  assert(height_> 0);
  assert(depth_== 3);

  cv::Mat bitmap(height_,width_, CV_8UC3);

  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_; ++x) {
      float normal[3];
      GetSlice(y, x, normal);
      if (normal[0] != 0 || normal[1] != 0 || normal[2] != 0) {
		  const cv::Vec3f colorF(127.5f*(-normal[0] + 1), 127.5f*(-normal[1] + 1), -255.0f*normal[2]);
		  cv::Vec3b color;

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

}  // namespace mvs
}  // namespace colmap
