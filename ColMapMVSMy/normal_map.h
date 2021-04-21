#ifndef COLMAP_SRC_MVS_NORMAL_MAP_H_
#define COLMAP_SRC_MVS_NORMAL_MAP_H_

#include <string>
#include <vector>

#include "mat.h"
#include <opencv2\highgui.hpp>

namespace colmap {
namespace mvs {

// Normal map class that stores per-pixel normals as a MxNx3 image.
class NormalMap : public Mat<float> {
 public:
  NormalMap();
  NormalMap(const size_t width, const size_t height);
  explicit NormalMap(const Mat<float>& mat);

  void Rescale(const float factor);
  void Downsize(const size_t max_width, const size_t max_height);

  cv::Mat ToBitmap() const;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_NORMAL_MAP_H_
