#ifndef COLMAP_SRC_MVS_CONSISTENCY_GRAPH_H_
#define COLMAP_SRC_MVS_CONSISTENCY_GRAPH_H_

#include <string>
#include <vector>

#include <opencv2\core.hpp>

//#include <Eigen/Core>


namespace colmap {
namespace mvs {

// List of geometrically consistent images, in the following format:
//
//    r_1, c_1, N_1, i_11, i_12, ..., i_1N_1,
//    r_2, c_2, N_2, i_21, i_22, ..., i_2N_2, ...
//
// where r, c are the row and column image coordinates of the pixel,
// N is the number of consistent images, followed by the N image identifiers.
// Note that only pixels are listed which are not filtered and that the
// consistency graph is only filled if filtering is enabled.
class ConsistencyGraph {
 public:
  ConsistencyGraph();
  ConsistencyGraph(const size_t width, const size_t height,
                   const std::vector<int>& data);

  size_t GetNumBytes() const;

  void GetImageIds(const int row, const int col, int* num_images,
                   const int** image_ids) const;

  void Read(const std::string& path);
  void Write(const std::string& path) const;

 private:
  void InitializeMap(const size_t width, const size_t height);

  const static int kNoConsistentImageIds;
  std::vector<int> data_;
  //Eigen::MatrixXi map_;
  cv::Mat map_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CONSISTENCY_GRAPH_H_
