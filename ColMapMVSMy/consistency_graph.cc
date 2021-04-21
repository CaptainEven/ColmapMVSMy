#include "consistency_graph.h"

#include <iostream>
#include <fstream>
#include <numeric>


namespace colmap {
namespace mvs {

const int ConsistencyGraph::kNoConsistentImageIds = -1;

ConsistencyGraph::ConsistencyGraph() {}

ConsistencyGraph::ConsistencyGraph(const size_t width, const size_t height,
                                   const std::vector<int>& data)
    : data_(data) {
  InitializeMap(width, height);
}

size_t ConsistencyGraph::GetNumBytes() const {
  return (data_.size() + map_.cols*map_.rows) * sizeof(int);
}

void ConsistencyGraph::GetImageIds(const int row, const int col,
                                   int* num_images,
                                   const int** image_ids) const {
  //const int index = map_(row, col);
	const int index = map_.at<int>(row, col);
  if (index == kNoConsistentImageIds) {
    *num_images = 0;
    *image_ids = nullptr;
  } else {
    *num_images = data_.at(index);
    *image_ids = &data_.at(index + 1);
  }
}

void ConsistencyGraph::Read(const std::string& path)
{
  std::fstream text_file(path, std::ios::in);
  if (!text_file.is_open())
  {
	  std::cout << path << " Open failed !" << std::endl;
	  std::system("pause");
	  std::exit(1);
  }

  size_t width = 0;
  size_t height = 0;
  size_t depth = 0;
  char unused_char;

  text_file >> width >> unused_char >> height >> unused_char >> depth >>
      unused_char;
  
  assert(width> 0);
  assert(height> 0);
  assert(depth> 0);

  int value;
  while (text_file >> value)
  {
	  data_.push_back(value);
  }

  text_file.close();

  InitializeMap(width, height);
}

void ConsistencyGraph::Write(const std::string& path) const 
{
  std::fstream text_file(path, std::ios::out);
  if (!text_file.is_open())
  {
	  std::cout << path << " Open failed !" << std::endl;
	  std::system("pause");
	  std::exit(1);
  }
  text_file << map_.cols << "&" << map_.rows << "&" << 1 << "&" << std::endl;

  //将data数据写入本地
  for (size_t i = 0; i < data_.size(); i++)
  {
	  text_file << data_.at(i) << " ";
  }

  text_file.close();
}

void ConsistencyGraph::InitializeMap(const size_t width, const size_t height)
{
 // map_.resize(height, width);
 // map_.setConstant(kNoConsistentImageIds);
  map_.create(height, width, CV_16UC1);
  map_.setTo(kNoConsistentImageIds);

  for (size_t i = 0; i < data_.size();)
  {
    const int num_images = data_.at(i + 2);
    if (num_images > 0)
	{
      const int col = data_.at(i);
      const int row = data_.at(i + 1);
      //map_(row, col) = i + 2;  // 表示该点所对应的一致性个数在data中的索引
	  map_.at<int>(row, col) = i + 2;
    }

    i += 3 + num_images;
  }
}

}  // namespace mvs
}  // namespace colmap
