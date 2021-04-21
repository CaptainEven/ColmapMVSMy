#ifndef COLMAP_SRC_MVS_FUSION_H_
#define COLMAP_SRC_MVS_FUSION_H_

#include <vector>
#include <memory>

#include <Eigen/Core>

#include "depth_map.h"
#include "image.h"
#include "mat.h"
#include "model.h"
#include "normal_map.h"
#include "workspace.h"

#include "math.h"


namespace colmap {
	namespace mvs {

		struct FusedPoint
		{
			float x = 0.0f;
			float y = 0.0f;
			float z = 0.0f;
			float nx = 0.0f;
			float ny = 0.0f;
			float nz = 0.0f;
			uint8_t r = 0;
			uint8_t g = 0;
			uint8_t b = 0;
		};

		class StereoFusion
		{
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW

				struct Options
			{
				// Maximum image size in either dimension.
				int max_image_size = 3200;  // 2000

				// Minimum number of fused pixels to produce a point.
				int min_num_pixels = 3;  // 5

				// Maximum number of pixels to fuse into a single point.
				int max_num_pixels = 10000;

				// Maximum depth in consistency graph traversal.
				int max_traversal_depth = 100;

				// Maximum relative difference between measured and projected pixel.
				double max_reproj_error = 2.0f;

				// Maximum relative difference between measured and projected depth.
				double max_depth_error = 0.01f;

				// Maximum angular difference in degrees of normals of pixels to be fused.
				double max_normal_error = 10.0f;

				// Number of overlapping images to transitively check for fusing points.
				int check_num_images = 50;

				// Cache size in gigabytes for fusion. The fusion keeps the bitmaps, depth
				// maps, normal maps, and consistency graphs of this number of images in
				// memory. A higher value leads to less disk access and faster fusion, while
				// a lower value leads to reduced memory usage. Note that a single image can
				// consume a lot of memory, if the consistency graph is dense.
				double cache_size = 32.0;

				// Check the options for validity.
				bool Check() const;

				// Print the options to stdout.
				void Print() const;
			};

			StereoFusion(const Options& options, 
						 const std::string& workspace_path,
						 const std::string& workspace_format,
						 const std::string& input_type);

			const std::vector<FusedPoint>& GetFusedPoints() const;

			void Run();
		private:

			void Fuse();

			const Options options_;
			const std::string workspace_path_;
			const std::string newPath_;
			const std::string workspace_format_;
			const std::string input_type_;
			const float max_squared_reproj_error_;
			const float min_cos_normal_error_;

			std::unique_ptr<Workspace> workspace_;
			std::vector<char> used_images_;
			std::vector<char> fused_imgs;
			std::vector<std::vector<int>> overlapping_imgs;
			std::vector<Mat<bool>> fused_pixel_masks_;
			std::vector<std::pair<int, int>> depth_map_sizes_;
			std::vector<std::pair<float, float>> bitmap_scales_;
			std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> P_;
			std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> inv_P_;
			std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> inv_R_;

			struct FusionData
			{
				int img_id = -1;
				int row = 0;
				int col = 0;
				int traversal_depth = -1;

				bool operator()(const FusionData& data_1, const FusionData& data_2)
				{
					return data_1.img_id > data_2.img_id;
				}
			};

			std::vector<FusionData> fusion_queue_;
			std::vector<FusedPoint> fused_points_;
			std::vector<float> fused_points_x_;
			std::vector<float> fused_points_y_;
			std::vector<float> fused_points_z_;
			std::vector<float> fused_points_nx_;
			std::vector<float> fused_points_ny_;
			std::vector<float> fused_points_nz_;
			std::vector<uint8_t> fused_points_r_;
			std::vector<uint8_t> fused_points_g_;
			std::vector<uint8_t> fused_points_b_;
		};

		// Write the point cloud to PLY file.
		void WritePlyText(const std::string& path,
			const std::vector<FusedPoint>& points);
		void WritePlyBinary(const std::string& path,
			const std::vector<FusedPoint>& points);

	}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_FUSION_H_
