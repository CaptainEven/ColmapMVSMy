#ifndef COLMAP_SRC_MVS_PATCH_MATCH_H_
#define COLMAP_SRC_MVS_PATCH_MATCH_H_

#include <iostream>
#include <memory>
#include <vector>

#include "depth_map.h"
#include "image.h"
#include "model.h"
#include "normal_map.h"
#include "consistency_graph.h"
#include "workspace.h"
#include "threading.h"


namespace colmap {
	namespace mvs {

		//class ConsistencyGraph;
		class PatchMatchCuda;
		class MyPatchMatch;


		// This is a wrapper class around the actual PatchMatchCuda implementation. This
		// class is necessary to hide Cuda code from any boost or Eigen code, since
		// NVCC/MSVC cannot compile complex C++ code.
		class PatchMatch {
		public:
			// Maximum possible window radius for the photometric consistency cost. This
			// value is equal to THREADS_PER_BLOCK in patch_match_cuda.cu and the limit
			// arises from the shared memory implementation.
			const static size_t kMaxWindowRadius = 32;

			struct Options
			{
				// Maximum image size in either dimension.
				int max_image_size = 400;  // -1

				// Index of the GPU used for patch match. For multi-GPU usage,
				// you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
			   // std::string gpu_index = "-1";
				int gpu_index = -1;

				// Depth range in which to randomly sample depth hypotheses.
				double depth_min = 0.0f;
				double depth_max = 1.0f;

				// Half window size to compute NCC photo-consistency cost.
				int window_radius = 5;

				// Number of pixels to skip when computing NCC. For a value of 1, every
				// pixel is used to compute the NCC. For larger values, only every n-th row
				// and column is used and the computation speed thereby increases roughly by
				// a factor of window_step^2. Note that not all combinations of window sizes
				// and steps produce nice results, especially if the step is greather than 2.
				int window_step = 1;

				// Parameters for bilaterally weighted NCC.
				double sigma_spatial = window_radius;
				double sigma_color = 0.2f;

				// Number of random samples to draw in Monte Carlo sampling.
				int num_samples = 15;

				// Spread of the NCC likelihood function.
				double ncc_sigma = 0.6f;

				// Minimum triangulation angle in degrees.
				double min_triangulation_angle = 1.0f;

				// Spread of the incident angle likelihood function.
				double incident_angle_sigma = 0.9f;

				// Number of coordinate descent iterations. Each iteration consists
				// of four sweeps from left to right, top to bottom, and vice versa.
				int num_iterations = 5;

				// 按照论文里面的内容，photometric和geomotric迭代次数不同
				int num_photometric_iteratoins = 3;  // 图像度量一致性3次迭代
				int num_geometric_iterations = 2;  // 几何度量一致性2次迭代

				// 是否利用稀疏点云数据
				bool bUse_sparse_points = false;

				// Whether to add a regularized geometric consistency term to the cost
				// function. If true, the `depth_maps` and `normal_maps` must not be null.
				bool geom_consistency = true;

				// The relative weight of the geometric consistency term w.r.t. to
				// the photo-consistency term.
				double geom_consistency_regularizer = 0.3f;

				// Maximum geometric consistency cost in terms of the forward-backward
				// reprojection error in pixels.
				double geom_consistency_max_cost = 3.0f;

				// Whether to enable filtering.
				bool filter = true;

				// Minimum NCC coefficient for pixel to be photo-consistent.
				double filter_min_ncc = 0.1f;

				// Minimum triangulation angle to be stable.
				double filter_min_triangulation_angle = 3.0f;

				// Minimum number of source images have to be consistent
				// for pixel not to be filtered.
				int filter_min_num_consistent = 2;

				// Maximum forward-backward reprojection error for pixel
				// to be geometrically consistent.
				double filter_geom_consistency_max_cost = 1.0f;

				// Cache size in gigabytes for patch match, which keeps the bitmaps, depth
				// maps, and normal maps of this number of images in memory. A higher value
				// leads to less disk access and faster computation, while a lower value
				// leads to reduced memory usage. Note that a single image can consume a lot
				// of memory, if the consistency graph is dense.
				double cache_size = 32.0;

				// Whether to write the consistency graph.
				bool write_consistency_graph = false;

				void Print() const;

				bool Check() const
				{
					assert(depth_min < depth_max);
					assert(depth_min >= 0.0f);
					assert(window_radius <= static_cast<int>(kMaxWindowRadius));
					assert(sigma_spatial > 0.0f);
					assert(sigma_color > 0.0f);
					assert(window_radius > 0);
					assert(num_samples > 0);
					assert(ncc_sigma > 0.0f);
					assert(min_triangulation_angle >= 0.0f);
					assert(min_triangulation_angle < 180.0f);
					assert(incident_angle_sigma > 0.0f);
					assert(num_iterations > 0);
					assert(geom_consistency_regularizer >= 0.0f);
					assert(geom_consistency_max_cost >= 0.0f);
					assert(filter_min_ncc >= -1.0f);
					assert(filter_min_ncc <= 1.0f);
					assert(filter_min_triangulation_angle >= 0.0f);
					assert(filter_min_triangulation_angle <= 180.0f);
					assert(filter_min_num_consistent >= 0);
					assert(filter_geom_consistency_max_cost >= 0.0f);
					assert(cache_size > 0);

					return true;
				}
			};

			struct Problem
			{
				// Index of the reference image.
				int ref_img_id = -1;

				// Indices of the source images.
				std::vector<int> src_img_ids;

				// Input images for the photometric consistency term.
				std::vector<Image>* images = nullptr;

				// Input depth maps for the geometric consistency term.
				std::vector<DepthMap>* depth_maps = nullptr;

				// Input normal maps for the geometric consistency term.
				std::vector<NormalMap>* normal_maps = nullptr;

				std::vector<float> sparsePoints;
				std::vector<float> sparseNormals;
				std::vector<int> tracks;

				// Print the configuration to stdout.
				void Print() const;
			};

			PatchMatch(const Options& options, Problem& problem);
			~PatchMatch();

			// Check the options and the problem for validity.
			void Check() const;

			// Run the patch match algorithm.
			void Run();

			// Get the computed values after running the algorithm.
			DepthMap GetDepthMap() const;
			NormalMap GetNormalMap() const;
			ConsistencyGraph GetConsistencyGraph() const;
			Mat<float> GetSelProbMap() const;

		private:
			const Options options_;
			const Problem problem_;
			std::unique_ptr<PatchMatchCuda> patch_match_cuda_;

		};

		// This thread processes all problems in a workspace. A workspace has the
		// following file structure, if the workspace format is "COLMAP":
		//
		//    images/*
		//    sparse/{cameras.txt, images.txt, points3D.txt}
		//    stereo/
		//      depth_maps/*
		//      normal_maps/*
		//      consistency_graphs/*
		//      patch-match.cfg
		//
		// The `patch-match.cfg` file specifies the images to be processed as:
		//
		//    image_name1.jpg
		//    __all__
		//    image_name2.jpg
		//    __auto__, 20
		//    image_name3.jpg
		//    image_name1.jpg, image_name2.jpg
		//
		// Two consecutive lines specify the images used to compute one patch match
		// problem. The first line specifies the reference image and the second line the
		// source images. Image names are relative to the `images` directory. In this
		// example, the first reference image uses all other images as source images,
		// the second reference image uses the 20 most connected images as source
		// images, and the third reference image uses the first and second as source
		// images. Note that all specified images must be reconstructed in the COLMAP
		// reconstruction provided in the `sparse` folder.

		class PatchMatchController {
		public:
			PatchMatchController(const PatchMatch::Options& options,
				const std::string& workspace_path,
				const std::string& workspace_format,
				const std::string& pmvs_option_name);

			void Run();
			void RunMyPm();

		private:
			bool findOrCreateDirectory(const char *path) const;
			void ReadWorkspace();
			void ReadModels();
			void ReadProblems();
			void ReadGpuIndices();
			void ProcessProblem(const PatchMatch::Options& options,
				const size_t problem_idx);

			void CalculateInitialNormals();  // 计算初始法向量
			void SetSparsePointToGpu(const size_t problem_idx);  // 为GPU准备稀疏点数据

			const PatchMatch::Options options_;
			const std::string workspace_path_;
			const std::string workspace_format_;
			const std::string pmvs_option_name_;

			std::string newPath_;  // 所有结果所在目录
			std::string depthMapPath_;
			std::string normalMapPath_;
			std::string consistencyGraphPath_;
			std::string undistortPath_;
			std::string slicPath_;

			std::unique_ptr<ThreadPool> thread_pool_;
			std::mutex workspace_mutex_;
			std::unique_ptr<Workspace> workspace_;
			std::vector<PatchMatch::Problem> problems_;
			std::vector<int> gpu_indices_;
			std::vector<std::pair<float, float>> depth_ranges_;
			std::vector<cv::Point3f> sparse_normals_;
		};


		void PrintHeading2(const std::string& heading);
		void PrintHeading1(const std::string& heading);

	}  // namespace mvs
}  // namespace colmap

#endif // COLMAP_SRC_MVS_PATCH_MATCH_H_
