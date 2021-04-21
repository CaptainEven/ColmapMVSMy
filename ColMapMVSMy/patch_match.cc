#include "patch_match.h"

#include <numeric>
#include <unordered_set>
#include <io.h>
#include <direct.h>

//#include "consistency_graph.h"
#include "patch_match_cuda.h"
#include "MyPatchMatch.h"
#include "math.h"

//#：构串操作符
//构串操作符#只能修饰带参数的宏的形参，它将实参的字符序列（而不是实参代表的值）转换成字符串常量
#define PrintOption(option) std::cout << #option ": " << option << std::endl

namespace colmap {
	namespace mvs {
		namespace {

			//void ImportPMVSOption(const Model& model, const std::string& path,
			//                      const std::string& option_name) {
			//  CreateDirIfNotExists(JoinPaths(path, "stereo"));
			//  CreateDirIfNotExists(JoinPaths(path, "stereo/depth_maps"));
			//  CreateDirIfNotExists(JoinPaths(path, "stereo/normal_maps"));
			//  CreateDirIfNotExists(JoinPaths(path, "stereo/consistency_graphs"));
			//
			//  const auto option_lines = ReadTextFileLines(JoinPaths(path, option_name));
			//  for (const auto& line : option_lines) {
			//    if (StringStartsWith(line, "timages")) {
			//      const auto elems = StringSplit(line, " ");
			//      const int num_images = std::stoi(elems[1]);
			//      CHECK_EQ(num_images + 2, elems.size());
			//      std::vector<std::string> image_names;
			//      image_names.reserve(num_images);
			//      for (size_t i = 2; i < elems.size(); ++i) {
			//        const int image_id = std::stoi(elems[i]);
			//        const std::string image_name = model.GetImageName(image_id);
			//        image_names.push_back(image_name);
			//      }
			//
			//      const auto patch_match_path = JoinPaths(path, "stereo/patch-match.cfg");
			//      const auto fusion_path = JoinPaths(path, "stereo/fusion.cfg");
			//      std::ofstream patch_match_file(patch_match_path, std::ios::trunc);
			//      std::ofstream fusion_file(fusion_path, std::ios::trunc);
			//      CHECK(patch_match_file.is_open()) << patch_match_path;
			//      CHECK(fusion_file.is_open()) << fusion_path;
			//      for (const auto ref_image_name : image_names) {
			//        patch_match_file << ref_image_name << std::endl;
			//        fusion_file << ref_image_name << std::endl;
			//        std::ostringstream line;
			//        for (const auto& image_name : image_names) {
			//          if (ref_image_name != image_name) {
			//            line << image_name << ",";
			//          }
			//        }
			//        const auto line_string = line.str();
			//        patch_match_file << line_string.substr(0, line_string.size() - 1)
			//                         << std::endl;
			//      }
			//    }
			//  }
			//}


		}  // namespace

		void PrintHeading1(const std::string& heading) 
		{
			std::cout << std::endl << std::string(78, '=') << std::endl;
			std::cout << heading << std::endl;
			std::cout << std::string(78, '=') << std::endl << std::endl;
		}

		void PrintHeading2(const std::string& heading)
		{
			std::cout << std::endl << heading << std::endl;
			std::cout << std::string(std::min<int>(heading.size(), 78), '-') << std::endl;
		}

		PatchMatch::PatchMatch(const Options& options, Problem& problem)
			: options_(options), problem_(problem) {}

		PatchMatch::~PatchMatch() {  }

		void PatchMatch::Options::Print() const 
		{
			PrintHeading2("PatchMatch::Options");
			PrintOption(max_image_size);
			PrintOption(gpu_index);
			PrintOption(depth_min);
			PrintOption(depth_max);
			PrintOption(window_radius);
			PrintOption(sigma_spatial);
			PrintOption(sigma_color);
			PrintOption(num_samples);
			PrintOption(ncc_sigma);
			PrintOption(min_triangulation_angle);
			PrintOption(incident_angle_sigma);
			PrintOption(num_iterations);
			PrintOption(num_photometric_iteratoins);
			PrintOption(num_geometric_iterations);
			PrintOption(geom_consistency);
			PrintOption(geom_consistency_regularizer);
			PrintOption(geom_consistency_max_cost);
			PrintOption(filter);
			PrintOption(filter_min_ncc);
			PrintOption(filter_min_triangulation_angle);
			PrintOption(filter_min_num_consistent);
			PrintOption(filter_geom_consistency_max_cost);
			PrintOption(write_consistency_graph);
		}

		void PatchMatch::Problem::Print() const
		{
			PrintHeading2("PatchMatch::Problem");

			PrintOption(ref_img_id);

			std::cout << "src_image_ids: ";
			if (!src_img_ids.empty())
			{
				for (size_t i = 0; i < src_img_ids.size() - 1; ++i)
				{
					std::cout << src_img_ids[i] << " ";
				}
				std::cout << src_img_ids.back() << std::endl;
			}
			else
			{
				std::cout << std::endl;
			}
		}

		void PatchMatch::Check() const
		{
			assert(options_.Check());

			//assert(!options_.gpu_index.empty());
			//const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
			// assert(gpu_indices.size()==1);
			//assert(gpu_indices[0]>= -1);

			assert(!problem_.images->empty());
			if (options_.geom_consistency) 
			{
				assert(!problem_.depth_maps->empty());
				assert(!problem_.normal_maps->empty());
				assert(problem_.depth_maps->size() == problem_.images->size());
				assert(problem_.normal_maps->size() == problem_.images->size());
			}

			assert(problem_.src_img_ids.size() > 0);

			// Check that there are no duplicate images and that the reference image
			// is not defined as a source image.
			std::set<int> unique_image_ids(problem_.src_img_ids.begin(),
				problem_.src_img_ids.end());
			unique_image_ids.insert(problem_.ref_img_id);
			assert(problem_.src_img_ids.size() + 1 == unique_image_ids.size());

			// Check that input data is well-formed.
			for (const int image_id : unique_image_ids)
			{
				assert(image_id >= 0);
				assert(image_id < problem_.images->size());

				const Image& image = problem_.images->at(image_id);
				assert(image.GetBitmap().cols > 0);;
				assert(image.GetBitmap().rows > 0);
				assert(image.GetBitmap().channels() == 1);//保证是灰度图像
				assert(image.GetWidth() == image.GetBitmap().cols);
				assert(image.GetHeight() == image.GetBitmap().rows);

				// Make sure, the calibration matrix only contains fx, fy, cx, cy.
				assert(std::abs(image.GetK()[1] - 0.0f) < 1e-6f);
				assert(std::abs(image.GetK()[3] - 0.0f) < 1e-6f);
				assert(std::abs(image.GetK()[6] - 0.0f) < 1e-6f);
				assert(std::abs(image.GetK()[7] - 0.0f) < 1e-6f);
				assert(std::abs(image.GetK()[8] - 1.0f) < 1e-6f);

				if (options_.geom_consistency)
				{
					assert(image_id < problem_.depth_maps->size());
					const DepthMap& depth_map = problem_.depth_maps->at(image_id);
					assert(image.GetWidth() == depth_map.GetWidth());
					assert(image.GetHeight() == depth_map.GetHeight());
				}
			}

			if (options_.geom_consistency)
			{
				const Image& ref_image = problem_.images->at(problem_.ref_img_id);
				const NormalMap& ref_normal_map =
					problem_.normal_maps->at(problem_.ref_img_id);
				assert(ref_image.GetWidth() == ref_normal_map.GetWidth());
				assert(ref_image.GetHeight() == ref_normal_map.GetHeight());
			}
		}

		void PatchMatch::Run()
		{
			PrintHeading2("PatchMatch::Run");

			Check();

			patch_match_cuda_.reset(new PatchMatchCuda(options_, problem_));
			patch_match_cuda_->Run();
		}

		DepthMap PatchMatch::GetDepthMap() const
		{
			return patch_match_cuda_->GetDepthMap();
		}

		NormalMap PatchMatch::GetNormalMap() const
		{
			return patch_match_cuda_->GetNormalMap();
		}

		Mat<float> PatchMatch::GetSelProbMap() const
		{
			return patch_match_cuda_->GetSelProbMap();
		}

		ConsistencyGraph PatchMatch::GetConsistencyGraph() const
		{
			const auto& ref_image = problem_.images->at(problem_.ref_img_id);
			return ConsistencyGraph(ref_image.GetWidth(), ref_image.GetHeight(),
				patch_match_cuda_->GetConsistentImageIds());
		}


		PatchMatchController::PatchMatchController(const PatchMatch::Options& options,
			const std::string& workspace_path,
			const std::string& workspace_format,
			const std::string& pmvs_option_name)
			: options_(options),
			workspace_path_(workspace_path),
			workspace_format_(workspace_format),
			pmvs_option_name_(pmvs_option_name)
		{
			std::cout << std::string(78, '=') << std::endl;

			std::cout << "=> Work Directory: " << workspace_path_ << std::endl;

			//*******************************************//
			//这个目录是所有结果所在目录
			newPath_ = "/MyMvs";
			//*******************************************//

			const std::string firstPath_ = workspace_path_ + newPath_;
			depthMapPath_ = firstPath_ + "/depth_maps";
			normalMapPath_ = firstPath_ + "/normal_maps";
			consistencyGraphPath_ = firstPath_ + "/consistency_graphs";
			undistortPath_ = firstPath_ + "/undistort_images";
			slicPath_ = firstPath_ + "/slic_image";

			//检查或者创建文件夹
			this->findOrCreateDirectory(firstPath_.c_str());
			this->findOrCreateDirectory(depthMapPath_.c_str());
			this->findOrCreateDirectory(normalMapPath_.c_str());
			this->findOrCreateDirectory(consistencyGraphPath_.data());
			this->findOrCreateDirectory(undistortPath_.data());
			this->findOrCreateDirectory(slicPath_.c_str());

			const std::string secondPath_ = "/dslr_images_undistorted";
			depthMapPath_ += secondPath_;
			normalMapPath_ += secondPath_;
			consistencyGraphPath_ += secondPath_;
			undistortPath_ += secondPath_;
			slicPath_ += secondPath_;

			this->findOrCreateDirectory(depthMapPath_.c_str());
			this->findOrCreateDirectory(normalMapPath_.c_str());
			this->findOrCreateDirectory(consistencyGraphPath_.data());
			this->findOrCreateDirectory(undistortPath_.data());
			this->findOrCreateDirectory(slicPath_.c_str());

			std::cout << std::string(78, '=') << std::endl << std::endl;
		}

		//检查目录是否存在，不存在则创建!!!!!目录需要一级一级的创建！！！！！！！！！！！！
		bool PatchMatchController::findOrCreateDirectory(const char *path) const
		{
			//如果存在目录返回为0，否则返回-1
			if (_access(path, 0) == 0)
			{
				std::cout << "Find directory：" << path << std::endl;
				return true;
			}
			else
			{
				//创建成功返回0，否则为-1
				if (_mkdir(path) == 0)
				{
					std::cout << "Create directory success:" << path << std::endl;
					return true;
				}
				else
				{
					std::cout << "No find directory：" << path << "，and can not create it." << std::endl;
					return false;
				}
			}
		}

		void PatchMatchController::RunMyPm()
		{
			this->ReadWorkspace();

			MyPatchMatch::Options MyPmOptions;
			MyPatchMatch MyPm(MyPmOptions, options_, std::move(workspace_));
			MyPm.Run();
		}

		void PatchMatchController::Run()
		{
			this->ReadWorkspace();
			this->ReadProblems();
			this->ReadGpuIndices();

			// 是否利用稀疏点云数据
			if (options_.bUse_sparse_points)
			{
				CalculateInitialNormals();
			}

			thread_pool_.reset(new ThreadPool(gpu_indices_.size()));

			// If geometric consistency is enabled, then photometric output must be
			// computed first for all images without filtering.
			if (options_.geom_consistency)
			{
				auto photometric_options = options_;
				photometric_options.geom_consistency = false;
				photometric_options.filter = false;

				for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx)
				{
					// 是否利用稀疏点云数据
					if (options_.bUse_sparse_points)
						SetSparsePointToGpu(problem_idx);

					// ProcessProblem(photometric_options, problem_idx);
					this->thread_pool_->AddTask(&PatchMatchController::ProcessProblem,
						this,
						photometric_options,
						problem_idx);  // 调用patch_match生成深度图和法向图
				}

				thread_pool_->Wait();
			}

			std::cout << "=> Reading photometirc inputs..." << std::endl;

			// 读取深度和法向图
			workspace_->ReadDepthAndNormalMaps(false);

			for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx)
			{
				//ProcessProblem(options_, problem_idx);
				thread_pool_->AddTask(&PatchMatchController::ProcessProblem, this,
					options_, problem_idx);
			}
			thread_pool_->Wait();

			//////进行上采样,重新在跑一边colmap，只跑一次geom
			//std::cout << "Reading geometric inputs..." << std::endl;
			//workspace_->ReadDepthAndNormalMaps(true);//读取小图像的geometric.txt
			//workspace_->upSamplingMapAndModel();
			//for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx)//geometric,一次迭代(不用几何一致性信息)
			//{
			   // auto geometric_options = options_;
			   // geometric_options.num_geometric_iterations = 1;
			   // geometric_options.bUpsamling = true;
			   // ProcessProblem(geometric_options, problem_idx);
			//}

			////进行上采样,重新再跑一边colmap，只跑一次photo一次geom
			//std::cout << "Reading geometric inputs..." << std::endl;
			//workspace_->ReadDepthAndNormalMaps(true);//读取小图像的geometric.txt
			//workspace_->upSamplingMapAndModel();
			//for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx)//geometric,一次迭代(不用几何一致性信息)
			//{
			   // auto geometric_options = options_;
			   // geometric_options.num_geometric_iterations = 1;
			   // geometric_options.filter = false;
			   // geometric_options.bUpsamling = true;
			   // geometric_options.bUseGeom_consistency = false;
			   // ProcessProblem(geometric_options, problem_idx);
			//}
			//std::cout << "Reading Up-photometirc inputs..." << std::endl;
			//workspace_->ReadDepthAndNormalMapsUpSampling(false);//读取生成的photometricUp.txt
			//for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx)//geometric，一次迭代
			//{
			   // auto geometirc_options = options_;
			   // geometirc_options.num_geometric_iterations = 1;
			   // geometirc_options.bUpsamling = true;
			   // ProcessProblem(geometirc_options, problem_idx);
			//}
		}

		void PatchMatchController::ReadWorkspace()
		{
			std::cout << "=> Reading workspace..." << std::endl;

			Workspace::Options workspace_options;
			workspace_options.max_image_size = options_.max_image_size;
			workspace_options.image_as_rgb = false;
			workspace_options.cache_size = options_.cache_size;
			workspace_options.workspace_path = workspace_path_;
			workspace_options.newPath = newPath_;  // 所有结果所在目录
			workspace_options.workspace_format = workspace_format_;
			workspace_options.undistorte_path = undistortPath_;  // 去扭曲的目录
			workspace_options.slic_path = slicPath_;  // 超像素分割目录
			workspace_options.input_type = "photometric";
			workspace_options.input_type_geom = "geometric";

			// set workspace
			this->workspace_.reset(new Workspace(workspace_options));

			depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();

			//workspace_->runSLIC(workspace_options.slic_path);
			//workspace_->showImgPointToSlicImage(workspace_options.slic_path);
		}

		void PatchMatchController::ReadProblems() 
		{
			problems_.clear();

			const auto model = workspace_->GetModel();

			if (workspace_format_ == "PMVS")
			{
				std::cout << "Importing PMVS options..." << std::endl;
				//ImportPMVSOption(model, workspace_path_, pmvs_option_name_);
			}

			std::cout << "Reading configuration..." << std::endl;

			std::vector<std::map<int, int>> shared_num_points;
			std::vector<std::map<int, float>> triangulation_angles;

			const float min_triangulation_angle_rad =
				(float)DegToRad(options_.min_triangulation_angle);

			for (int i = 0; i < model.m_images.size(); i++)
			{
				PatchMatch::Problem problem;

				problem.ref_img_id = i;

				// Use maximum number of overlapping images as source images. Overlapping
				// will be sorted based on the number of shared points to the reference
				// image and the top ranked images are selected. Note that images are only
				// selected if some points have a sufficient triangulation angle.

				if (shared_num_points.empty())
				{
					shared_num_points = model.ComputeSharedPoints();
				}
				if (triangulation_angles.empty())
				{
					const float kTriangulationAnglePercentile = 75.0f;
					triangulation_angles =
						model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
				}

				//const size_t max_num_src_images =
				//    boost::lexical_cast<int>(src_image_names[1]);
				const size_t max_num_src_images = 20;//默认情况下，最多20个原图像

				const auto& overlapping_images =
					shared_num_points.at(problem.ref_img_id);
				const auto& overlapping_triangulation_angles =
					triangulation_angles.at(problem.ref_img_id);

				if (max_num_src_images >= overlapping_images.size())
				{
					problem.src_img_ids.reserve(overlapping_images.size());
					for (const auto& image : overlapping_images)
					{
						if (overlapping_triangulation_angles.at(image.first) >=
							min_triangulation_angle_rad) {
							problem.src_img_ids.push_back(image.first);
						}
					}
				}
				else
				{
					std::vector<std::pair<int, int>> src_images;
					src_images.reserve(overlapping_images.size());
					for (const auto& image : overlapping_images) 
					{
						if (overlapping_triangulation_angles.at(image.first) >=
							min_triangulation_angle_rad)
						{
							src_images.emplace_back(image.first, image.second);
						}
					}

					const size_t eff_max_num_src_images =
						std::min(src_images.size(), max_num_src_images);

					std::partial_sort(src_images.begin(),
						src_images.begin() + eff_max_num_src_images,
						src_images.end(),
						[](const std::pair<int, int> image1,
							const std::pair<int, int> image2) 
					{
						return image1.second > image2.second;
					});

					problem.src_img_ids.reserve(eff_max_num_src_images);
					for (size_t i = 0; i < eff_max_num_src_images; ++i)
					{
						problem.src_img_ids.push_back(src_images[i].first);
					}
				}

				if (problem.src_img_ids.empty())
				{
					std::printf("WARNING: Ignoring reference image %s, because it has no source images.\n",
						model.GetImageName(problem.ref_img_id));

				}
				else
				{
					problems_.push_back(problem);
				}

			}
		}

		void PatchMatchController::ReadGpuIndices()
		{
			std::cout << "Reading GpuIndices..." << std::endl;
			gpu_indices_.push_back(options_.gpu_index);
			if (gpu_indices_.size() == 1 && gpu_indices_[0] == -1)
			{
				const int num_cuda_devices = GetNumCudaDevices();

				assert(num_cuda_devices > 0);

				gpu_indices_.resize(num_cuda_devices);
				std::iota(gpu_indices_.begin(), gpu_indices_.end(), 0);  // STL中的iota是批量递增赋值vector, 而C中iota是字符串转数字
			}
		}

		// 计算初始法向量
		void PatchMatchController::CalculateInitialNormals()
		{
			std::cout << "Calculate Initial Sparse Normals..." << std::endl;
			const auto &model = workspace_->GetModel();

			sparse_normals_.reserve(model.m_points.size());  // 稀疏点云法向量

			// 处理每个稀疏点云，计算法向量
			for (const auto &point : model.m_points)
			{
				const int trackSize = point.track.size();  // 轨道数量	
				const Eigen::Vector4f pt(point.x, point.y, point.z, 1.0f);  // 三维点坐标

				////开始计算track中视图关于三维点的中间视图
				int id1, id2;//最大夹角的两个图像索引
				float angle_bisect[3];//最大角平分线向量

				//寻找最大夹角，也即最小的cos值
				float min_cos_trang_angle = 1;
				for (int i = 0; i < trackSize - 1; i++)
				{
					const int imageId_i = point.track[i];  // i视图索引
					const float *center_i = model.m_images.at(imageId_i).GetCenter();  // i摄像机中心
					float view_i[3] = { center_i[0] - point.x,
										center_i[1] - point.y,
										center_i[2] - point.z };

					NormVec3(view_i);  // 归一化向量
					for (int j = i + 1; j < trackSize; j++)
					{
						const int imageId_j = point.track[j];  // j视图索引
						const float *center_j = model.m_images.at(imageId_j).GetCenter();  // j摄像机中心
						float view_j[3] = { center_j[0] - point.x, center_j[1] - point.y, center_j[2] - point.z };
						NormVec3(view_j);
						const float cosAngle = DotProduct3(view_i, view_j);

						//const float angle = RadToDeg( acos(cosAngle) );  // 调试用，看角度是多少,弧度转角度

						if (cosAngle < min_cos_trang_angle)
						{
							min_cos_trang_angle = cosAngle;
							angle_bisect[0] = view_i[0] + view_j[0];  // 角平分线向量
							angle_bisect[1] = view_i[1] + view_j[1];
							angle_bisect[2] = view_i[2] + view_j[2];
							id1 = imageId_i;
							id2 = imageId_j;
						}
					}
				}

				// 归一化法向量（这是全局法向量！！！！！）
				NormVec3(angle_bisect);
				sparse_normals_.push_back(cv::Point3f(angle_bisect[0], 
													  angle_bisect[1],
													  angle_bisect[2]));
			}
		}

		//为Gpu准备稀疏点数据
		void PatchMatchController::SetSparsePointToGpu(const size_t problem_idx)
		{
			auto& problem = problems_.at(problem_idx);
			const auto& model = workspace_->GetModel();
			const auto& refImage = model.m_images.at(problem_idx);

			//// Extract 1/fx, -cx/fx, fy, -cy/fy.
			//const float *K = refImage.GetK();
			//const float ref_inv_K[4] = { 1.0f / K[0], -K[2] / K[0], 1.0f / K[4], -K[5] / K[4] };
			const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R =
				Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(refImage.GetR());

			// 准备稀疏点云的深度和法向
			std::vector<float> sparsePoints;
			std::vector<float> sparseNormals;
			std::vector<int> tracks;
			for (int i = 0; i < model.m_points.size(); i++)
			{
				const auto &point = model.m_points[i];
				const int tracksNum = point.track.size();

				// track中没有参考图像
				if (find(point.track.begin(), point.track.end(), problem.ref_img_id) == point.track.end())
					continue;

				// 把全局法向量转化为每幅图像内的法向量
				Eigen::Vector3f normal = R * Eigen::Vector3f(sparse_normals_[i].x, sparse_normals_[i].y, sparse_normals_[i].z);

				// 计算三维点在图像上的投影
				const Eigen::Vector4f pt(point.x, point.y, point.z, 1.0f);  // 三维点坐标
				const Eigen::Vector3f xyz = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(refImage.GetP())*pt;
				const int row = xyz(1) / xyz(2) + 0.5f;
				const int col = xyz(0) / xyz(2) + 0.5f;

				////查看法向量与视野朝向是否相反
				//// Make sure the normal is looking in the same direction as the viewing direction.
				//const Eigen::Vector3f view_ray = { ref_inv_K[0] * col + ref_inv_K[1],
				//	ref_inv_K[2] * row + ref_inv_K[3], 1.0f };
				//if (view_ray.dot(normal) >= 0)
				//{
				//	normal[0] = -normal[0], normal[1] = -normal[1], normal[2] = -normal[2];
				//}

				sparsePoints.push_back(row);
				sparsePoints.push_back(col);
				sparsePoints.push_back(xyz(2));
				sparseNormals.push_back(normal[0]);
				sparseNormals.push_back(normal[1]);
				sparseNormals.push_back(normal[2]);

				tracks.push_back(-1);  // 每两个-1之间记录着一个点的track信息

				//把轨道中的图像写入到选择概率为1
				for (int k = 0; k < tracksNum; k++)
				{
					int srcId = point.track[k];
					if (srcId == problem.ref_img_id)
						continue;
					for (int l = 0; l < problem.src_img_ids.size(); l++)
					{
						if (problem.src_img_ids[l] == srcId)
						{
							tracks.push_back(l);
						}
					}
				}
			}

			tracks.push_back(-1);

			problem.sparsePoints.swap(sparsePoints);
			problem.sparseNormals.swap(sparseNormals);
			problem.tracks.swap(tracks);
		}

		void PatchMatchController::ProcessProblem(const PatchMatch::Options& options,
			const size_t problem_idx) 
		{

			const auto &model = workspace_->GetModel();

			auto& problem = problems_.at(problem_idx);
			//const int gpu_index = gpu_indices_.at(0);  // 只用一个GPU计算
			const int gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());

			assert(gpu_index >= -1);

			const std::string depth_map_path = workspace_->GetDepthMapPath(problem.ref_img_id, options.geom_consistency);
			const std::string normal_map_path = workspace_->GetNormalMapPath(problem.ref_img_id, options.geom_consistency);
			const std::string consistency_graph_path = workspace_->GetConsistencyGaphPath(problem.ref_img_id);

			char temp[50];
			sprintf_s(temp, "Processing view %d / %d", problem_idx + 1, problems_.size());
			const std::string printString = temp;
			PrintHeading1(printString);

			std::vector<Image> images = model.m_images;
			std::vector<DepthMap> depth_maps;
			std::vector<NormalMap> normal_maps;

			problem.images = &images;
			problem.depth_maps = &depth_maps;
			problem.normal_maps = &normal_maps;

			{
				// Collect all used images in current problem.
				std::unordered_set<int> used_image_ids(problem.src_img_ids.begin(),
					problem.src_img_ids.end());
				used_image_ids.insert(problem.ref_img_id);

				// Only access workspace from one thread at a time and only spawn resample
				// threads from one master thread at a time.
			   // std::unique_lock<std::mutex> lock(workspace_mutex_);

				std::cout << "Reading inputs..." << std::endl;
				for (const auto image_id : used_image_ids)
				{
					images.at(image_id).SetBitmap(workspace_->GetBitmap(image_id));
				}

				if (options.geom_consistency)
				{
					//workspace_->ReadDepthAndNormalMaps(true);  // 此时读入的photo一致性生成的map图
					depth_maps = workspace_->GetAllDepthMaps();
					normal_maps = workspace_->GetAllNormalMaps();
				}
			}

			problem.Print();

			auto patch_match_options = options;
			patch_match_options.depth_min = depth_ranges_.at(problem.ref_img_id).first;
			patch_match_options.depth_max = depth_ranges_.at(problem.ref_img_id).second;
			patch_match_options.gpu_index = gpu_index;
			patch_match_options.Print();

			// --------------- call gpu to do patch match(stereo) calculation 
			PatchMatch patch_match(patch_match_options, problem);
			patch_match.Run();  
			// ---------------

			const string tempOut = options.geom_consistency ? "geometric" : "photometric";
			std::cout << std::endl
				<< "Writing " << tempOut << " output for " << model.m_images.at(problem.ref_img_id).GetfileName() << std::endl;

			// 把生成的深度/法向图写到硬盘
			//patch_match.GetDepthMap().Write(depth_map_path);
			//patch_match.GetNormalMap().Write(normal_map_path);
			patch_match.GetDepthMap().WriteBinary(depth_map_path);
			patch_match.GetNormalMap().WriteBinary(normal_map_path);

			if (options.write_consistency_graph)
			{
				patch_match.GetConsistencyGraph().Write(consistency_graph_path);
			}

			// 把深度和法向map用图像表示
			cv::imwrite(depth_map_path + ".jpg", patch_match.GetDepthMap().ToBitmap(2, 98));
			cv::imwrite(normal_map_path + ".jpg", patch_match.GetNormalMap().ToBitmap());

			////把photoconsistency写入workspace的变量里面，供之后的geoConsistency计算使用
			//workspace_->WriteDepthMap(problem.ref_image_id, patch_match.GetDepthMap());
			//workspace_->WriteNormalMap(problem.ref_image_id, patch_match.GetNormalMap());
			//workspace_->WriteConsistencyGraph(problem.ref_image_id, patch_match.GetConsistencyGraph());
			//
			//workspace_->GetDepthMap(problem.ref_image_id).Write(depth_map_path);
			//workspace_->GetNormalMap(problem.ref_image_id).Write(normal_map_path);
			//if (options.write_consistency_graph)
			//{
			   // workspace_->GetConsistencyGraph(problem.ref_image_id).Write(consistency_graph_path);
			//}
			//
			//cv::imwrite(depthMapPath_ + file_name_matImage, workspace_->GetDepthMap(problem.ref_image_id).ToBitmap(2, 98));
			//cv::imwrite(normalMapPath_ + file_name_matImage, workspace_->GetNormalMap(problem.ref_image_id).ToBitmap());

		}

	}  // namespace mvs
}  // namespace colmap
