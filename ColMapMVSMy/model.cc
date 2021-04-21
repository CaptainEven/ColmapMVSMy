#include "model.h"
#include "math.h"
#include "Undistortion.h"
#include "Utils.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>

namespace colmap {
	namespace mvs {


		void Model::Read(const std::string& path,
			const std::string& format,
			const std::string &newPath)
		{
			auto format_lower_case = format;

			//StringToLower(&format_lower_case);
			if (format_lower_case == "colmap")
			{
				//ReadFromCOLMAP(path);
				ReadFromBundlerOfColMap(path, newPath);
			}
			else if (format_lower_case == "pmvs")
			{
				ReadFromPMVS(path, newPath);
			}
			else
			{
				// LOG(FATAL) << "Invalid input format";
				std::printf("Invalid input format\n");
				exit(1);
			}
		}

		//void Model::ReadFromCOLMAP(const std::string& path) {
		//  Reconstruction reconstruction;
		//  reconstruction.Read(JoinPaths(path, "sparse"));
		//
		//  images.reserve(reconstruction.NumRegImages());
		//  std::unordered_map<image_t, size_t> image_id_map;
		//  for (size_t i = 0; i < reconstruction.NumRegImages(); ++i) {
		//    const auto image_id = reconstruction.RegImageIds()[i];
		//    const auto& image = reconstruction.Image(image_id);
		//    const auto& camera = reconstruction.Camera(image.CameraId());
		//
		//    CHECK_EQ(camera.ModelId(), PinholeCameraModel::model_id);
		//
		//    const std::string image_path = JoinPaths(path, "images", image.Name());
		//    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K =
		//        camera.CalibrationMatrix().cast<float>();
		//    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R =
		//        QuaternionToRotationMatrix(image.Qvec()).cast<float>();
		//    const Eigen::Vector3f T = image.Tvec().cast<float>();
		//
		//    images.emplace_back(image_path, camera.Width(), camera.Height(), K.data(),
		//                        R.data(), T.data());
		//    image_id_map.emplace(image_id, i);
		//    image_names_.push_back(image.Name());
		//    image_name_to_id_.emplace(image.Name(), i);
		//  }
		//
		//  points.reserve(reconstruction.NumPoints3D());
		//  for (const auto& point3D : reconstruction.Points3D()) {
		//    Point point;
		//    point.x = point3D.second.X();
		//    point.y = point3D.second.Y();
		//    point.z = point3D.second.Z();
		//    point.track.reserve(point3D.second.Track().Length());
		//    for (const auto& track_el : point3D.second.Track().Elements()) {
		//      point.track.push_back(image_id_map.at(track_el.image_id));
		//    }
		//    points.push_back(point);
		//  }
		//}

		// image_id从1开始
		std::unordered_map<std::string, int> Model::ReadImagesTxt(const std::string& root)
		{
			std::unordered_map<std::string, int> img_name2id;
			std::ifstream in_file(root + "/dslr_calibration_undistorted/images.txt");
			if (in_file)
			{
				std::string line;
				std::vector<std::string> sep_1_items, sep_2_items;
				while (std::getline(in_file, line))
				{
					// 字符串分割, 判断最后一项是否包含.JPG
					sep_1_items.clear();
					sep_2_items.clear();  // 清空, 存放下一行
					StringSplit(line, std::string(" "), sep_1_items);

					const std::string& img_name_item = sep_1_items[sep_1_items.size() - 1];
					int pos = img_name_item.find(std::string(".JPG"));
					if (std::string::npos != pos)
					{
						std::string& img_id_item = sep_1_items[0];
						int img_id = std::stoi(img_id_item);
						if (0 != img_id)
						{
							// 获取文件名
							StringSplit(img_name_item, std::string("/"), sep_2_items);
							if (2 == sep_2_items.size())
							{
								std::string img_name = sep_2_items[1];

								// 添加image_name => image_id
								img_name2id[img_name] = img_id;
							}
						}
					}
				}
			}
			
			return img_name2id;
		}

		void Model::ReadFromPMVS(const std::string& path, const std::string &newPath)
		{
			const std::string bundle_file_path = path + "/bundle.rd.out";

			std::ifstream file(bundle_file_path);
			assert(file.is_open());

			// Header line.
			std::string header;
			std::getline(file, header);

			int num_images, num_points;
			file >> num_images >> num_points;

			m_images.reserve(num_images);
			for (int image_id = 0; image_id < num_images; ++image_id)
			{

				std::string image_name;
				char imgName[20];
				sprintf_s(imgName, "%08d.jpg", image_id);
				image_name = imgName;

				const std::string image_path = path + "/visualize/" + image_name;

				float K[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
				file >> K[0];
				K[4] = K[0];

				cv::Mat bitmap = cv::imread(image_name);
				assert(bitmap.data);
				K[2] = bitmap.cols / 2.0f;
				K[5] = bitmap.rows / 2.0f;

				float k[2];
				file >> k[0] >> k[1];

				float R[9];
				for (size_t i = 0; i < 9; ++i) {
					file >> R[i];
				}
				for (size_t i = 3; i < 9; ++i) {
					R[i] = -R[i];
				}

				float T[3];
				file >> T[0] >> T[1] >> T[2];
				T[1] = -T[1];
				T[2] = -T[2];

				m_images.emplace_back(image_path, image_name, path, newPath, bitmap.cols, bitmap.rows, K, R, T, k);
				m_img_names.push_back(image_name);
				m_img_name_to_id.emplace(image_name, image_id);
			}

			m_points.resize(num_points);
			for (int point_id = 0; point_id < num_points; ++point_id) {
				auto& point = m_points[point_id];

				file >> point.x >> point.y >> point.z;

				int color[3];
				file >> color[0] >> color[1] >> color[2];

				int track_len;
				file >> track_len;
				point.track.resize(track_len);

				for (int i = 0; i < track_len; ++i)
				{
					int feature_idx;
					float imx, imy;
					file >> point.track[i] >> feature_idx >> imx >> imy;
					assert(point.track[i] < m_images.size());
				}
			}
		}

		//newPath是所有结果的生成目录
		void Model::ReadFromBundlerOfColMap(const std::string& path,
			const std::string& newPath)
		{

			const std::string bundle_file_path = path + "/bundler.out";
			const std::string bundler_list_path = path + "/bundler.out.list.txt";

			//// 读取images.txt, 初始化每张图(视角)<=>ID,
			//auto img_name2id = this->ReadImagesTxt(path);
			//this->m_img_name2image_id = img_name2id;
			//this->InitImageID2ImgName();

			std::ifstream file(bundle_file_path);
			std::ifstream file2(bundler_list_path);

			assert(file.is_open() && file2.is_open());

			// Header line.
			std::string header;
			std::getline(file, header);

			int num_images, num_points;
			file >> num_images >> num_points;

			m_images.reserve(num_images);
			m_img_pts.resize(num_images);  // 初始化大小

			// load image info
			for (int img_id = 0; img_id < num_images; ++img_id)
			{

				std::string img_name;
				//int n= img_name.find_last_not_of("\r\n\t");
				//if (n != std::string::npos)
				//	img_name.erase(n + 1, img_name.size() - n);

				//bundler.out.list.txt的图像中可能包含路径 dslr_images_undistorted/DSC_0647.JPG
				std::getline(file2, img_name);

				int i = img_name.size() - 1, num = 0;
				while (i >= 0 && img_name[i] != '/')
				{
					i--;
					num++;
				}

				// 略去图像路径名字中/之前的内容，只保留图像的名字
				const std::string img_name_pure = img_name.substr(i + 1, num);

				// 这个路径对应于ETH3D benchmark中的
				//const std::string image_path = path + "/images/" + image_name;
				const std::string image_path = path + "/" + this->m_src_img_rel_dir + img_name_pure; // image_name

				// 相机矩阵
				float K[9] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
				file >> K[0];
				K[4] = K[0];  // fx == fy?
				//printf("Focus length: %.3f\n", K[0]);

				cv::Mat bitmap = cv::imread(image_path);

				assert(!bitmap.empty());

				K[2] = bitmap.cols / 2.0f;  // cx
				K[5] = bitmap.rows / 2.0f;  // cy

				float k[2];  // 畸变系数?
				file >> k[0] >> k[1];

				// 旋转矩阵
				float R[9];
				for (size_t i = 0; i < 9; ++i)
				{
					file >> R[i];
				}
				for (size_t i = 3; i < 9; ++i)  // why?
				{
					R[i] = -R[i];
				}

				// 平移向量
				float T[3];
				file >> T[0] >> T[1] >> T[2];
				T[1] = -T[1];
				T[2] = -T[2];

				// 构造Image对象
				m_images.emplace_back(image_path,
									  img_name_pure,
									  path,
									  newPath,
									  bitmap.cols,
									  bitmap.rows,
									  K, R, T, k);  // 根据参数初始化临时对象的成员

				//images.at(image_id).SetBitmap(bitmap);  // 把彩色图像读到bitmap中

				this->m_img_names.push_back(img_name_pure);
				this->m_img_name_to_id.emplace(img_name_pure, img_id);  // image_name map to id
			}

			m_points.resize(num_points);

			// load image points info
			for (int pt_id = 0; pt_id < num_points; ++pt_id)
			{
				auto& point = m_points[pt_id];

				// load point 3D coordinate in world coordinate
				file >> point.x >> point.y >> point.z;

				// load point color: not stored
				int color[3];
				file >> color[0] >> color[1] >> color[2];

				int track_len;
				file >> track_len;
				point.track.resize(track_len);

				// load 3D point track info
				for (int i = 0; i < track_len; ++i)
				{
					int feature_idx;
					float im_x, im_y;

					file >> point.track[i] >> feature_idx >> im_x >> im_y;

					assert(point.track[i] < m_images.size());
				}
			}
		}

		void Model::RunUndistortion(const std::string& path)
		{
			// 如果图像信息中有畸变参数，那么就进行去畸变操作
			if (m_images.at(0).Getk()[0] != 0.0f || m_images.at(0).Getk()[1] != 0.0f)
			{
				std::cout << "\t" << "Begin Undistorting..." << std::endl;

				const Undistorter::Options undistorerOptions;
				Undistorter *undistorter = new Undistorter(undistorerOptions, path, m_images);

				undistorter->run();

				delete undistorter;
				undistorter = nullptr;

				std::cout << "\t" << "Done Undistortion" << std::endl;
			}
			else
			{
				std::cout << "\t" << "Don't have distortion params" << std::endl;
			}
		}

		int Model::GetImageId(const std::string& name) const 
		{
			assert(m_img_name_to_id.count(name) >= 0);
			return m_img_name_to_id.at(name);
		}

		std::string Model::GetImageName(const int image_id) const
		{
			assert(image_id >= 0);
			assert(image_id < m_img_names.size());
			//return image_names_.at(image_id);
			return m_images.at(image_id).GetfileName();
		}

		std::vector<std::vector<int>> Model::GetMaxOverlappingImgs(
			const size_t num_images,
			const double min_triangulation_angle) const
		{
			const float min_triangulation_angle_rad = DegToRad(min_triangulation_angle);

			// 统计图像之间特征点的匹配数
			const auto shared_num_points = this->ComputeSharedPoints();

			// 对于每个overlapping的图片, 取三角化角度从小到大0.75处的值
			const float kTriangulationAnglePercentile = 75.0f;
			const auto triangulation_angles =
				this->ComputeTriangulationAngles(kTriangulationAnglePercentile);

			std::vector<std::vector<int>> overlapping_imgs(m_images.size());

			for (size_t img_id = 0; img_id < m_images.size(); ++img_id)
			{
				const auto& shared_imgs = shared_num_points.at(img_id);
				const auto& overlapping_triangulation_angles =
					triangulation_angles.at(img_id);

				std::vector<std::pair<int, int>> ordered_images;

				ordered_images.reserve(shared_imgs.size());
				for (const auto& img_item : shared_imgs)
				{
					// 选择满足三角化角度要求的匹配
					if (overlapping_triangulation_angles.at(img_item.first) >=
						min_triangulation_angle_rad)
					{
						ordered_images.emplace_back(img_item.first, img_item.second);
					}
				}

				const size_t eff_num_imgs = std::min(ordered_images.size(), num_images);
				if (eff_num_imgs < shared_imgs.size()) 
				{
					std::partial_sort(ordered_images.begin(),
						ordered_images.begin() + eff_num_imgs,
						ordered_images.end(),
						[](const std::pair<int, int> image_1,
							const std::pair<int, int> image_2) 
					{
						return image_1.second > image_2.second;
					});  // 局部排序: 按照匹配数从大到小排序
				}
				else 
				{
					std::sort(ordered_images.begin(), ordered_images.end(),
						[](const std::pair<int, int> image_1,
							const std::pair<int, int> image_2) 
					{
						return image_1.second > image_2.second;
					});  // 全部排序：按照匹配数从大到小排序
				}

				// 统计图像的overlapping
				overlapping_imgs[img_id].reserve(eff_num_imgs);
				for (size_t i = 0; i < eff_num_imgs; ++i)
				{
					overlapping_imgs[img_id].push_back(ordered_images[i].first);
				}
			}

			return overlapping_imgs;
		}

		std::vector<std::pair<float, float>> Model::ComputeDepthRanges() const
		{
			std::vector<std::vector<float>> depths(m_images.size());

			// traverse each 3D point in world coordinate
			for (const auto& point : m_points)
			{
				// 3D空间点世界坐标
				const Eigen::Vector3f X(point.x, point.y, point.z);

				// traverse each track of image
				for (const auto& img_id : point.track) 
				{
					const auto& image = m_images.at(img_id);

					// 计算3D空间点的深度
					const float depth =
						Eigen::Map<const Eigen::Vector3f>(&image.GetR()[6]).dot(X) 
						+ image.GetT()[2];

					// 仅统计非零深度
					if (depth > 0.0f) 
					{
						depths[img_id].push_back(depth);
					}
				}
			}

			// depth range of each image
			std::vector<std::pair<float, float>> depth_ranges(depths.size());
			for (size_t img_id = 0; img_id < depth_ranges.size(); ++img_id)
			{
				auto& depth_range = depth_ranges[img_id];
				auto& image_depths = depths[img_id];

				if (image_depths.empty())
				{
					depth_range.first = -1.0f;
					depth_range.second = -1.0f;
					continue;
				}

				std::sort(image_depths.begin(), image_depths.end());

				const float kMinPercentile = 0.01f;
				const float kMaxPercentile = 0.99f;
				depth_range.first = image_depths[image_depths.size() * kMinPercentile];
				depth_range.second = image_depths[image_depths.size() * kMaxPercentile];

				const float kStretchRatio = 0.25f;  // empiric value?
				depth_range.first *= (1.0f - kStretchRatio);
				depth_range.second *= (1.0f + kStretchRatio);
			}

			return depth_ranges;
		}

		std::vector<std::map<int, int>> Model::ComputeSharedPoints() const 
		{
			std::vector<std::map<int, int>> shared_points(m_images.size());
			for (const auto& point : m_points) 
			{
				for (size_t i = 0; i < point.track.size(); ++i) 
				{
					const int& img_id_1 = point.track[i];
					for (size_t j = 0; j < i; ++j)
					{
						const int& img_id_2 = point.track[j];
						if (img_id_1 != img_id_2)
						{
							shared_points.at(img_id_1)[img_id_2] += 1;
							shared_points.at(img_id_2)[img_id_1] += 1;
						}
					}
				}
			}

			return shared_points;
		}

		std::vector<std::map<int, float>> Model::ComputeTriangulationAngles(
			const float percentile) const 
		{
			// 每张图都是一个视角, 根据相机外参计算每个视角的相机中心坐标
			std::vector<Eigen::Vector3d> proj_centers(this->m_images.size());
			for (size_t img_id = 0; img_id < m_images.size(); ++img_id)
			{
				const auto& image = this->m_images[img_id];

				// 计算投影中心(相机中心在世界坐标系中的坐标)
				Eigen::Vector3f C;
				ComputeProjectionCenter(image.GetR(), image.GetT(), C.data());
				proj_centers[img_id] = C.cast<double>();
			}

			// 每张图对应一个map
			std::vector<std::map<int, std::vector<float>>> all_triangulation_angles(
				m_images.size());

			// 计算所有对应点的三角化角度
			for (const auto& pt_3d : m_points)
			{
				for (size_t i = 0; i < pt_3d.track.size(); ++i) 
				{
					const int& img_id_1 = pt_3d.track[i];
					for (size_t j = 0; j < i; ++j) 
					{
						const int& img_id_2 = pt_3d.track[j];
						if (img_id_1 != img_id_2)
						{
							const double angle = CalculateTriangulationAngle(
								proj_centers.at(img_id_1),
								proj_centers.at(img_id_2),
								Eigen::Vector3d(pt_3d.x, pt_3d.y, pt_3d.z));

							all_triangulation_angles.at(img_id_1)[img_id_2].push_back(angle);
							all_triangulation_angles.at(img_id_2)[img_id_1].push_back(angle);
						}
					}
				}
			}

			// 计算每张图按照从小到大的顺序排序，取percentile处的值作为角度代表
			std::vector<std::map<int, float>> triangulation_angles(m_images.size());
			for (size_t img_id = 0; 
				img_id < all_triangulation_angles.size();
				++img_id) 
			{
				// 存在对应点的图像即overlapping images
				const auto& overlapping_imgs = all_triangulation_angles[img_id];
				for (const auto& img_item : overlapping_imgs) 
				{// emplace()可避免复制和移动操作
					triangulation_angles[img_id].emplace(
						img_item.first, Percentile(img_item.second, percentile));
				}
			}

			return triangulation_angles;
		}

		void Model::ProjectToImage()
		{
			// traverse each point
			for (const auto& point : m_points)
			{
				//const cv::Mat cvPoint=(cv::Mat_<float>(4,1)<< point.x, point.y, point.z, 1.0f);
				const Eigen::Vector4f pt3D_H(point.x, point.y, point.z, 1.0f);  // 齐次坐标

				// traverse each track of image
				for (size_t i = 0; i < point.track.size(); i++)
				{
					const int img_id = point.track[i];
					const auto& image = m_images.at(img_id);

					// project 3D point in world coodinate to pixel coordinate: PX=λx
					const Eigen::Vector3f xyz = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(image.GetP()) * pt3D_H;

					// pixel coordinate
					const cv::Point3f cv_img_pt(xyz(0), xyz(1), xyz(2));
					this->m_img_pts.at(img_id).push_back(cv_img_pt);

					//// ----------- 测试2种方式计算的Depth是否相同: 相同
					//const float& depth_1 = xyz(2);
					//printf("Depth1: %.3f\n", depth_1);

					//const Eigen::Vector3f X(point.x, point.y, point.z);
					//float depth_2 = Eigen::Map<const Eigen::Vector3f>(&image.GetR()[6]).dot(X)
					//	+ image.GetT()[2];
					//printf("Depth2: %.3f\n", depth_2);
					//
					//// ----------- 测试2D->3D是否正确
					//cv::Point2f pt2D(xyz(0) / xyz(2), xyz(1) / xyz(2));
					//cv::Point3f pt3D_CV = BackProjTo3D(image.GetK(), image.GetR(), image.GetT(), xyz(2), pt2D);
					//if (abs(pt3D_CV.x - point.x) > 1e-3
					//	|| abs(pt3D_CV.y - point.y) > 1e-3                                                      
					//	|| abs(pt3D_CV.z - point.z) > 1e-3)
					//{
					//	printf("Test wrong!\n");
					//}
					//else
					//	printf("Test correct!\n");

					// 调试用信息，用以检测投影点是否和bundler中数据一样
					//cv::Point2f temp(xyz(0) / xyz(2) - image.GetWidth() / 2,
					//	image.GetHeight() / 2 - xyz(1) / xyz(2));
				}
			}
		}

		//cv::Point3f Model::BackProjTo3D(const float* K_arr,
		//	const float* R_arr,
		//	const float* T_arr,
		//	const float& depth,
		//	const cv::Point2f pt2D)
		//{
		//	const Eigen::Vector3f pt2D_H(depth * pt2D.x, depth * pt2D.y, depth);  // λX_2D

		//	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr);
		//	const Eigen::Vector3f X_cam = K.inverse() * pt2D_H;
		//	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R_arr);
		//	const Eigen::Vector3f pt3D = R.inverse() \
		//		* (X_cam - Eigen::Map <const Eigen::Vector3f>(T_arr));

		//	cv::Point3f pt3D_CV(pt3D(0), pt3D(1), pt3D(2));
		//	return pt3D_CV;
		//}
		
		// 3D平面方程写成Ax=B的形式: aX + bY + Z + c = 0(aX + bY + c = -Z)
		cv::Mat Model::PlaneFitOLS(const std::vector<cv::Point3f>& Pts3D)
		{// 输出结果向量X: a, b, c
			assert(Pts3D.size() >= 3);

			cv::Mat A((int)Pts3D.size(), 3, CV_32F);
			cv::Mat B((int)Pts3D.size(), 1, CV_32F);

			// 系数矩阵A和结果向量b初始化
			for (size_t i = 0; i < Pts3D.size(); ++i)
			{// 
				A.at<float>((int)i, 0) = Pts3D[i].x;
				A.at<float>((int)i, 1) = Pts3D[i].y;
				A.at<float>((int)i, 2) = 1.0f;

				B.at<float>((int)i, 0) = -Pts3D[i].z;
			}

			// 解线性方程组: x = (A' * A)^-1 * A' * b
			cv::Mat X = -((A.t() * A).inv() * A.t() * B);  // 3×1

			// 平面方程法向量n归一化
			//std::cout << X << std::endl;
			//std::cout << X.at<float>(0, 0) << std::endl;
			//std::cout << X.at<float>(1, 0) << std::endl;
			//std::cout << X.at<float>(2, 0) << std::endl;

			const float& a = X.at<float>(0, 0);
			const float& b = X.at<float>(1, 0);
			const float& c = X.at<float>(2, 0);
			const float denom = std::sqrtf(a * a + b * b + 1.0f);

			X.at<float>(0, 0) /= denom;  // a
			X.at<float>(1, 0) /= denom;  // b
			X.at<float>(2, 0) /= denom;  // c

			//std::cout << X << std::endl;

			cv::Mat plane(4, 1, CV_32F);  // 4×1
			plane.at<float>(0, 0) = X.at<float>(0, 0);
			plane.at<float>(1, 0) = X.at<float>(1, 0);
			plane.at<float>(2, 0) = 1.0f / denom;
			plane.at<float>(3, 0) = X.at<float>(2, 0);

			//float norm = plane.at<float>(0, 0) * plane.at<float>(0, 0) \
			//	+ plane.at<float>(1, 0) * plane.at<float>(1, 0) \
			//	+ plane.at<float>(2, 0) * plane.at<float>(2, 0);
			//std::cout << "norm: " << norm << std::endl;
			//std::cout << "plane: " << plane << std::endl;

			return plane;
		}

		int Model::PlaneFitBy3Pts(const cv::Point3f* pts, float* plane_arr)
		{
			const float& x1 = pts[0].x;
			const float& y1 = pts[0].y;
			const float& z1 = pts[0].z;

			const float& x2 = pts[1].x;
			const float& y2 = pts[1].y;
			const float& z2 = pts[1].z;

			const float& x3 = pts[2].x;
			const float& y3 = pts[2].y;
			const float& z3 = pts[2].z;

			float A = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1);
			float B = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1);
			float C = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);

			const float DENOM = std::sqrtf(A * A + B * B + C * C);

			// 判断三点是否共线
			if (DENOM < 1e-12)
			{
				printf("[Warning]: 3 Points may near colinear\n");
				return -1;
			}

			A /= DENOM;
			B /= DENOM;
			C /= DENOM;
			float D = -(A*x1 + B*y1 + C*z1);

			plane_arr[0] = A;
			plane_arr[1] = B;
			plane_arr[2] = C;
			plane_arr[3] = D;

			//// ----- verify the 3 points are on the plane
			//float Norm = std::sqrtf(A * A + B * B + C * C);
			////printf("Norm: %.5f\n", Norm);

			//cv::Mat plane_mat(4, 1, CV_32F);
			//cv::Mat point_mat(4, 1, CV_32F);
			//plane_mat.at<float>(0, 0) = A;
			//plane_mat.at<float>(1, 0) = B;
			//plane_mat.at<float>(2, 0) = C;
			//plane_mat.at<float>(3, 0) = D;

			//for (int i = 0; i < 3; i++)
			//{
			//	point_mat.at<float>(0, 0) = x1;
			//	point_mat.at<float>(1, 0) = y1;
			//	point_mat.at<float>(2, 0) = z1;
			//	point_mat.at<float>(3, 0) = 1.0f;

			//	float res = fabs(plane_mat.dot(point_mat));
			//	if (res > 1e-6)
			//		printf("Res: %.8f\n", res);

			//}

			return 0;
		}

		float Model::MeanDistOfPtToPlane(const std::vector<cv::Point3f>& Pts3D, const cv::Mat& plane)
		{
			float dist_sum = 0.0f;
			for (auto pt3D : Pts3D)
			{
				cv::Mat pt3d_H(4, 1, CV_32F);
				pt3d_H.at<float>(0, 0) = pt3D.x;
				pt3d_H.at<float>(1, 0) = pt3D.y;
				pt3d_H.at<float>(2, 0) = pt3D.z;
				pt3d_H.at<float>(3, 0) = 1.0f;

				float denom = sqrtf(plane.at<float>(0, 0) * plane.at<float>(0, 0) + \
					plane.at<float>(1, 0) * plane.at<float>(1, 0) + \
					plane.at<float>(2, 0) * plane.at<float>(2, 0));
				dist_sum += fabs(plane.dot(pt3d_H)) / denom;
			}

			return dist_sum / float(Pts3D.size());
		}

		void Model::SetSrcImgRelDir(const std::string& img_dir)
		{
			this->m_src_img_rel_dir = img_dir;
		}

	}  // namespace mvs
}  // namespace colmap
