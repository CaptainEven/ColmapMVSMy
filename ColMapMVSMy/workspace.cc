#include "workspace.h"
#include "RansacRunner.h"
#include "MyPatchMatch.h"
#include "JointBilateralFilter.h"
#include "upsampler.h"//双边引导上采样
#include "BilateralGrid.h"//双边网格
#include "FastBilateralSolverMe.h"
#include "guidedfilter.h"
#include "consistency_graph.h"

//#include "./peac/AHCPlaneFitter.hpp"

#include <io.h>
#include <ctime>
#include <Eigen/Dense>
#include <opencv2/ximgproc.hpp>


//#include "BilateralTextureFilter.h";//细节和结构增强
//#include <Eigen/core>

#include "Utils.h"

// 是否开启MRR GPU优化
#define MRF_GPU

#ifdef MRF_GPU
#define NUM_MRF_ITER 1
#else
#define NUM_ITER 1
#endif // MRF_GPU

// 是否绘制mask
//#define DRAW_MASK

// 是否输出log
//#define LOGGING

extern int MRFGPU(const cv::Mat& labels,  // 每个pt2d点的label
	const std::vector<int>& SPLabels,  // 候选的的labels
	const std::vector<float>& SPLabelDepths,  // 按照label排列点的深度值
	const std::vector<int>& pts2d_size, // 按照label排列点的pt2d点个数
	const std::vector<cv::Point2f>& NoDepthPts2d,  // 无深度值pt2d点
	const std::vector<float>& sp_label_plane_arrs,  // 按照label排列点的平面方程
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet);

extern int MRFGPU2(const cv::Mat& labels,  // 每个pt2d点的label
	const std::vector<int>& SPLabels,  // 候选的的labels
	const std::vector<float>& SPLabelDepths,  // 按照label排列点的深度值
	const std::vector<int>& pts2d_size, // 按照label排列点的pt2d点个数
	const std::vector<cv::Point2f>& NoDepthPts2d,  // 无深度值pt2d点
	const std::vector<int>& NoDepthPts2dLabelIdx,  // 每个pt2d点的label idx
	const std::vector<float>& sp_label_plane_arrs,  // 按照label排列点的平面方程
	const std::vector<int>& sp_label_neighs_idx,  // 每个label_idx对应的sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet);

extern int MRFGPU3(
	const cv::Mat& depth_mat,
	const cv::Mat& labels,  // 每个pt2d点的label
	const std::vector<int>& SPLabels,  // 候选的的labels
	const std::vector<cv::Point2f>& NoDepthPts2d,  // 无深度值pt2d点
	const std::vector<int>& NoDepthPts2dLabelIdx,  // 每个pt2d点的label idx
	const std::vector<float>& sp_label_plane_arrs,  // 按照label排列点的平面方程
	const std::vector<int>& sp_label_neighs_idx,  // 每个label_idx对应的sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet);

extern int MRFGPU4(const cv::Mat& depth_mat,
	const cv::Mat& labels,  // 每个pt2d点的label
	const std::vector<int>& SPLabels,  // 候选的的labels
	const std::vector<float>& SPLabelDepths,  // 按照label排列点的深度值
	const std::vector<int>& pts2d_size, // 按照label排列点的pt2d点个数
	const std::vector<cv::Point2f>& NoDepthPts2d,  // 无深度值pt2d点
	const std::vector<int>& NoDepthPts2dLabelIdx,  // 每个pt2d点的label idx
	const std::vector<float>& sp_label_plane_arrs,  // 按照label排列点的平面方程
	const std::vector<int>& sp_label_neighs_idx,  // 每个label_idx对应的sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // 每个label_idx对应的neighbor数量
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet);

extern int BlockMRF(const cv::Mat& depth_mat,
	const int blk_size,  // block size
	const float* K_inv_arr,  // 相机内参矩阵的逆
	const std::vector<int>& blks_pt_cnt,  // 记录所有block的pt2d数量
	const std::vector<cv::Point2f>& blks_pts2d,  // 记录所有block的pt2d点坐标
	const std::vector<int>& blks_pt_cnt_has,  // 记录待处理block的有深度值点个数
	const std::vector<int>& blks_pt_cnt_non,  // 记录待处理block的无深度值点个数
	const std::vector<std::vector<float>>& plane_equa_arr,  // 记录作为label的blk_id对应的平面方程
	const std::vector<int>& label_blk_ids,  // 记录有足够多深度值点的blk_id: 可当作label
	const std::vector<int>& process_blk_ids,  // 记录待(MRF)处理的blk_id
	const std::vector<float>& process_blks_depths_has,  // 记录待处理block的有深度值(组成的数组)
	const std::vector<int>& process_blks_pts2d_has_num,  // 记录待处理block的有深度值点个数
	const std::vector<int>& process_blks_pts2d_non_num,  // 记录待处理block无深度点个数
	const std::vector<cv::Point2f>& process_blks_pt2d_non,  // 记录待处理block的无深度值点坐标
	const std::vector<int>& all_blks_labels,  // 记录每个block对应的label(blk_id): 初始label数组
	const int num_x, const int num_y,  // y方向block数量, x方向block数量
	const int radius, const float beta, const float depth_range,
	std::vector<int>& labels_ret);

extern int JBUSPGPU(const cv::Mat& src,
	const cv::Mat& depth_mat,
	const std::vector<cv::Point2f>& pts2d_has_no_depth_jbu,  // 待处理的pt2d点
	const std::vector<int>& sp_labels_idx_jbu,  // 每个待处理的pt2d点对应的label_idx
	const std::vector<cv::Point2f>& pts2d_has_depth_jbu,  // 用来计算JBU有深度值的pt2d点
	const std::vector<int>& sp_has_depth_pt2ds_num,  // 每个label_idx对应的有深度值pt2d点数
	const std::vector<float>& sigmas_s_jbu, // // 每个label_idx对应的sigma_s
	std::vector<float>& depths_ret);


namespace colmap {
	namespace mvs {

		//如果 * elem1 应该排在 * elem2 前面，则函数返回值是负整数（任何负整数都行）。
		//如果 * elem1 和* elem2 哪个排在前面都行，那么函数返回0
		//如果 * elem1 应该排在 * elem2 后面，则函数返回值是正整数（任何正整数都行）。
		bool pairIfAscend(pair<float, int> &a, pair<float, int> &b)
		{
			//if (a.second >= b.second)//如果a要排在b后面，返回正整数
			//{
			//	return 1;
			//}
			//else
			//{
			//	return -1;
			//}

			return a.first < b.first;
		}

		//降序，由大到小
		bool pairIfDescend(pair<float, int> &a, pair<float, int> &b)
		{
			return a.first > b.first;
		}

		template <typename T>
		float Median(std::vector<T>* elems)
		{
			assert(!elems->empty());
			const size_t mid_idx = elems->size() / 2;
			std::nth_element(elems->begin(), elems->begin() + mid_idx, elems->end());
			if (elems->size() % 2 == 0)
			{
				const float mid_element1 = static_cast<float>((*elems)[mid_idx]);
				const float mid_element2 = static_cast<float>(
					*std::max_element(elems->begin(), elems->begin() + mid_idx));
				return (mid_element1 + mid_element2) / 2.0f;
			}
			else
			{
				return static_cast<float>((*elems)[mid_idx]);
			}
		}

		// 构造函数: 初始化
		Workspace::Workspace(const Options& options)
			: options_(options)
		{
			// 从bundler数据中读取稀疏点云信息
			// StringToLower(&options_.input_type);

			// 设置原图相对路径
			this->m_model.SetSrcImgRelDir(std::string(options.src_img_dir));

			// 读取工作空间(SFM稀疏重建结果)
			m_model.Read(options_.workspace_path,
				options_.workspace_format,
				options_.newPath);

			if (options_.max_image_size != -1)
			{
				for (auto& image : m_model.m_images)
				{
					// 先缩减图像尺寸
					image.Downsize(options_.max_image_size, options_.max_image_size);
				}
			}
			if (options_.bDown_sampling)  // 是否进行降采样处理
			{
				for (auto& image : m_model.m_images)
				{
					// 图像和摄像机参数缩放比例
					image.Rescale(options_.fDown_scale);
				}
			}

			// 对输入图像进行去畸变处理
			//model_.RunUndistortion(options_.undistorte_path);

			// 计算bundler中三维点投影到二维点
			m_model.ProjectToImage();

			// 计算深度范围
			depth_ranges_ = m_model.ComputeDepthRanges();

			// 初始状态，都没有申请各种map数据
			hasReadMapsPhoto_ = false;
			hasReadMapsGeom_ = false;
			hasBitMaps_.resize(m_model.m_images.size(), false);
			bitMaps_.resize(m_model.m_images.size());

			// 初始化深度图容器大小
			this->m_depth_maps.resize(m_model.m_images.size());
			this->m_normal_maps.resize(m_model.m_images.size());
		}

		void Workspace::runSLIC(const std::string &path)
		{
			std::cout << "\t" << "=> Begin SLIC..." << std::endl;

			slicLabels_.resize(m_model.m_images.size());

			int k = 1500;
			int m = 10;
			float ss = 15;  // 超像素的步长
			for (int i = 0; i < m_model.m_images.size(); i++)
			{
				SLIC *slic = new SLIC(ss, m, m_model.m_images[i].GetPath(), path);
				slic->run(i).copyTo(slicLabels_[i]);//把超像素分割出来的label图拷贝到slicLabels_
				delete slic;
			}

			std::cout << "\t" << "Done SLIC" << std::endl;
		}

		void Workspace::showImgPointToSlicImage(const std::string &path)
		{
			//for (const auto &point : model_.points)
			//{
			//	//const cv::Mat cvPoint=(cv::Mat_<float>(4,1)<< point.x, point.y, point.z, 1.0f);
			//	const Eigen::Vector4f pt(point.x, point.y, point.z, 1.0f);
			//	for (size_t i = 0; i < point.track.size(); i++)
			//	{
			//		const int img_id = point.track[i];
			//		const auto &image = model_.images.at(img_id);
			//		const Eigen::Vector3f xyz = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>
			//			(image.GetP())*pt;
			//		const cv::Point3f cvImgPoint(xyz(0), xyz(1), xyz(2));
			//		model_.imagePoints.at(img_id).push_back(cvImgPoint);
			//		//调试用信息，用以检测投影点是否和bundler中数据一样
			//		float x = xyz(0) / xyz(2) - image.GetWidth() / 2;
			//		float y = -xyz(1) / xyz(2) + image.GetHeight() / 2;
			//
			//	}
			//}

			///// 将三维点投影到图像上面
			for (int img_id = 0; img_id < m_model.m_images.size(); img_id++)
			{
				const auto &image = m_model.m_images.at(img_id);
				cv::Mat img = cv::imread(image.GetPath());

				// 减小图像饱和度，是的三维投影点在图像上面看的更清楚
				cv::Mat whiteImg(img.size(), img.type(), cv::Scalar::all(255));
				cv::addWeighted(img, 0.6, whiteImg, 0.4, 0.0, img);

				// 开始在图像点出画圆
				for (const auto &imgPt : m_model.m_img_pts.at(img_id))
				{
					const cv::Point2d pp(imgPt.x / imgPt.z, imgPt.y / imgPt.z);
					cv::circle(img, pp, 1, cv::Scalar(0, 0, 255), -1, 8);
				}

				char filename[20];
				sprintf_s(filename, "SparsePoject%d.jpg", img_id);
				const string filePath = path + filename;
				imwrite(filePath, img);
			}

			// 将投影点画到超像素图像上面
			for (int img_id = 0; img_id < m_model.m_images.size(); img_id++)
			{
				const auto &image = m_model.m_images.at(img_id);
				const auto &label = slicLabels_.at(img_id);
				cv::Mat img = cv::imread(image.GetPath());

				// 减少图像饱和度，使得二维点在图像上看的清晰
				cv::Mat whiteImg(img.rows, img.cols, CV_8UC3, cv::Scalar::all(255));
				cv::addWeighted(img, 0.8, whiteImg, 0.2, 0.0, img);

				//// 画轮廓线
				int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
				int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

				cv::Mat istaken(img.rows, img.cols, CV_8UC1, cv::Scalar::all(false));
				for (int i = 0; i < img.rows; i++)
				{
					for (int j = 0; j < img.cols; j++)
					{
						int np = 0;
						for (int k = 0; k < 8; k++)
						{
							int x = j + dx8[k];
							int y = i + dy8[k];

							if (x > -1 && x < img.cols && y > -1 && y < img.rows)
							{
								if (istaken.at<bool>(y, x) == false)
								{
									if (label.at<int>(i, j) != label.at<int>(y, x))
									{
										np++;
									}
								}
							}
						}
						if (np > 1)  // 增大可减细超像素分割线
						{
							img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);//白线
							//img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);//黑线
							istaken.at<bool>(i, j) = true;
						}
					}
				}

				//// 开始在图像点出画圆
				for (const auto &imgPt : m_model.m_img_pts.at(img_id))
				{
					const cv::Point2d pp(imgPt.x / imgPt.z, imgPt.y / imgPt.z);
					cv::circle(img, pp, 1, cv::Scalar(0, 0, 255), -1, 8);
				}

				char filename[20];
				sprintf_s(filename, "slicPoject%d.jpg", img_id);
				const string filePath = path + filename;
				imwrite(filePath, img);
			}
		}

		////----------------------------------------------------------------------////
		//对深度和法向量图进行上采样
		void Workspace::UpSampleMapAndModel()
		{
		}

		const Model& Workspace::GetModel() const { return m_model; }

		const cv::Mat& Workspace::GetBitmap(const int img_id)
		{
			if (!hasBitMaps_.at(img_id))
			{
				std::string img_path = m_model.m_images.at(img_id).GetPath();

				// @even
				StringReplace(img_path, std::string("MyMvs"),
					std::string(this->m_model.m_src_img_rel_dir));

				//printf("img_path: %s", img_path.c_str());
				cv::Mat bitmap = imread(img_path);
				if (bitmap.empty())
				{
					printf("[Err]: empty bitmap!\n");
					return cv::Mat();
				}

				if (!options_.image_as_rgb)  // 如果不需要rgb图像，那么转化为灰度图像
				{
					cv::cvtColor(bitmap, bitmap, CV_BGR2GRAY);
				}

				if (options_.bDetailEnhance)  // 是否细节增强
				{
					//DentailEnhance(bitmap, bitmap);
					//detailEnhance(bitmap, bitmap);//opencv
					const string &tempFileName = "/" + m_model.m_images.at(img_id).GetfileName() + ".jpg";
					imwrite(options_.workspace_path + options_.newPath + tempFileName, bitmap);
				}
				else if (options_.bStructureEnhance)//是否结构增强
				{
					//MultiscaleStructureEnhance(bitmap, bitmap);
					const string &tempFileName = "/" + m_model.m_images.at(img_id).GetfileName() + ".jpg";
					imwrite(options_.workspace_path + options_.newPath + tempFileName, bitmap);
				}

				bitMaps_.at(img_id) = bitmap;
				hasBitMaps_.at(img_id) = true;
			}
			return bitMaps_.at(img_id);
		}

		// 读入photometirc或者geometric深度和法向map图
		const void Workspace::ReadDepthAndNormalMaps(const bool isGeometric)
		{
			// 如果要求Geom并且已经读入，或者要求photo并且已经读入，则返回
			if (isGeometric && hasReadMapsGeom_)
			{
				std::cout << "**Have Read geometric depth/normalMaps Before**" << std::endl;
				return;
			}
			else if (!isGeometric && hasReadMapsPhoto_)
			{
				std::cout << "**Have Read photometric depth/normalMaps Before**" << std::endl;
				return;
			}

			// 读取所有深度和法向map
			for (int img_id = 0; img_id < m_model.m_images.size(); img_id++)
			{
				//DepthMap depth_map(model_.images.at(image_id).GetWidth(), model_.images.at(image_id).GetHeight(),
				//	depth_ranges_.at(image_id).first, depth_ranges_.at(image_id).second);

				// 初始化depth map
				DepthMap depth_map(depth_ranges_.at(img_id).first,
					depth_ranges_.at(img_id).second);

				string& depth_map_path = this->GetDepthMapPath(img_id, isGeometric);

				depth_map.ReadBinary(depth_map_path);

				// 因为图像尺寸可能相差+-1, 因此就简单的把图像尺寸修改一下
				const size_t mapWidth = depth_map.GetWidth();
				const size_t mapHeigh = depth_map.GetHeight();
				const size_t imgWidth = m_model.m_images.at(img_id).GetWidth();
				const size_t imgHeigh = m_model.m_images[img_id].GetHeight();

				assert(mapWidth == imgWidth && mapHeigh == imgHeigh);

				//if (mapWidth!=imgWidth || mapHeigh!=imgHeigh)
				//{
				//	model_.images.at(image_id).SetWidth(mapWidth);
				//	model_.images.at(image_id).SetHeight(mapHeigh);
				//	model_.images.at(image_id).ResizeBitMap(); 
				//	model_.images.at(image_id).WriteBitMap();
				//}
				m_depth_maps.at(img_id) = depth_map;

				NormalMap normal_map;

				string& normal_map_path = GetNormalMapPath(img_id, isGeometric);

				normal_map.ReadBinary(normal_map_path);
				m_normal_maps.at(img_id) = normal_map;

				//depth_map.WriteBinary(GetDepthMapPath(image_id, isGeometric));
				//normal_map.WriteBinary(GetNormalMapPath(image_id, isGeometric));
			}
			if (isGeometric)
			{
				hasReadMapsGeom_ = true;
				hasReadMapsPhoto_ = false;
				std::cout << "**Read geometric depth/normalMap to workspace Done**" << std::endl;
			}
			else
			{
				hasReadMapsPhoto_ = true;
				hasReadMapsGeom_ = false;
				std::cout << "**Read photometric depth/normalMap to workspace Done**" << std::endl;
			}
		}

		const DepthMap& Workspace::GetDepthMap(const int image_id) const
		{
			assert(hasReadMapsPhoto_ || hasReadMapsGeom_);
			return this->m_depth_maps.at(image_id);
		}

		const NormalMap& Workspace::GetNormalMap(const int image_id) const
		{
			assert(hasReadMapsPhoto_ || hasReadMapsGeom_);
			return m_normal_maps.at(image_id);
		}

		const std::vector<DepthMap>& Workspace::GetAllDepthMaps() const
		{
			assert(hasReadMapsPhoto_ || hasReadMapsGeom_);
			return m_depth_maps;
		}

		const std::vector<NormalMap>& Workspace::GetAllNormalMaps() const
		{
			assert(hasReadMapsPhoto_ || hasReadMapsGeom_);
			return m_normal_maps;
		}


		void Workspace::WriteDepthMap(const int image_id, const DepthMap &depthmap)
		{
			m_depth_maps.at(image_id) = depthmap;
		}

		void Workspace::WriteNormalMap(const int image_id, const NormalMap &normalmap)
		{
			m_normal_maps.at(image_id) = normalmap;
		}

		const ConsistencyGraph& Workspace::GetConsistencyGraph(const int image_id) const
		{
			ConsistencyGraph consistecyGraph;
			consistecyGraph.Read(GetConsistencyGaphPath(image_id));
			return consistecyGraph;
		}

		std::string Workspace::GetBitmapPath(const int img_id) const
		{
			return m_model.m_images.at(img_id).GetPath();
		}

		std::string Workspace::GetDepthMapPath(const int img_id, const bool isGeom) const
		{
			return m_model.m_images.at(img_id).GetDepthMapPath() + GetFileName(img_id, isGeom);
		}

		std::string Workspace::GetNormalMapPath(const int img_id, const bool isGeom) const
		{
			return m_model.m_images.at(img_id).GetNormalMapPath() + GetFileName(img_id, isGeom);
		}

		std::string Workspace::GetConsistencyGaphPath(const int image_id) const
		{
			return m_model.m_images.at(image_id).GetConsistencyPath() + GetFileName(image_id, false);
		}

		std::string Workspace::GetFileName(const int image_id, const bool isGeom) const
		{
			const auto& image_name = m_model.GetImageName(image_id);

			const std::string file_type = ".bin";
			std::string fileName;
			if (!isGeom)  // 如果不是几何一致性的，
			{
				fileName = image_name + "." + options_.input_type + file_type;
			}
			else  // 如果是几何一致性的
			{
				fileName = image_name + "." + options_.input_type_geom + file_type;
			}
			return fileName;
		}

		float Workspace::GetDepthRange(const int image_id, bool isMax) const
		{
			return isMax ? depth_ranges_.at(image_id).second : depth_ranges_.at(image_id).first;
		}

		bool Workspace::HasBitmap(const int image_id) const
		{
			return hasBitMaps_.at(image_id);
		}

		bool Workspace::HasDepthMap(const int image_id, const bool isGeom) const
		{

			//return (hasReadMapsGeom_ || hasReadMapsPhoto_);
			return _access(GetDepthMapPath(image_id, isGeom).c_str(), 0);
		}

		bool Workspace::HasNormalMap(const int image_id, const bool isGeom) const
		{

			//return (hasReadMapsGeom_ || hasReadMapsPhoto_);
			return _access(GetNormalMapPath(image_id, isGeom).c_str(), 0);
		}

		//联合双边上采样
		void Workspace::jointBilateralUpsampling(const cv::Mat &joint, const cv::Mat &lowin, const float upscale,
			const double sigma_color, const double sigma_space, int radius, cv::Mat &highout) const
		{
			highout.create(joint.size(), lowin.type());
			const int highRow = joint.rows;
			const int highCol = joint.cols;
			const int lowRow = lowin.rows;
			const int lowCol = lowin.cols;

			if (radius <= 0)
				radius = round(sigma_space * 1.5);
			const int d = 2 * radius + 1;

			// 原联合图像的通道数
			const int cnj = joint.channels();

			float *color_weight = new float[cnj * 256];
			float *space_weight = new float[d*d];
			int *space_ofs_row = new int[d*d];  // 坐标的差值
			int *space_ofs_col = new int[d*d];

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

			// initialize color-related bilateral filter coefficients  
			// 色差的高斯权重  
			for (int i = 0; i < 256 * cnj; i++)
				color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

			int maxk = 0;   // 0 - (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			// 空间差的高斯权重
			for (int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					double r = std::sqrt((double)i * i + (double)j * j);
					if (r > radius)
						continue;

					// 空间权重是作用在小图像上的
					space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff / (upscale*upscale));
					space_ofs_row[maxk] = i;
					space_ofs_col[maxk++] = j;
				}
			}

			for (int r = 0; r < highRow; r++)
			{
				for (int l = 0; l < highCol; l++)
				{
					int px = l, py = r;  // 窗口中心像素
					//float fpx = (float)px / upscale;
					//float fpy = (float)py / upscale;
					const cv::Vec3b color0 = joint.ptr<cv::Vec3b>(py)[px];
					float sum_w = 0;
					float sum_value[3] = { 0 };
					for (int k = 0; k < maxk; k++)
					{
						const int qy = py + space_ofs_row[k];
						const int qx = px + space_ofs_col[k];

						if (qx < 0 || qx >= highCol || qy < 0 || qy >= highRow)
							continue;

						float fqx = (float)qx / upscale;//低分辨率图像对应坐标
						float fqy = (float)qy / upscale;
						int iqx = roundf(fqx);//四舍五入
						int iqy = roundf(fqy);
						if (iqx >= lowCol || iqy >= lowRow)
							continue;

						// 颜色距离权重，是作用在高分辨率图像上的
						cv::Vec3b color1 = joint.ptr<cv::Vec3b>(qy)[qx];

						// 根据joint当前像素和邻域像素的 距离权重 和 色差权重，计算综合的权重  
						float w = space_weight[k] * color_weight[abs(color0[0] - color1[0]) + abs(color0[1] - color1[1]) + abs(color0[2] - color1[2])];

						if (lowin.type() == CV_8UC3)
						{
							sum_value[0] += lowin.ptr<cv::Vec3b>(iqy)[iqx][0] * w;
							sum_value[1] += lowin.ptr<cv::Vec3b>(iqy)[iqx][1] * w;
							sum_value[2] += lowin.ptr<cv::Vec3b>(iqy)[iqx][2] * w;
						}
						else if (lowin.type() == CV_8UC1)
						{
							sum_value[0] += lowin.ptr<uchar>(iqy)[iqx] * w;
						}
						else if (lowin.type() == CV_32FC3)
						{
							sum_value[0] += lowin.ptr<cv::Vec3f>(iqy)[iqx][0] * w;
							sum_value[1] += lowin.ptr<cv::Vec3f>(iqy)[iqx][1] * w;
							sum_value[2] += lowin.ptr<cv::Vec3f>(iqy)[iqx][2] * w;
						}
						else if (lowin.type() == CV_32FC1)
						{
							sum_value[0] += lowin.ptr<float>(iqy)[iqx] * w;
						}
						sum_w += w;
					}
					//for (int i = -radius; i <= radius; i++)
					//{
					//	for (int j = -radius; j <= radius; j++)
					//	{
					//		int qx = px + j, qy = py + i;//窗口内像素
					//		if (qx < 0 || qx >= highCol || qy < 0 || qy >= highRow)
					//			continue;
					//
					//		float fqx = (float)qx / upscale;//低分辨率图像对应坐标
					//		float fqy = (float)qy / upscale;
					//		int iqx = roundf(fqx);//四舍五入
					//		int iqy = roundf(fqy);
					//		if (iqx >= lowCol || iqy >= lowRow)
					//			continue;
					//
					//		//空间距离权重，是作用在低分辨率图像上的
					//		float spaceDis = (i*i + j*j) / (upscale*upscale);
					//		float space_w = (float)std::exp(spaceDis * gauss_space_coeff);
					//		//颜色距离权重，是作用在高分辨率图像上的
					//		cv::Vec3b color1 = joint.ptr<cv::Vec3b>(qy)[qx];
					//		float color_w = color_weight[abs(color0[0] - color1[0]) + abs(color0[1] - color1[1]) + abs(color0[2] - color1[2])];
					//
					//		float w = space_w*color_w;
					//		if (lowin.type()==CV_8UC3)
					//		{
					//			sum_value[0] += lowin.ptr<cv::Vec3b>(iqy)[iqx][0] * w;
					//			sum_value[1] += lowin.ptr<cv::Vec3b>(iqy)[iqx][1] * w;
					//			sum_value[2] += lowin.ptr<cv::Vec3b>(iqy)[iqx][2] * w;
					//		}
					//		else if (lowin.type()==CV_8UC1)
					//		{
					//			sum_value[0] += lowin.ptr<uchar>(iqy)[iqx] * w;
					//		}
					//		else if (lowin.type() == CV_32FC3)
					//		{
					//			sum_value[0] += lowin.ptr<cv::Vec3f>(iqy)[iqx][0] * w;
					//			sum_value[1] += lowin.ptr<cv::Vec3f>(iqy)[iqx][1] * w;
					//			sum_value[2] += lowin.ptr<cv::Vec3f>(iqy)[iqx][2] * w;
					//		}
					//		else if (lowin.type() == CV_32FC1)
					//		{
					//			sum_value[0] += lowin.ptr<float>(iqy)[iqx] * w;
					//		}
					//		sum_w += w;
					//	}
					//}
					sum_w = 1.f / sum_w;
					if (lowin.type() == CV_8UC3)
					{
						highout.ptr<cv::Vec3b>(py)[px] = cv::Vec3b(sum_value[0] * sum_w, sum_value[1] * sum_w, sum_value[2] * sum_w);
					}
					else if (lowin.type() == CV_8UC1)
					{
						highout.ptr<uchar>(py)[px] = sum_value[0] * sum_w;
					}
					else if (lowin.type() == CV_32FC3)
					{
						highout.ptr<cv::Vec3f>(py)[px] = cv::Vec3f(sum_value[0] * sum_w, sum_value[1] * sum_w, sum_value[2] * sum_w);
					}
					else if (lowin.type() == CV_32FC1)
					{
						highout.ptr<float>(py)[px] = sum_value[0] * sum_w;
					}

				}
			}
		}

		// 联合双边传播上采样
		void Workspace::jointBilateralPropagationUpsampling(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
			const float upscale, const double sigma_color, const double sigma_space, const int radius, cv::Mat &highDepthMat) const
		{

		}

		// 联合双边传播滤波
		void Workspace::jointBilateralDepthMapFilter1(const cv::Mat &srcDepthMap, const cv::Mat &srcNormalMap, const cv::Mat &srcImage, const float *refK,
			const int radius, const double sigma_color, const double sigma_space, DepthMap &desDepMap, NormalMap &desNorMap, const bool DoNormal)const
		{

		}

		float Workspace::PropagateDepth(const float *refK,
			const float depth_1, const float normal_1[3],
			const float row_1, const float col_1, const float row_2, const float col_2) const
		{
			// Extract 1/fx, -cx/fx, 1/fy, -cy/fy.
			const float ref_inv_K[4] = { 1.0f / refK[0], -refK[2] / refK[0], 1.0f / refK[4], -refK[5] / refK[4] };

			// Point along first viewing ray.
			const float p1[3] = {
				depth_1 * (ref_inv_K[0] * col_1 + ref_inv_K[1]),
				depth_1 * (ref_inv_K[2] * row_1 + ref_inv_K[3]),
				depth_1
			};

			// Point on second viewing ray.
			const float p2[3] = {
				ref_inv_K[0] * col_2 + ref_inv_K[1],
				ref_inv_K[2] * row_2 + ref_inv_K[3],
				1.0f
			};

			const float denom = (p1[0] * normal_1[0] + p1[1] * normal_1[1] + p1[2] * normal_1[2]) /
				(p2[0] * normal_1[0] + p2[1] * normal_1[1] + p2[2] * normal_1[2]);

			const float lowDepth = depth_1 * 0.95;
			const float highDepth = depth_1 * 1.05;

			//cout << row1 << "," << col1 << " --->" << row2 << "," << col2 << endl;
			//cout << depth1<<"--->"<< denom <<"  (" <<lowDepth << "," << highDepth <<")"<< endl;

			return denom < lowDepth ? lowDepth : (denom > highDepth ? highDepth : denom);
		}

		void Workspace::SuitNormal(const int row, const int col,
			const float* refK, float normal[3]) const
		{

			// Extract 1/fx, -cx/fx, 1/fy, -cy/fy.
			const float ref_inv_K[4] = {
				1.0f / refK[0],            // 1/fx
				-refK[2] / refK[0],        // -cx/fx
				1.0f / refK[4],            // 1/fy
				-refK[5] / refK[4]         // -cy/fy
			};

			// Make sure the perturbed normal is still looking in the same direction as
			// the viewing direction.
			const float view_ray[3] = {
				ref_inv_K[0] * col + ref_inv_K[1],
				ref_inv_K[2] * row + ref_inv_K[3],
				1.0f
			};
			if ((normal[0] * view_ray[0] + normal[1] * view_ray[1] + normal[2] * view_ray[2]) >= 0.0f)
			{

				normal[0] *= -1.0f;
				normal[1] *= -1.0f;
				normal[2] *= -1.0f;
			}

			// Make sure normal has unit norm.
			float norm = sqrt(normal[0] * normal[0]
				+ normal[1] * normal[1] + normal[2] * normal[2]);
			if (norm < 1e-8)
			{
				//cout << "[Warning]: very small normal L2 norm!" << endl;
				norm += float(1e-8);
			}

			// 如果不是单位向量，那么归一化: 除以一个极小值可能溢出
			const float inv_norm = 1.0f / norm;
			if (inv_norm != 1.0f)
			{
				normal[0] *= inv_norm;
				normal[1] *= inv_norm;
				normal[2] *= inv_norm;
			}
		}

		// 对法向量图进行类中值滤波
		void Workspace::NormalMapMediaFilter(const cv::Mat& InNormalMapMat,
			cv::Mat& OutNormalMapMat, const int windowRadis) const
		{

		}

		// 对法向量图进行类中值滤波，剔除为0的数据
		void Workspace::NormalMapMediaFilter1(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat, const int windowRadis) const
		{

		}

		// 对法向量和深度图都进行中值滤波操作，剔除为0的数据
		void Workspace::NormalMapMediaFilterWithDepth(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat,
			const cv::Mat &InDepthMapMat, cv::Mat &OutDepthMapMat, int windowRadis) const
		{

		}

		void Workspace::newPropagation(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
			const float upscale, const double sigma_color, const double sigma_space, int radius, const int maxSrcPoint,
			cv::Mat &highDepthMat, cv::Mat &highNormalMat) const
		{

		}

		void Workspace::newPropagationFast(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat,
			const float *refK, const double sigma_color, const double sigma_space, int radius, const int maxSrcPoint,
			cv::Mat &outDepthMat, cv::Mat &outNormalMat) const
		{

		}

		// CV_32F进行FilterSpeckles
		typedef cv::Point_<short> Point2s;
		//typedef cv::Point_<float> Point2s;

		template <typename T>
		void Workspace::FilterSpeckles(cv::Mat& img, T newVal, int maxSpeckleSize, T maxDiff)
		{
			using namespace cv;

			cv::Mat _buf;

			int width = img.cols, height = img.rows, npixels = width * height;

			// each pixel contains: pixel coordinate(Point2S), label(int), 是否是blob(uchar)
			size_t bufSize = npixels * (int)(sizeof(Point2s) + sizeof(int) + sizeof(uchar));
			if (!_buf.isContinuous() || _buf.empty() || _buf.cols*_buf.rows*_buf.elemSize() < bufSize)
				_buf.reserveBuffer(bufSize);

			uchar* buf = _buf.ptr();
			int i, j, dstep = (int)(img.step / sizeof(T));
			int* labels = (int*)buf;
			buf += npixels * sizeof(labels[0]);
			Point2s* wbuf = (Point2s*)buf;
			buf += npixels * sizeof(wbuf[0]);
			uchar* rtype = (uchar*)buf;
			int curlabel = 0;

			// clear out label assignments
			memset(labels, 0, npixels * sizeof(labels[0]));

			for (i = 0; i < height; i++)
			{
				T* ds = img.ptr<T>(i);
				int* ls = labels + width * i;

				for (j = 0; j < width; j++)
				{
					if (ds[j] != newVal)   // not a bad disparity
					{
						if (ls[j])     // has a label, check for bad label
						{
							if (rtype[ls[j]]) // small region, zero out disparity
								ds[j] = (T)newVal;
						}
						// no label, assign and propagate
						else
						{
							Point2s* ws = wbuf; // initialize wavefront
							Point2s p((short)j, (short)i);  // current pixel
							curlabel++; // next label
							int count = 0;  // current region size
							ls[j] = curlabel;

							// wavefront propagation
							while (ws >= wbuf) // wavefront not empty
							{
								count++;
								// put neighbors onto wavefront
								T* dpp = &img.at<T>(p.y, p.x);
								T dp = *dpp;
								int* lpp = labels + width * p.y + p.x;

								// down neighbor
								if (p.y < height - 1 && !lpp[+width] && dpp[+dstep] != newVal && std::abs(dp - dpp[+dstep]) <= maxDiff)
								{
									lpp[+width] = curlabel;
									*ws++ = Point2s(p.x, p.y + 1);
								}

								// top neighbor
								if (p.y > 0 && !lpp[-width] && dpp[-dstep] != newVal && std::abs(dp - dpp[-dstep]) <= maxDiff)
								{
									lpp[-width] = curlabel;
									*ws++ = Point2s(p.x, p.y - 1);
								}

								// right neighbor
								if (p.x < width - 1 && !lpp[+1] && dpp[+1] != newVal && std::abs(dp - dpp[+1]) <= maxDiff)
								{
									lpp[+1] = curlabel;
									*ws++ = Point2s(p.x + 1, p.y);
								}

								// left neighbor
								if (p.x > 0 && !lpp[-1] && dpp[-1] != newVal && std::abs(dp - dpp[-1]) <= maxDiff)
								{
									lpp[-1] = curlabel;
									*ws++ = Point2s(p.x - 1, p.y);
								}

								// pop most recent and propagate
								// NB: could try least recent, maybe better convergence
								p = *--ws;
							}

							// assign label type
							if (count <= maxSpeckleSize)   // speckle region
							{
								rtype[ls[j]] = 1;   // small region label
								ds[j] = (T)newVal;
							}
							else
								rtype[ls[j]] = 0;   // large region label
						}
					}
				}
			}
		}

		float Workspace::GetDepthCoPlane(const float* K_inv_arr,
			const float* R_inv_arr,
			const float* T_arr,
			const float* plane_arr,
			const cv::Point2f& pt2D)
		{
			// K_inv parameters
			const float& k11 = K_inv_arr[0];
			const float& k12 = K_inv_arr[1];
			const float& k13 = K_inv_arr[2];
			const float& k21 = K_inv_arr[3];
			const float& k22 = K_inv_arr[4];
			const float& k23 = K_inv_arr[5];
			const float& k31 = K_inv_arr[6];
			const float& k32 = K_inv_arr[7];
			const float& k33 = K_inv_arr[8];

			// R_inv parameters
			const float& r11 = R_inv_arr[0];
			const float& r12 = R_inv_arr[1];
			const float& r13 = R_inv_arr[2];
			const float& r21 = R_inv_arr[3];
			const float& r22 = R_inv_arr[4];
			const float& r23 = R_inv_arr[5];
			const float& r31 = R_inv_arr[6];
			const float& r32 = R_inv_arr[7];
			const float& r33 = R_inv_arr[8];

			// T parameters
			const float& t1 = T_arr[0];
			const float& t2 = T_arr[1];
			const float& t3 = T_arr[2];

			// plane parameters
			const float& n1 = plane_arr[0];
			const float& n2 = plane_arr[1];
			const float& n3 = plane_arr[2];
			const float&  d = plane_arr[3];

			// 2d point coordinates
			const float& x2d = pt2D.x;
			const float& y2d = pt2D.y;

			// calculate depth by co-plane constraint
			return (n1*r11*t1 + n1 * r12*t2 + n1 * r13*t3 + n2 * r21*t1 + n2 * r22*t2 + n2 * r23*t3 + n3 * r31*t1 + n3 * r32*t2 + n3 * r33*t3 - d) / ((n1*r11 + n2 * r21 + n3 * r31)*(k11*x2d + k12 * y2d + k13) + (n1*r12 + n2 * r22 + n3 * r32)*(k21*x2d + k22 * y2d + k23) + (n1*r13 + n2 * r23 + n3 * r33)*(k31*x2d + k32 * y2d + k33));
		}

		float Workspace::GetDepthCoPlaneCam(const float* K_inv_arr,
			const float* plane_arr,
			const cv::Point2f& pt2D)
		{
			// K_inv parameters
			const float& k11 = K_inv_arr[0];
			const float& k12 = K_inv_arr[1];
			const float& k13 = K_inv_arr[2];
			const float& k21 = K_inv_arr[3];
			const float& k22 = K_inv_arr[4];
			const float& k23 = K_inv_arr[5];
			const float& k31 = K_inv_arr[6];
			const float& k32 = K_inv_arr[7];
			const float& k33 = K_inv_arr[8];

			// plane parameters
			const float& n1 = plane_arr[0];
			const float& n2 = plane_arr[1];
			const float& n3 = plane_arr[2];
			const float&  d = plane_arr[3];

			// 2d point coordinates
			const float& x2d = pt2D.x;
			const float& y2d = pt2D.y;

			return -d / (n1*(k11*x2d + k12 * y2d + k13) + n2 * (k21*x2d + k22 * y2d + k23) + n3 * (k31*x2d + k32 * y2d + k33));
		}

		// 修正superpixel点云的切平面
		int Workspace::CorrectPCPlane(const float* P_arr,
			const float* K_inv_arr,
			const float* R_inv_arr,
			const float* T_arr,
			const int IMG_WIDTH,   // 2D图像宽度
			const int IMG_HEIGHT,  // 2D图像高度
			const float DIST_THRESH,  // 切平面空间距离阈值
			const float fold,
			const std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<float>>& plane_map,
			std::unordered_map<int, cv::Point3f>& center_map,  // superpixel, 3D点云中心坐标
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, std::vector<float>>& eigen_vals_map,
			std::unordered_map<int, std::vector<float>>& eigen_vects_map)
		{
			for (auto it_1 = label_map.begin(); it_1 != label_map.end(); ++it_1)
			{
				// 遍历每一个superpixel, 获取其neighbors
				std::set<int> Neighbors;

				for (auto it_2 = label_map.begin(); it_2 != label_map.end(); ++it_2)
				{
					if (it_2->first != it_1->first)
					{
						// Ray tracing: 光线与两个切平面区域相交于2个3D点对
						std::vector<std::pair<cv::Point3f, cv::Point3f>> In_2_Plane_3DPtPairs;

						// ----- 对计算plane之间的距离(取2交点之间距离最大值)
						// 计算superpixel1的2D坐标范围
						float x_range_1[2], y_range_1[2], x_range_2[2], y_range_2[2];

						const auto& center_1 = center_map[it_1->first];
						const auto& center_2 = center_map[it_2->first];

						const float* ei_vals_1 = eigen_vals_map[it_1->first].data();
						const float* ei_vals_2 = eigen_vals_map[it_2->first].data();

						const float* ei_vects_1 = eigen_vects_map[it_1->first].data();  // 第一个plane的3个特征向量
						const float* ei_vects_2 = eigen_vects_map[it_2->first].data();  // 第二个plane的3个特征向量

						const float* ei_vects_1_tagent_1 = &ei_vects_1[3];  // 第一个plane的切向分量 1
						const float* ei_vects_1_tagent_2 = &ei_vects_1[6];  // 第一个plane的切向分量 2

						const float* ei_vects_2_tagent_1 = &ei_vects_2[3];  // 第二个plane的切向分量 1
						const float* ei_vects_2_tagent_2 = &ei_vects_2[6];  // 第二个plane的切向分量 2 

						int ret = this->GetSearchRange(center_1,
							P_arr,
							ei_vals_1, ei_vects_1,
							fold,
							x_range_1, y_range_1);
						if (ret < 0)
						{
							continue;
						}

						// 计算superpixel2的2D坐标范围
						ret = this->GetSearchRange(center_2,
							P_arr,
							ei_vals_2, ei_vects_2,
							fold,
							x_range_2, y_range_2);
						if (ret < 0)
						{
							continue;
						}

						x_range_1[0] = x_range_1[0] >= 0 ? x_range_1[0] : 0;
						x_range_1[1] = x_range_1[1] <= IMG_WIDTH - 1 ? x_range_1[1] : IMG_WIDTH - 1;
						y_range_1[0] = y_range_1[0] >= 0 ? y_range_1[0] : 0;
						y_range_1[1] = y_range_1[1] <= IMG_HEIGHT - 1 ? y_range_1[1] : IMG_HEIGHT - 1;

						x_range_2[0] = x_range_2[0] >= 0 ? x_range_2[0] : 0;
						x_range_2[1] = x_range_2[1] <= IMG_WIDTH - 1 ? x_range_2[1] : IMG_WIDTH - 1;
						y_range_2[0] = y_range_2[0] >= 0 ? y_range_2[0] : 0;
						y_range_2[1] = y_range_2[1] <= IMG_HEIGHT - 1 ? y_range_2[1] : IMG_HEIGHT - 1;

						// 如果2个矩形区域不相交
						if (y_range_1[1] <= y_range_2[0] || y_range_2[1] <= y_range_1[0]
							|| x_range_1[1] <= x_range_2[0] || x_range_2[1] <= x_range_1[0])
						{
							//printf("Superpixel %d and superpixel %d not intersect\n",
							//	it_1->first, it_2->first);
							continue;
						}

						const int X_MIN = (int)std::max(x_range_1[0], x_range_2[0]);
						const int X_MAX = (int)std::min(x_range_1[1], x_range_2[1]);
						const int Y_MIN = (int)std::max(y_range_1[0], y_range_2[0]);
						const int Y_MAX = (int)std::min(y_range_1[1], y_range_2[1]);

						if (Y_MIN >= Y_MAX || X_MIN >= X_MAX)
						{
							continue;
						}

						// ----- 遍历plane_1, plane_2重合的搜索区域(不用全部2D坐标遍历)
						for (int y = Y_MIN; y <= Y_MAX; ++y)
						{
							for (int x = X_MIN; x <= X_MAX; ++x)
							{
								cv::Point2f pt2D((float)x, (float)y);

								// 计算与plane的交点的深度值
								float depth_1 = this->GetDepthCoPlane(K_inv_arr, R_inv_arr, T_arr,
									plane_map[it_1->first].data(),
									pt2D);
								float depth_2 = this->GetDepthCoPlane(K_inv_arr, R_inv_arr, T_arr,
									plane_map[it_2->first].data(),
									pt2D);

								// 无效深度值, 跳过
								if (isnan(depth_1) || isnan(depth_2))
								{
									continue;
								}

								// 计算与plane的交点(3D)
								cv::Point3f pt3D_1 = this->m_model.BackProjTo3D(K_inv_arr,
									R_inv_arr,
									T_arr,
									depth_1, pt2D);
								cv::Point3f pt3D_2 = this->m_model.BackProjTo3D(K_inv_arr,
									R_inv_arr,
									T_arr,
									depth_2, pt2D);

								// 如果两个3D交点, 同时在各自切平面范围内
								if (
									this->IsPt3DInPlaneRange(center_1, pt3D_1,
										ei_vals_1[1], ei_vals_1[2],  // plane_1的切平面特征值分量
										ei_vects_1_tagent_1, ei_vects_1_tagent_2,  // plane_1的特征向量,切平面分量
										fold)  // 如何设置合理的切平面范围
									&&
									this->IsPt3DInPlaneRange(center_2, pt3D_2,
										ei_vals_2[1], ei_vals_2[2],  // plane_2的切平面特征值分量
										ei_vects_2_tagent_1, ei_vects_2_tagent_2,  // plane_2的特征向量,切平面分量
										fold))
								{
									In_2_Plane_3DPtPairs.push_back(std::make_pair(pt3D_1, pt3D_2));
								}

								//In_2_Plane_3DPtPairs.push_back(std::make_pair(pt3D_1, pt3D_2));
							}
						}

						// 3D点对为空, 跳过it_2->first
						if (In_2_Plane_3DPtPairs.size() < 3)
						{
							continue;
						}

						// 统计距离最大值
						float dist_max = FLT_MIN;
						for (auto pts_pair : In_2_Plane_3DPtPairs)
						{
							float dist = sqrtf((pts_pair.first.x - pts_pair.second.x)
								* (pts_pair.first.x - pts_pair.second.x)
								+ (pts_pair.first.y - pts_pair.second.y)
								* (pts_pair.first.y - pts_pair.second.y)
								+ (pts_pair.first.z - pts_pair.second.z)
								* (pts_pair.first.z - pts_pair.second.z));
							if (dist > dist_max)
							{
								dist_max = dist;
							}
						}

						//// 统计距离均值
						//float dist_mean = 0.0f;
						//for (auto pts_pair : In_2_Plane_3DPtPairs)
						//{
						//	dist_mean += sqrtf((pts_pair.first.x - pts_pair.second.x)
						//		* (pts_pair.first.x - pts_pair.second.x)
						//		+ (pts_pair.first.y - pts_pair.second.y)
						//		* (pts_pair.first.y - pts_pair.second.y)
						//		+ (pts_pair.first.z - pts_pair.second.z)
						//		* (pts_pair.first.z - pts_pair.second.z));
						//}
						//dist_mean /= float(In_2_Plane_3DPtPairs.size());

						// 判断superpixel it_2->first 是否是neighbor
						if (dist_max < DIST_THRESH)
							//if (dist_mean < DIST_THRESH)
						{
							Neighbors.insert(it_2->first);
						}
					}
				}

				// 如果Neighbor太少, 不修正
				if (Neighbors.size() < 4)  // 设置最小Neighbor数量
				{
					//printf("[Note]: Superpixel %d, neighbor less than 2\n", 
					//	it_1->first);
					continue;
				}

				// ----- 基于Neighbors, 修正superpixel it_1->first 的切平面
				// 获取Neighbor superpixel的点云中心
				std::vector<cv::Point3f> neighbor_3dpts(Neighbors.size() + 1);
				int k = 0;
				for (int neigh : Neighbors)
				{
					neighbor_3dpts[k++] = center_map[neigh];
				}
				neighbor_3dpts[k] = center_map[it_1->first];

				// 基于Neighbor中心的点云, 重新拟合切平面
				this->FitPlaneForSuperpixel(neighbor_3dpts,
					plane_map[it_1->first],                // 更新superpixel切平面方程
					eigen_vals_map[it_1->first],           // 更新superpixel特征值
					eigen_vects_map[it_1->first],          // 更新superpixel特征向量
					center_map[it_1->first]);              // 更新superpixel点云中心

				// 更新superpixel法向量
				plane_normal_map[it_1->first].reserve(3);
				plane_normal_map[it_1->first].resize(3);
				memcpy(plane_normal_map[it_1->first].data(),
					plane_map[it_1->first].data(), sizeof(float) * 3);

				//printf("Superpixel %d tagent plane corrected by %d neighbors\n", 
				//	it_1->first, (int)Neighbors.size());
			}

			return 0;
		}

		int Workspace::CorrectPlaneCam(const float* K_arr,
			const float* K_inv_arr,
			const int IMG_WIDTH,
			const int IMG_HEIGHT,
			const float DIST_THRESH,
			const float fold, const int TH_Num_Neigh,
			const std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<float>>& plane_map,
			std::unordered_map<int, cv::Point3f>& center_map,
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, std::vector<float>>& eigen_vals_map,
			std::unordered_map<int, std::vector<float>>& eigen_vects_map)
		{
			for (auto it_1 = label_map.begin(); it_1 != label_map.end(); ++it_1)
			{
				// 遍历每一个superpixel, 获取其neighbors
				std::set<int> Neighbors;

				for (auto it_2 = label_map.begin(); it_2 != label_map.end(); ++it_2)
				{
					if (it_2->first != it_1->first)
					{
						// Ray tracing: 光线与两个切平面区域相交于2个3D点对
						std::vector<std::pair<cv::Point3f, cv::Point3f>> In_2_Plane_3DPtPairs;

						// ----- 对计算plane之间的距离(取2交点之间距离最大值)
						// 计算superpixel1的2D坐标范围
						float x_range_1[2], y_range_1[2], x_range_2[2], y_range_2[2];

						const auto& center_1 = center_map[it_1->first];
						const auto& center_2 = center_map[it_2->first];

						const float* ei_vals_1 = eigen_vals_map[it_1->first].data();
						const float* ei_vals_2 = eigen_vals_map[it_2->first].data();

						const float* ei_vects_1 = eigen_vects_map[it_1->first].data();  // 第一个plane3个特征向量
						const float* ei_vects_2 = eigen_vects_map[it_2->first].data();  // 第二个plane3个特征向量

						const float* ei_vects_1_tagent_1 = &ei_vects_1[3];  // 第一个plane切向分量 1
						const float* ei_vects_1_tagent_2 = &ei_vects_1[6];  // 第一个plane切向分量 2

						const float* ei_vects_2_tagent_1 = &ei_vects_2[3];  // 第二个plane切向分量 1
						const float* ei_vects_2_tagent_2 = &ei_vects_2[6];  // 第二个plane切向分量 2 

						int ret = this->GetSearchRangeCam(center_1,
							K_arr,
							ei_vals_1, ei_vects_1,
							fold,
							x_range_1, y_range_1);
						if (ret < 0)
						{
							continue;
						}

						// 计算superpixel2的2D坐标范围
						ret = this->GetSearchRangeCam(center_2,
							K_arr,
							ei_vals_2, ei_vects_2,
							fold,
							x_range_2, y_range_2);
						if (ret < 0)
						{
							continue;
						}

						x_range_1[0] = x_range_1[0] >= 0 ? x_range_1[0] : 0;
						x_range_1[1] = x_range_1[1] <= IMG_WIDTH - 1 ? x_range_1[1] : IMG_WIDTH - 1;
						y_range_1[0] = y_range_1[0] >= 0 ? y_range_1[0] : 0;
						y_range_1[1] = y_range_1[1] <= IMG_HEIGHT - 1 ? y_range_1[1] : IMG_HEIGHT - 1;

						x_range_2[0] = x_range_2[0] >= 0 ? x_range_2[0] : 0;
						x_range_2[1] = x_range_2[1] <= IMG_WIDTH - 1 ? x_range_2[1] : IMG_WIDTH - 1;
						y_range_2[0] = y_range_2[0] >= 0 ? y_range_2[0] : 0;
						y_range_2[1] = y_range_2[1] <= IMG_HEIGHT - 1 ? y_range_2[1] : IMG_HEIGHT - 1;

						// 如果2个矩形区域不相交
						if (y_range_1[1] <= y_range_2[0] || y_range_2[1] <= y_range_1[0]
							|| x_range_1[1] <= x_range_2[0] || x_range_2[1] <= x_range_1[0])
						{
							//printf("Superpixel %d and superpixel %d not intersect\n",
							//	it_1->first, it_2->first);
							continue;
						}

						const int X_MIN = (int)std::max(x_range_1[0], x_range_2[0]);
						const int X_MAX = (int)std::min(x_range_1[1], x_range_2[1]);
						const int Y_MIN = (int)std::max(y_range_1[0], y_range_2[0]);
						const int Y_MAX = (int)std::min(y_range_1[1], y_range_2[1]);

						if (Y_MIN >= Y_MAX || X_MIN >= X_MAX)
						{
							continue;
						}

						// ----- 遍历plane_1, plane_2重合的搜索区域(不用全部2D坐标遍历)
						for (int y = Y_MIN; y <= Y_MAX; ++y)
						{
							for (int x = X_MIN; x <= X_MAX; ++x)
							{
								cv::Point2f pt2D((float)x, (float)y);

								// 计算与plane的交点的深度值
								float depth_1 = this->GetDepthCoPlaneCam(K_inv_arr,
									plane_map[it_1->first].data(),
									pt2D);
								float depth_2 = this->GetDepthCoPlaneCam(K_inv_arr,
									plane_map[it_2->first].data(),
									pt2D);

								// 无效深度值, 跳过
								if (isnan(depth_1) || isnan(depth_2))
								{
									continue;
								}

								// 计算与plane的交点(3D)
								cv::Point3f pt3D_1 = this->m_model.BackProjTo3DCam(K_inv_arr,
									depth_1, pt2D);
								cv::Point3f pt3D_2 = this->m_model.BackProjTo3DCam(K_inv_arr,
									depth_2, pt2D);

								// 如果两个3D交点, 同时在各自切平面范围内
								if (
									this->IsPt3DInPlaneRange(center_1, pt3D_1,
										ei_vals_1[1], ei_vals_1[2],  // plane_1的切平面特征值分量
										ei_vects_1_tagent_1, ei_vects_1_tagent_2,  // plane_1的特征向量,切平面分量
										fold)  // 如何设置合理的切平面范围
									&&
									this->IsPt3DInPlaneRange(center_2, pt3D_2,
										ei_vals_2[1], ei_vals_2[2],  // plane_2的切平面特征值分量
										ei_vects_2_tagent_1, ei_vects_2_tagent_2,  // plane_2的特征向量,切平面分量
										fold))
								{
									In_2_Plane_3DPtPairs.push_back(std::make_pair(pt3D_1, pt3D_2));
								}

								//In_2_Plane_3DPtPairs.push_back(std::make_pair(pt3D_1, pt3D_2));
							}
						}

						// 3D点对为空, 跳过it_2->first
						if (In_2_Plane_3DPtPairs.size() < 3)
						{
							continue;
						}

						// 统计距离最大值
						float dist_max = FLT_MIN;
						for (auto pts_pair : In_2_Plane_3DPtPairs)
						{
							float dist = sqrtf((pts_pair.first.x - pts_pair.second.x)
								* (pts_pair.first.x - pts_pair.second.x)
								+ (pts_pair.first.y - pts_pair.second.y)
								* (pts_pair.first.y - pts_pair.second.y)
								+ (pts_pair.first.z - pts_pair.second.z)
								* (pts_pair.first.z - pts_pair.second.z));
							if (dist > dist_max)
							{
								dist_max = dist;
							}
						}

						//// 统计距离均值
						//float dist_mean = 0.0f;
						//for (auto pts_pair : In_2_Plane_3DPtPairs)
						//{
						//	dist_mean += sqrtf((pts_pair.first.x - pts_pair.second.x)
						//		* (pts_pair.first.x - pts_pair.second.x)
						//		+ (pts_pair.first.y - pts_pair.second.y)
						//		* (pts_pair.first.y - pts_pair.second.y)
						//		+ (pts_pair.first.z - pts_pair.second.z)
						//		* (pts_pair.first.z - pts_pair.second.z));
						//}
						//dist_mean /= float(In_2_Plane_3DPtPairs.size());

						// 判断superpixel it_2->first 是否是neighbor
						if (dist_max < DIST_THRESH)
							//if (dist_mean < DIST_THRESH)
						{
							Neighbors.insert(it_2->first);
						}
					}
				}

				// 如果Neighbor太少, 不修正
				if (Neighbors.size() < TH_Num_Neigh)  // 设置最小Neighbor数量
				{
					//printf("[Note]: Superpixel %d, neighbor less than 5\n", 
					//	it_1->first);
					continue;
				}

				// ----- 基于Neighbors, 修正superpixel it_1->first 的切平面
				// 获取Neighbor superpixel的点云中心
				std::vector<cv::Point3f> neighbor_3dpts(Neighbors.size() + 1);
				int k = 0;
				for (int neigh : Neighbors)
				{
					neighbor_3dpts[k++] = center_map[neigh];
				}
				neighbor_3dpts[k] = center_map[it_1->first];

				// 基于Neighbor中心的点云, 重新拟合切平面
				this->FitPlaneForSuperpixel(neighbor_3dpts,
					plane_map[it_1->first],                // 更新superpixel切平面方程
					eigen_vals_map[it_1->first],           // 更新superpixel特征值
					eigen_vects_map[it_1->first],          // 更新superpixel特征向量
					center_map[it_1->first]);              // 更新superpixel点云中心

				// 更新superpixel法向量
				plane_normal_map[it_1->first].resize(3, 0.0f);
				memcpy(plane_normal_map[it_1->first].data(),
					plane_map[it_1->first].data(), sizeof(float) * 3);

#ifdef LOGGING
				std::printf("SP %d corrected by %d neighbors:",
					it_1->first, (int)Neighbors.size());
				for (auto neigh : Neighbors)
				{
					std::printf(" %d ", neigh);
				}
				std::printf("\n");
#endif // LOGGING
			}

			return 0;
		}

		// 基于点云切平面, 对点云进行网格平滑
		int Workspace::SmoothPointCloud(const float* R_arr,
			const float* T_arr,
			const float* K_inv_arr,
			const float* R_inv_arr,
			DepthMap& depth_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, std::vector<float>>& ei_vects_map)
		{
			// 遍历每个superpixel
			for (auto it = label_map.begin(); it != label_map.end(); ++it)
			{
				// 取superpixel对应的2D点集
				std::vector<cv::Point2f>& pts2D = it->second;

				// 反投影到3D空间
				std::vector<cv::Point3f> pts3D(pts2D.size());

				// 存放光滑后的3D点
				std::vector<cv::Point3f> pts3D_new(pts2D.size());

				// 计算3D点云坐标(世界坐标系)
				int k = 0;  // 计数
				for (auto pt2d : pts2D)
				{
					const float& depth = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
					cv::Point3f pt3d = this->m_model.BackProjTo3D(K_inv_arr, R_inv_arr, T_arr,
						depth, pt2d);
					pts3D[k++] = pt3d;
				}

				// 计算sigma
				float sigma = this->GetSigmaOfPts3D(pts3D);

				// 3D点光滑
				const auto& normal = plane_normal_map[it->first];  // 3个元素
				const float* tangent_1 = &ei_vects_map[it->first][3];  // 3个元素
				const float* tangent_2 = &ei_vects_map[it->first][6];  // 3个元素
				for (int i = 0; i < (int)pts3D.size(); ++i)
				{
					const cv::Point3f& pt3d_1 = pts3D[i];

					// 光滑点云中每一个点
					cv::Point3f sum(0.0f, 0.0f, 0.0f);
					float sum_weight = 0.0f;
					for (int j = 0; j < (int)pts3D.size(); ++j)
					{
						const cv::Point3f& pt3d_2 = pts3D[j];
						if (pt3d_2 != pt3d_1)
						{
							float diff = fabs(pt3d_1.x - pt3d_2.x)
								+ fabs(pt3d_1.y - pt3d_2.y)
								+ fabs(pt3d_1.z - pt3d_2.z);
							float gauss_weight = expf(-0.5f*diff*diff / (sigma * sigma));

							cv::Point3f normal_component = (pt3d_2.x * normal[0]
								+ pt3d_2.y * normal[1]
								+ pt3d_2.z * normal[2]) \
								* cv::Point3f(normal[0], normal[1], normal[2]);

							cv::Point3f tagent_component = (pt3d_1.x * tangent_1[0]
								+ pt3d_1.y * tangent_1[1]
								+ pt3d_1.z * tangent_1[2]) \
								* cv::Point3f(tangent_1[0], tangent_1[1], tangent_1[2]) \
								+ (pt3d_1.x * tangent_2[0]
									+ pt3d_1.y * tangent_2[1]
									+ pt3d_1.z * tangent_2[2])
								* cv::Point3f(tangent_2[0], tangent_2[1], tangent_2[2]);

							sum += gauss_weight * normal_component + tagent_component;
							sum_weight += gauss_weight;
						}
					}

					// 计算光滑后的3D坐标
					if (sum_weight > 1e-8)
					{
						cv::Point3f pt3d_1_new = sum / sum_weight;
						pts3D_new[i] = pt3d_1_new;

						// 更新深度值
						float depth_new = this->GetDepthBy3dPtRT(R_arr, T_arr, pt3d_1_new);
						depth_map.Set((int)pts2D[i].y, (int)pts2D[i].x, depth_new);
					}
				}

				//// 测试smooth前后3D点云的方差变化
				//float var_arr_1[3] = { 0.0f }, var_arr_2[3] = { 0.0f };
				//this->GetPointCloudVariance(pts3D, var_arr_1);
				//this->GetPointCloudVariance(pts3D_new, var_arr_2);

				//if (var_arr_1[0] + var_arr_1[1] + var_arr_1[2] > var_arr_2[0] + var_arr_2[1] + var_arr_2[2])
				//{
				//	printf("Point cloud smoothed\n");
				//}
				//else
				//{
				//	printf("Point cloud not smoothed\n");
				//}
			}

			return 0;
		}

		// 相机坐标系
		int Workspace::SmoothPointCloudCam(const float* K_arr,
			const float* K_inv_arr,
			const std::vector<cv::Point2f>& pts2D,
			const float* normal,
			const float* tangent_1,
			const float* tangent_2,
			DepthMap& depth_map)
		{
			// 反投影到3D空间
			std::vector<cv::Point3f> pts3D(pts2D.size());

			// 存放光滑后的3D点
			std::vector<cv::Point3f> pts3D_new(pts2D.size());

			// 计算3D点云(相机坐标系)
			int k = 0;  // 计数
			for (auto pt2d : pts2D)
			{
				const float& depth = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
				//if (depth < 0.0f)
				//{
				//	printf("pause\n");
				//}
				cv::Point3f pt3d = this->m_model.BackProjTo3DCam(K_inv_arr, depth, pt2d);
				pts3D[k++] = pt3d;
			}

			// 确定sigma
			float sigma = this->GetSigmaOfPts3D(pts3D);

			// 3D点光滑
			for (int i = 0; i < (int)pts3D.size(); ++i)
			{
				const cv::Point3f& pt3d_1 = pts3D[i];

				// 光滑点云中每一个点
				cv::Point3f sum(0.0f, 0.0f, 0.0f);
				float sum_weight = 0.0f;
				for (int j = 0; j < (int)pts3D.size(); ++j)
				{
					const cv::Point3f& pt3d_2 = pts3D[j];
					if (pt3d_2 != pt3d_1)
					{
						float diff = fabs(pt3d_1.x - pt3d_2.x)
							+ fabs(pt3d_1.y - pt3d_2.y)
							+ fabs(pt3d_1.z - pt3d_2.z);
						float gauss_weight = expf(-0.5f*diff*diff / (sigma * sigma));

						cv::Point3f normal_component = (pt3d_2.x * normal[0]
							+ pt3d_2.y * normal[1]
							+ pt3d_2.z * normal[2]) \
							* cv::Point3f(normal[0], normal[1], normal[2]);

						cv::Point3f tagent_component = (pt3d_1.x * tangent_1[0]
							+ pt3d_1.y * tangent_1[1]
							+ pt3d_1.z * tangent_1[2]) \
							* cv::Point3f(tangent_1[0], tangent_1[1], tangent_1[2]) \
							+ (pt3d_1.x * tangent_2[0]
								+ pt3d_1.y * tangent_2[1]
								+ pt3d_1.z * tangent_2[2])
							* cv::Point3f(tangent_2[0], tangent_2[1], tangent_2[2]);

						sum += gauss_weight * normal_component + tagent_component;
						sum_weight += gauss_weight;
					}
				}

				// 计算光滑后的3D坐标
				if (sum_weight > 1e-8)
				{
					cv::Point3f pt3d_1_new = sum / sum_weight;
					pts3D_new[i] = pt3d_1_new;

					// 更新深度值
					float depth_new = this->GetDepthBy3dPtK(K_arr, pt3d_1_new);
					//if (depth_new < 0.0f)
					//{
					//	printf("pause\n");
					//}
					depth_map.Set((int)pts2D[i].y, (int)pts2D[i].x, depth_new);
				}
			}

			//// 测试smooth前后3D点云的方差变化
			//float var_arr_1[3] = { 0.0f }, var_arr_2[3] = { 0.0f };
			//this->GetPointCloudVariance(pts3D, var_arr_1);
			//this->GetPointCloudVariance(pts3D_new, var_arr_2);

			//if (var_arr_1[0] + var_arr_1[1] + var_arr_1[2] > var_arr_2[0] + var_arr_2[1] + var_arr_2[2])
			//{
			//	printf("Point cloud smoothed\n");
			//}
			//else
			//{
			//	printf("Point cloud not smoothed\n");
			//}

			return 0;
		}

		// 根据平面中心点(center), 2个切向向量确定平面的四个顶点, 进而确定2D搜索范围
		int Workspace::GetSearchRange(const cv::Point3f& center,
			const float* P,  // 相机矩阵数组
			const float* ei_vals,  // 特征值
			const float* ei_vects,  // 特征向量
			const float fold,  // 单边扩展倍数(默认3.0f)
			float* x_range, float* y_range)
		{
			// 计算四个顶点的3D坐标(顺时针)
			cv::Point3f vertex_1 = center
				- fold * sqrtf(ei_vals[1]) * cv::Point3f(ei_vects[3], ei_vects[4], ei_vects[5])
				- fold * sqrtf(ei_vals[2]) * cv::Point3f(ei_vects[6], ei_vects[7], ei_vects[8]);
			cv::Point3f vertex_2 = center
				+ fold * sqrtf(ei_vals[1]) * cv::Point3f(ei_vects[3], ei_vects[4], ei_vects[5])
				- fold * sqrtf(ei_vals[2]) * cv::Point3f(ei_vects[6], ei_vects[7], ei_vects[8]);
			cv::Point3f vertex_3 = center
				+ fold * sqrtf(ei_vals[1]) * cv::Point3f(ei_vects[3], ei_vects[4], ei_vects[5])
				+ fold * sqrtf(ei_vals[2]) * cv::Point3f(ei_vects[6], ei_vects[7], ei_vects[8]);
			cv::Point3f vertex_4 = center
				- fold * sqrtf(ei_vals[1]) * cv::Point3f(ei_vects[3], ei_vects[4], ei_vects[5])
				+ fold * sqrtf(ei_vals[2]) * cv::Point3f(ei_vects[6], ei_vects[7], ei_vects[8]);

			// 齐次坐标
			const Eigen::Vector4f vert_1_H(vertex_1.x, vertex_1.y, vertex_1.z, 1.0f);
			const Eigen::Vector4f vert_2_H(vertex_2.x, vertex_2.y, vertex_2.z, 1.0f);
			const Eigen::Vector4f vert_3_H(vertex_3.x, vertex_3.y, vertex_3.z, 1.0f);
			const Eigen::Vector4f vert_4_H(vertex_4.x, vertex_4.y, vertex_4.z, 1.0f);

			// 将四个顶点投影到2D坐标系
			const Eigen::Vector3f xyd_1 = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(P) * vert_1_H;
			const Eigen::Vector3f xyd_2 = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(P) * vert_2_H;
			const Eigen::Vector3f xyd_3 = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(P) * vert_3_H;
			const Eigen::Vector3f xyd_4 = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(P) * vert_4_H;

			// 2D坐标标准化
			std::vector<float> x_arr(4), y_arr(4);
			x_arr[0] = xyd_1(0) / xyd_1(2);  // x/z
			x_arr[1] = xyd_2(0) / xyd_2(2);
			x_arr[2] = xyd_3(0) / xyd_3(2);
			x_arr[3] = xyd_4(0) / xyd_4(2);

			y_arr[0] = xyd_1(1) / xyd_1(2);  // y/z
			y_arr[1] = xyd_2(1) / xyd_2(2);
			y_arr[2] = xyd_3(1) / xyd_3(2);
			y_arr[3] = xyd_4(1) / xyd_4(2);

			for (int i = 0; i < 4; ++i)
			{
				if (isnan(x_arr[i]) || isnan(y_arr[i]))
				{
					// 无效2d坐标
					memset(x_arr.data(), 0, sizeof(float) * 4);
					memset(y_arr.data(), 0, sizeof(float) * 4);

					return -1;
				}
			}

			// 计算x, y坐标范围
			auto min_x = std::min_element(std::begin(x_arr), std::end(x_arr));
			auto min_y = std::min_element(std::begin(y_arr), std::end(y_arr));

			auto max_x = std::max_element(std::begin(x_arr), std::end(x_arr));
			auto max_y = std::max_element(std::begin(y_arr), std::end(y_arr));

			x_range[0] = *min_x;
			x_range[1] = *max_x;
			y_range[0] = *min_y;
			y_range[1] = *max_y;

			return 0;
		}

		// 相机坐标系
		int Workspace::GetSearchRangeCam(const cv::Point3f& center,  // 相机坐标系
			const float* K_arr,
			const float* ei_vals,
			const float* ei_vects,
			const float fold,
			float* x_range, float* y_range)
		{
			// 计算四个顶点的3D坐标(相机坐标系)(顺时针)
			cv::Point3f vertex_1 = center
				- fold * sqrtf(ei_vals[1]) * cv::Point3f(ei_vects[3], ei_vects[4], ei_vects[5])
				- fold * sqrtf(ei_vals[2]) * cv::Point3f(ei_vects[6], ei_vects[7], ei_vects[8]);
			cv::Point3f vertex_2 = center
				+ fold * sqrtf(ei_vals[1]) * cv::Point3f(ei_vects[3], ei_vects[4], ei_vects[5])
				- fold * sqrtf(ei_vals[2]) * cv::Point3f(ei_vects[6], ei_vects[7], ei_vects[8]);
			cv::Point3f vertex_3 = center
				+ fold * sqrtf(ei_vals[1]) * cv::Point3f(ei_vects[3], ei_vects[4], ei_vects[5])
				+ fold * sqrtf(ei_vals[2]) * cv::Point3f(ei_vects[6], ei_vects[7], ei_vects[8]);
			cv::Point3f vertex_4 = center
				- fold * sqrtf(ei_vals[1]) * cv::Point3f(ei_vects[3], ei_vects[4], ei_vects[5])
				+ fold * sqrtf(ei_vals[2]) * cv::Point3f(ei_vects[6], ei_vects[7], ei_vects[8]);

			// opencv -> eigen
			const Eigen::Vector3f vert_1(vertex_1.x, vertex_1.y, vertex_1.z);
			const Eigen::Vector3f vert_2(vertex_2.x, vertex_2.y, vertex_2.z);
			const Eigen::Vector3f vert_3(vertex_3.x, vertex_3.y, vertex_3.z);
			const Eigen::Vector3f vert_4(vertex_4.x, vertex_4.y, vertex_4.z);

			// 将四个顶点投影到2D坐标系
			const Eigen::Vector3f xyd_1 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * vert_1;
			const Eigen::Vector3f xyd_2 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * vert_2;
			const Eigen::Vector3f xyd_3 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * vert_3;
			const Eigen::Vector3f xyd_4 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * vert_4;

			// 2D坐标标准化
			std::vector<float> x_arr(4), y_arr(4);
			x_arr[0] = xyd_1(0) / xyd_1(2);  // x/z
			x_arr[1] = xyd_2(0) / xyd_2(2);
			x_arr[2] = xyd_3(0) / xyd_3(2);
			x_arr[3] = xyd_4(0) / xyd_4(2);

			y_arr[0] = xyd_1(1) / xyd_1(2);  // y/z
			y_arr[1] = xyd_2(1) / xyd_2(2);
			y_arr[2] = xyd_3(1) / xyd_3(2);
			y_arr[3] = xyd_4(1) / xyd_4(2);

			for (int i = 0; i < 4; ++i)
			{
				if (isnan(x_arr[i]) || isnan(y_arr[i]))
				{
					// 无效2d坐标
					memset(x_arr.data(), 0, sizeof(float) * 4);
					memset(y_arr.data(), 0, sizeof(float) * 4);

					return -1;
				}
			}

			// 计算x, y坐标范围
			auto min_x = std::min_element(std::begin(x_arr), std::end(x_arr));
			auto min_y = std::min_element(std::begin(y_arr), std::end(y_arr));

			auto max_x = std::max_element(std::begin(x_arr), std::end(x_arr));
			auto max_y = std::max_element(std::begin(y_arr), std::end(y_arr));

			x_range[0] = *min_x;
			x_range[1] = *max_x;
			y_range[0] = *min_y;
			y_range[1] = *max_y;

			return 0;
		}

		int Workspace::ReadConsistencyGraph(const std::string& f_path)
		{
			ConsistencyGraph graph;
			graph.Read(f_path);
			return 0;
		}

		void Workspace::TestDepth1()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			//const string consistency_dir = this->options_.workspace_path \
			//	+ "/dense/stereo/consistency_graphs/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";

			// 总的视角个数
			int NumView = (int)this->m_model.m_images.size();

			//// 获取基于SFM的配对表
			//auto pair_map = this->GetImgPairMap();

			for (int img_id = 0; img_id < NumView; ++img_id)  // 注意: img_id != IMAGE_ID
			{
				//// 基于配对表计算平均基线距离
				//float MeanBaseline = this->GetMeanBaseline(img_id, pair_map);
				//if (0.0 == MeanBaseline)
				//{
				//	printf("[Warning]: found zero baseline\n");
				//	continue;
				//}

				string depth_f_name = GetFileName(img_id, true);
				string consistency_f_name(depth_f_name);

				string file_name(depth_f_name);
				StringReplace(file_name, string(".geometric.bin"), string(""));

				// modify bin depth file name
				StringReplace(depth_f_name, string(".geometric.bin"), string(".geometric_win5.bin"));

				DepthMap depth_map(this->depth_ranges_.at(img_id).first,
					this->depth_ranges_.at(img_id).second);
				depth_map.ReadBinary(depth_dir + depth_f_name);
				printf("%s read\n", depth_f_name.c_str());

				// ----------- Test output depth map for visualization
				string depth_map_path = depth_dir + depth_f_name + ".jpg";

				//imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				//printf("%s written\n", string(file_name + string(".jpg")).c_str());

				// ----------- speckle filtering using CV_32F data
				const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					/ 80.0f);
				const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
				const float maxDiff = 0.038f * depth_range;

				cv::Mat depth_mat = depth_map.Depth2Mat();

				// speckle filtering for depth_mat
				this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

				// fill depth_map
				depth_map.fillDepthWithMat(depth_mat);

				// write to disk for visualization
				string filter_name = file_name + "_filterSpecke_32F.jpg";
				depth_map_path = depth_dir + filter_name;
				cv::imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				printf("%s written\n", filter_name.c_str());

				// ----------- super-pixel segmentation
				cv::Mat src, mask, labels;

				// read color image
				src = cv::imread(src_dir + file_name, cv::IMREAD_COLOR);  // 原图读取BGR彩色图
				if (src.empty())
				{
					printf("[Err]: empty src image\n");
					return;
				}

				// ----- super-pixel segmentation using SEEDS or SLIC
				// SEEDS super-pixel segmentation
				const int num_superpixels = 700;  // 更多的初始分割保证边界
				Ptr<cv::ximgproc::SuperpixelSEEDS> superpixel = cv::ximgproc::createSuperpixelSEEDS(src.cols,
					src.rows,
					src.channels(),
					num_superpixels,  // num_superpixels
					15,  // num_levels: 5, 15
					2,
					5,
					true);
				superpixel->iterate(src);  // 迭代次数，默认为4
				superpixel->getLabels(labels);  // 获取labels
				superpixel->getLabelContourMask(mask);  // 获取超像素的边界

				//// 测试SLIC super-pixel segmentation
				//Ptr<cv::ximgproc::SuperpixelSLIC> superpixel = cv::ximgproc::createSuperpixelSLIC(src,
				//	101,
				//	50);
				//superpixel->iterate();  // 迭代次数，默认为10
				//superpixel->enforceLabelConnectivity();
				//superpixel->getLabelContourMask(mask);  // 获取超像素的边界
				//superpixel->getLabels(labels);  // 获取labels

				// 获取超像素的数量
				//int actual_number = superpixel->getNumberOfSuperpixels();  

				// construct 2 Hashmaps for each super-pixel
				std::unordered_map<int, std::vector<cv::Point2f>> label_map, has_depth_map, has_no_depth_map;

				// traverse each pxiel to put into hashmaps
				for (int y = 0; y < labels.rows; ++y)
				{
					for (int x = 0; x < labels.cols; ++x)
					{
						const int& label = labels.at<int>(y, x);

						// label -> 图像2D坐标点集
						label_map[label].push_back(cv::Point2f(float(x), float(y)));

						if (depth_mat.at<float>(y, x) > 0.0f)
						{
							has_depth_map[label].push_back(cv::Point2f(float(x), float(y)));
						}
						else
						{
							has_no_depth_map[label].push_back(cv::Point2f(float(x), float(y)));
						}
					}
				}

				// ----- 为原图绘制superpixel merge之前的mask
				cv::Mat src_1 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_1);
				string mask_src1_name = file_name + "_before_merge_mask.jpg";
				depth_map_path = depth_dir + mask_src1_name;
				cv::imwrite(depth_map_path, src_1);
				printf("%s written\n", mask_src1_name.c_str());

				// ----- superpixel合并(合并有效depth少于阈值的)
				//printf("Before merging, %d superpixels\n", has_depth_map.size());
				this->MergeSuperpixels(src, 500, labels, label_map, has_depth_map, has_no_depth_map);
				printf("After merging, %d superpixels\n", has_depth_map.size());

				// ----- 为原图绘制superpixel merge之后的mask
				cv::Mat src_2 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_2);
				string mask_src2_name = file_name + "_after_merge_mask.jpg";
				depth_map_path = depth_dir + mask_src2_name;
				cv::imwrite(depth_map_path, src_2);
				printf("%s written\n", mask_src2_name.c_str());

				// ----------- traverse each super-pixel for 3D plane fitting
				// project 2D points with depth back to 3D world coordinate
				const float* P_arr = this->m_model.m_images[img_id].GetP();
				//const float* K_arr = this->m_model.m_images[img_id].GetK();
				//const float* R_arr = this->m_model.m_images[img_id].GetR();
				const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();
				const float* R_inv_arr = this->m_model.m_images[img_id].GetInvR();
				const float* T_arr = this->m_model.m_images[img_id].GetT();

				//// 为当前图像(视角)读取consistency map...
				//string consistency_path = consistency_dir + consistency_f_name;
				//this->ReadConsistencyGraph(consistency_path);

				// ----- 遍历Merge后的superpixels, 平面拟合
				std::unordered_map<int, std::vector<float>> eigen_vals_map;  // superpixel的特征值
				std::unordered_map<int, std::vector<float>> eigen_vects_map;  // superpixel特征向量
				std::unordered_map<int, std::vector<float>> plane_normal_map;  // superpxiel的法向量
				std::unordered_map<int, std::vector<float>> plane_map;  // superpixel的切平面方程
				std::unordered_map<int, cv::Point3f> center_map;  // superpixel的中心点坐标
				this->FitPlaneForSuperpixels(depth_map,
					K_inv_arr, R_inv_arr, T_arr,
					label_map, has_depth_map,
					center_map, eigen_vals_map, eigen_vects_map,
					plane_normal_map, plane_map);
				printf("Superpixel plane fitting done\n");

				// ----- 修正tagent plane
				this->CorrectPCPlane(P_arr, K_inv_arr, R_inv_arr, T_arr,
					src.cols, src.rows,
					depth_range * 0.1f,
					3.0f,
					label_map,
					plane_map, center_map, plane_normal_map,
					eigen_vals_map, eigen_vects_map);
				printf("Superpixel plane correction done\n");

				// ----- superpixel连接: 连接可连接的相邻superpixel,
				//printf("Before connecting,  %d superpixels\n", has_depth_map.size());
				this->ConnectSuperpixels(0.0005f, 0.28f, depth_map,
					K_inv_arr, R_inv_arr, T_arr,
					eigen_vals_map, plane_normal_map, center_map,
					labels, label_map, has_depth_map, has_no_depth_map);
				printf("After connecting, %d superpixels\n", has_depth_map.size());

				// ----- 绘制Connect之后的mask
				cv::Mat src_3 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_3);
				string mask_src3_name = file_name + "_after_connect_mask.jpg";
				depth_map_path = depth_dir + mask_src3_name;
				cv::imwrite(depth_map_path, src_3);
				printf("%s written\n", mask_src3_name.c_str());

				// traver each super-pixel
				// 记录plane和non-plane编号, 便于debug
				std::vector<int> plane_ids, non_plane_ids;

				for (auto it = has_depth_map.begin();
					it != has_depth_map.end(); it++)
				{
					//printf("Processing superpixel %d\n", it->first);

					if ((int)it->second.size() < 3)
					{
						printf("[Warning]: Not enough valid depth within super-pixel %d\n", it->first);
						continue;
					}

					const std::vector<cv::Point2f>& Pts2D = it->second;

					// ----- calculate 3D world coordinates points
					std::vector<cv::Point3f> Pts3D(Pts2D.size());
					for (int i = 0; i < (int)Pts2D.size(); i++)
					{
						cv::Point3f Pt3D = this->m_model.BackProjTo3D(K_inv_arr,
							R_inv_arr,
							T_arr,
							depth_map.GetDepth((int)Pts2D[i].y, (int)Pts2D[i].x),
							Pts2D[i]);

						Pts3D[i] = Pt3D;
					}

					// ----- 3D plane fittin using RANSAC
					float plane_arr[4] = { 0.0f };
					if ((int)Pts3D.size() < 3)
					{
						printf("[Err]: less than 3 pts\n");
						continue;
					}
					else if (3 == (int)Pts3D.size())
					{
						// using 3 pts
						RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
						ransac.RunRansac(Pts3D);  // 估算ETH3D焦距~0.0204m(20.4mm)
						memcpy(plane_arr, ransac.m_plane, sizeof(float) * 4);
					}
					else if ((int)Pts3D.size() <= 5)
					{
						// using OLS(SVD or PCA)
						RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
						ransac.RunRansac(Pts3D);
						memcpy(plane_arr, ransac.m_plane, sizeof(float) * 4);
					}
					else
					{
						// using OLS(SVD or PCA)
						RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
						ransac.RunRansac(Pts3D);
						memcpy(plane_arr, ransac.m_plane, sizeof(float) * 4);
					}

					// ----- 分类: 平面与非平面
					// 平面使用平面约束, 非平面超像素内JBU+网格光滑
					if (this->IsPlaneSuperpixelCloud(Pts3D,
						plane_arr,
						eigen_vals_map[it->first],
						0.0002f,
						depth_range * 0.01f))  // modify Dist_TH?
					{
						//printf("Superpixel %d is a plane\n", it->first);
						plane_ids.push_back(it->first);

						// 平面约束求深度值
						float sigma_s = this->GetSigmaOfPts2D(label_map[it->first]);
						for (auto pt2D : has_no_depth_map[it->first])
						{
							float depth = this->GetDepthCoPlane(K_inv_arr, R_inv_arr, T_arr,
								plane_arr, pt2D);
							if (depth <= 0.0f)
							{
								depth = this->JBUSP(pt2D,
									has_depth_map[it->first], src, depth_map, sigma_s);
								//printf("Pixel[%d, %d] @Superpixel%d, depth: %.3f filled with JBU\n",
								//	(int)pt2D.x, (int)pt2D.y, it->first, depth);
							}

							// 更新depth
							depth_map.Set(pt2D.y, pt2D.x, depth);
						}
					}
					else
					{
						//printf("Superpixel %d is not a plane\n", it->first);
						non_plane_ids.push_back(it->first);

						float sigma_s = this->GetSigmaOfPts2D(label_map[it->first]);
						for (auto pt2D : has_no_depth_map[it->first])
						{
							// JBU插值计算深度值, 在同一个superpixel内部使用
							float depth = this->JBUSP(pt2D,
								has_depth_map[it->first], src, depth_map, sigma_s);

							// 更新depth
							depth_map.Set(pt2D.y, pt2D.x, depth);
						}
					}

					printf("Superpixel %d processed\n", it->first);
				}
				printf("Total %d planes, %d non-planes\n",
					(int)plane_ids.size(), (int)non_plane_ids.size());

				// 输出plane_ids和non_plane_ids
				ofstream f_out;
				string pl_id_out = depth_dir + file_name + string("plane_ids.txt");
				f_out.open(pl_id_out, ios_base::out);
				for (int id : plane_ids)
				{
					f_out << id << std::endl;
				}
				f_out.close();
				printf("%s written\n", pl_id_out.c_str());

				string non_pl_id_out = depth_dir + file_name + string("non_plane_ids.txt");
				f_out.open(non_pl_id_out, ios_base::out);
				for (int id : non_plane_ids)
				{
					f_out << id << std::endl;
				}
				f_out.close();
				printf("%s written\n", non_pl_id_out.c_str());

				// ----- 点云光滑
				//this->SmoothPointCloud(R_arr, T_arr, K_inv_arr, R_inv_arr, 0.1f,
				//	depth_map,
				//	label_map,
				//	plane_normal_map,
				//	eigen_vects_map);

				// ----- 输出Filled二进制深度图: 用于重建的最终深度图数据
				string filled_out_name = file_name + ".geometric.bin";
				depth_map_path = depth_dir + filled_out_name;
				depth_map.WriteBinary(depth_map_path);
				printf("%s written\n", filled_out_name.c_str());

				// 输出Filled深度图用于可视化[0, 255]
				string filled_name = file_name + "_filled.jpg";
				depth_map_path = depth_dir + filled_name;
				cv::imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				printf("%s written\n", filled_name.c_str());

				// ----- 为Filled深度图绘制mask, 输出深度图+super-pixel mask 用于可视化[0, 255]
				cv::Mat depth_filled = cv::imread(depth_map_path, cv::IMREAD_COLOR);
				this->DrawMaskOfSuperpixels(labels, depth_filled);

				// 绘制superpixel编号
				for (auto it = has_depth_map.begin(); it != has_depth_map.end(); ++it)
				{
					cv::Point2f center_2d(0.0f, 0.0f);
					for (auto pt2d : has_depth_map[it->first])
					{
						center_2d.x += pt2d.x;
						center_2d.y += pt2d.y;
					}
					center_2d.x /= float(has_depth_map[it->first].size());
					center_2d.y /= float(has_depth_map[it->first].size());

					cv::putText(depth_filled, std::to_string(it->first),
						cv::Point((int)center_2d.x, (int)center_2d.y),
						cv::FONT_HERSHEY_SIMPLEX,
						1.0,
						cv::Scalar(255, 0, 0));
				}
				string filled_mask_name = file_name + "_filled_mask.jpg";
				depth_map_path = depth_dir + filled_mask_name;
				cv::imwrite(depth_map_path, depth_filled);
				printf("%s written\n\n", filled_mask_name.c_str());
			}
		}

		void Workspace::TestDepth5()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";

			// 总的视角个数
			int NumView = (int)this->m_model.m_images.size();

			for (int img_id = 0; img_id < NumView; ++img_id)  // 注意: img_id != IMAGE_ID
			{
				string depth_f_name = GetFileName(img_id, true);

				string file_name(depth_f_name);
				StringReplace(file_name, string(".geometric.bin"), string(""));

				// modify bin depth file name
				StringReplace(depth_f_name, string(".geometric.bin"), string(".geometric_win5.bin"));

				DepthMap depth_map(this->depth_ranges_.at(img_id).first,
					this->depth_ranges_.at(img_id).second);
				depth_map.ReadBinary(depth_dir + depth_f_name);
				printf("%s read\n", depth_f_name.c_str());

				// ----------- Test output depth map for visualization
				string depth_map_path = depth_dir + depth_f_name + ".jpg";

				//imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				//printf("%s written\n", string(file_name + string(".jpg")).c_str());

				// ----------- speckle filtering using CV_32F data
				const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					/ 80.0f);
				const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
				const float maxDiff = 0.038f * depth_range;

				cv::Mat depth_mat = depth_map.Depth2Mat();

				// speckle filtering for depth_mat
				this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

				// fill depth_map
				depth_map.fillDepthWithMat(depth_mat);

				// write to disk for visualization
				string filter_name = file_name + "_filterSpecke_32F.jpg";
				depth_map_path = depth_dir + filter_name;
				cv::imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				printf("%s written\n", filter_name.c_str());

				// ----------- super-pixel segmentation
				cv::Mat src, mask, labels;

				// read color image
				src = cv::imread(src_dir + file_name, cv::IMREAD_COLOR);  // 原图读取BGR彩色图
				if (src.empty())
				{
					printf("[Err]: empty src image\n");
					return;
				}

				// ----- super-pixel segmentation using SEEDS or SLIC
				// SEEDS super-pixel segmentation
				const int num_superpixels = 700;  // 更多的初始分割保证边界
				Ptr<cv::ximgproc::SuperpixelSEEDS> superpixel = cv::ximgproc::createSuperpixelSEEDS(src.cols,
					src.rows,
					src.channels(),
					num_superpixels,  // num_superpixels
					15,  // num_levels: 5, 15
					2,
					5,
					true);
				superpixel->iterate(src);  // 迭代次数，默认为4
				superpixel->getLabels(labels);  // 获取labels
				superpixel->getLabelContourMask(mask);  // 获取超像素的边界

				// construct 2 Hashmaps for each super-pixel
				std::unordered_map<int, std::vector<cv::Point2f>> label_map, has_depth_map, has_no_depth_map;

				// traverse each pxiel to put into hashmaps
				for (int y = 0; y < labels.rows; ++y)
				{
					for (int x = 0; x < labels.cols; ++x)
					{
						const int& label = labels.at<int>(y, x);

						// label -> 图像2D坐标点集
						label_map[label].push_back(cv::Point2f(float(x), float(y)));

						if (depth_mat.at<float>(y, x) > 0.0f)
						{
							has_depth_map[label].push_back(cv::Point2f(float(x), float(y)));
						}
						else
						{
							has_no_depth_map[label].push_back(cv::Point2f(float(x), float(y)));
						}
					}
				}

				// ----- 为原图绘制superpixel merge之前的mask
				cv::Mat src_1 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_1);
				string mask_src1_name = file_name + "_before_merge_mask.jpg";
				depth_map_path = depth_dir + mask_src1_name;
				cv::imwrite(depth_map_path, src_1);
				printf("%s written\n", mask_src1_name.c_str());

				// ----- superpixel合并(合并有效depth少于阈值的)
				printf("Before merging, %d superpixels\n", has_depth_map.size());
				this->MergeSuperpixels(src, 500, labels, label_map, has_depth_map, has_no_depth_map);
				printf("After merging, %d superpixels\n", has_depth_map.size());

				// ----- 为原图绘制superpixel merge之后的mask
				cv::Mat src_2 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_2);
				string mask_src2_name = file_name + "_after_merge_mask.jpg";
				depth_map_path = depth_dir + mask_src2_name;
				cv::imwrite(depth_map_path, src_2);
				printf("%s written\n", mask_src2_name.c_str());

				// ----------- traverse each super-pixel for 3D plane fitting
				// project 2D points with depth back to 3D world coordinate
				const float* K_arr = this->m_model.m_images[img_id].GetK();
				const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();

				//// 为当前图像(视角)读取consistency map...
				//string consistency_path = consistency_dir + consistency_f_name;
				//this->ReadConsistencyGraph(consistency_path);

				// ----- 遍历Merge后的superpixels, 平面拟合
				std::unordered_map<int, std::vector<float>> eigen_vals_map;  // superpixel的特征值
				std::unordered_map<int, std::vector<float>> eigen_vects_map;  // superpixel特征向量
				std::unordered_map<int, std::vector<float>> plane_normal_map;  // superpxiel的法向量
				std::unordered_map<int, std::vector<float>> plane_map;  // superpixel的切平面方程
				std::unordered_map<int, cv::Point3f> center_map;  // superpixel的中心点坐标
				this->FitPlaneForSPsCam(depth_map,
					K_inv_arr,
					label_map, has_depth_map,
					center_map, eigen_vals_map, eigen_vects_map,
					plane_normal_map, plane_map);
				printf("Superpixel plane fitting done\n");

				// ----- 修正tagent plane
				this->CorrectPlaneCam(K_arr, K_inv_arr,
					src.cols, src.rows,
					depth_range * 0.1f,
					3.0f, 5,
					label_map,
					plane_map, center_map, plane_normal_map,
					eigen_vals_map, eigen_vects_map);
				printf("Superpixel plane correction done\n");

				// ----- superpixel连接: 连接可连接的相邻superpixel,
				//printf("Before connecting,  %d superpixels\n", has_depth_map.size());
				this->ConnectSuperpixelsCam(0.0005f, 0.28f, depth_map,
					K_inv_arr,
					plane_map,
					eigen_vals_map, plane_normal_map, center_map,
					labels, label_map, has_depth_map, has_no_depth_map);
				printf("After connecting, %d superpixels\n", has_depth_map.size());

				// ----- 绘制Connect之后的mask
				cv::Mat src_3 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_3);
				string mask_src3_name = file_name + "_after_connect_mask.jpg";
				depth_map_path = depth_dir + mask_src3_name;
				cv::imwrite(depth_map_path, src_3);
				printf("%s written\n", mask_src3_name.c_str());

				// traver each super-pixel
				// 记录plane和non-plane编号, 便于debug
				std::vector<int> plane_ids, non_plane_ids;

				for (auto it = has_depth_map.begin();
					it != has_depth_map.end(); it++)
				{
					//printf("Processing superpixel %d\n", it->first);

					if ((int)it->second.size() < 3)
					{
						printf("[Warning]: Not enough valid depth within super-pixel %d\n", it->first);
						continue;
					}

					const std::vector<cv::Point2f>& Pts2D = it->second;

					// ----- calculate 3D world coordinates points
					std::vector<cv::Point3f> Pts3D;
					Pts3D.reserve(Pts2D.size());
					Pts3D.resize(Pts2D.size());

					for (int i = 0; i < (int)Pts2D.size(); i++)
					{
						cv::Point3f Pt3D = this->m_model.BackProjTo3DCam(K_inv_arr,
							depth_map.GetDepth((int)Pts2D[i].y, (int)Pts2D[i].x),
							Pts2D[i]);

						Pts3D[i] = Pt3D;
					}

					// ----- 3D plane fittin using RANSAC
					float plane_arr[4] = { 0.0f };
					if ((int)Pts3D.size() < 3)
					{
						printf("[Err]: less than 3 pts\n");
						continue;
					}
					else if (3 == (int)Pts3D.size())
					{
						// using 3 pts
						RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
						ransac.RunRansac(Pts3D);  // 估算ETH3D焦距~0.0204m(20.4mm)
						memcpy(plane_arr, ransac.m_plane, sizeof(float) * 4);
					}
					else if ((int)Pts3D.size() <= 5)
					{
						// using OLS(SVD or PCA)
						RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
						ransac.RunRansac(Pts3D);
						memcpy(plane_arr, ransac.m_plane, sizeof(float) * 4);
					}
					else
					{
						// using OLS(SVD or PCA)
						RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
						ransac.RunRansac(Pts3D);
						memcpy(plane_arr, ransac.m_plane, sizeof(float) * 4);
					}

					// ----- 分类: 平面与非平面
					// 平面使用平面约束, 非平面超像素内JBU+网格光滑
					if (this->IsPlaneSuperpixelCloud(Pts3D,
						plane_arr,
						eigen_vals_map[it->first],
						0.0002f,
						depth_range * 0.01f))  // modify Dist_TH?
					{
						//printf("Superpixel %d is a plane\n", it->first);
						plane_ids.push_back(it->first);

						// 平面约束求深度值
						float sigma_s = this->GetSigmaOfPts2D(label_map[it->first]);
						for (auto pt2D : has_no_depth_map[it->first])
						{
							float depth = this->GetDepthCoPlaneCam(K_inv_arr,
								plane_arr, pt2D);
							if (depth <= 0.0f)
							{
								depth = this->JBUSP(pt2D,
									has_depth_map[it->first], src, depth_map, sigma_s);
								//printf("Pixel[%d, %d] @Superpixel%d, depth: %.3f filled with JBU\n",
								//	(int)pt2D.x, (int)pt2D.y, it->first, depth);
							}

							// 更新depth
							depth_map.Set(pt2D.y, pt2D.x, depth);
						}

						printf("Superpixel %d processed, plane\n", it->first);

					}
					else
					{
						//printf("Superpixel %d is not a plane\n", it->first);
						non_plane_ids.push_back(it->first);

						float sigma_s = this->GetSigmaOfPts2D(label_map[it->first]);
						for (auto pt2D : has_no_depth_map[it->first])
						{
							// JBU插值计算深度值, 在同一个superpixel内部使用
							float depth = this->JBUSP(pt2D,
								has_depth_map[it->first], src, depth_map, sigma_s);

							// 更新depth
							depth_map.Set(pt2D.y, pt2D.x, depth);
						}

						// 点云光滑
						this->SmoothPointCloudCam(K_arr, K_inv_arr,
							label_map[it->first],
							plane_normal_map[it->first].data(),
							eigen_vects_map[it->first].data() + 3,
							eigen_vects_map[it->first].data() + 6,
							depth_map);

						printf("Superpixel %d processed, non-plane\n", it->first);

					}

				}
				printf("Total %d planes, %d non-planes\n",
					(int)plane_ids.size(), (int)non_plane_ids.size());

				// 输出plane_ids和non_plane_ids
				ofstream f_out;
				string pl_id_out = depth_dir + file_name + string("plane_ids.txt");
				f_out.open(pl_id_out, ios_base::out);
				for (int id : plane_ids)
				{
					f_out << id << std::endl;
				}
				f_out.close();
				printf("%s written\n", pl_id_out.c_str());

				string non_pl_id_out = depth_dir + file_name + string("non_plane_ids.txt");
				f_out.open(non_pl_id_out, ios_base::out);
				for (int id : non_plane_ids)
				{
					f_out << id << std::endl;
				}
				f_out.close();
				printf("%s written\n", non_pl_id_out.c_str());

				// ----- 输出Filled二进制深度图: 用于重建的最终深度图数据
				string filled_out_name = file_name + ".geometric.bin";
				depth_map_path = depth_dir + filled_out_name;
				depth_map.WriteBinary(depth_map_path);
				printf("%s written\n", filled_out_name.c_str());

				// 输出Filled深度图用于可视化[0, 255]
				string filled_name = file_name + "_filled.jpg";
				depth_map_path = depth_dir + filled_name;
				cv::imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				printf("%s written\n", filled_name.c_str());

				// ----- 为Filled深度图绘制mask, 输出深度图+super-pixel mask 用于可视化[0, 255]
				cv::Mat depth_filled = cv::imread(depth_map_path, cv::IMREAD_COLOR);
				this->DrawMaskOfSuperpixels(labels, depth_filled);

				// 绘制superpixel编号
				for (auto it = has_depth_map.begin(); it != has_depth_map.end(); ++it)
				{
					cv::Point2f center_2d(0.0f, 0.0f);
					for (auto pt2d : has_depth_map[it->first])
					{
						center_2d.x += pt2d.x;
						center_2d.y += pt2d.y;
					}
					center_2d.x /= float(has_depth_map[it->first].size());
					center_2d.y /= float(has_depth_map[it->first].size());

					cv::putText(depth_filled, std::to_string(it->first),
						cv::Point((int)center_2d.x, (int)center_2d.y),
						cv::FONT_HERSHEY_SIMPLEX,
						1.0,
						cv::Scalar(255, 0, 0));
				}
				string filled_mask_name = file_name + "_filled_mask.jpg";
				depth_map_path = depth_dir + filled_mask_name;
				cv::imwrite(depth_map_path, depth_filled);
				printf("%s written\n\n", filled_mask_name.c_str());
			}
		}

		// 计算MRF能量
		float Workspace::GetEnergyOfPt2d(const cv::Point2f& Pt2D,
			const int SP_Label,
			std::unordered_map<cv::Point2f, int, HashFuncPt2f, EuqualPt2f>& pt2d2SPLabel,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map,
			std::unordered_map<int, std::vector<float>>& plane_map,
			const DepthMap& depth_map,
			const float* K_inv_arr,
			const float beta,
			const int radius)
		{
			// ----- 计算二元能量项
			int y_begin = Pt2D.y - radius;
			y_begin = y_begin >= 0 ? y_begin : 0;
			int y_end = Pt2D.y + radius;
			y_end = y_end <= depth_map.GetHeight() - 1 ? y_end : depth_map.GetHeight() - 1;

			int x_begin = Pt2D.x - radius;
			x_begin = x_begin >= 0 ? x_begin : 0;
			int x_end = Pt2D.x + radius;
			x_end = x_end <= depth_map.GetWidth() - 1 ? x_end : depth_map.GetWidth() - 1;

			float energy_binary = 0.0f;
			for (int y = y_begin; y <= y_end; ++y)
			{
				for (int x = x_begin; x <= x_end; ++x)
				{
					if (SP_Label == pt2d2SPLabel[cv::Point2f(x, y)])
					{
						energy_binary += -beta;
					}
					else
					{
						energy_binary += beta;
					}
				}
			}
			energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

			// ----- 计算一元能量项
			float depth_mean = 0.0f, depth_std = 0.0f;
			const int pts2d_size = int(has_depth_map[SP_Label].size()
				+ has_no_depth_map[SP_Label].size());
			std::vector<float> depths_(pts2d_size);
			// 有深度值的2d点
			const std::vector<cv::Point2f>& has_depth_pts = has_depth_map[SP_Label];
			for (int i = 0; i < (int)has_depth_pts.size(); ++i)
			{
				const cv::Point2f& pt2d = has_depth_pts[i];
				depths_[i] = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
			}
			// 没有深度值的2d点: 平面约束插值
			const std::vector<cv::Point2f>& has_no_depth_pts = has_no_depth_map[SP_Label];
			for (int i = 0; i < (int)has_no_depth_pts.size(); ++i)
			{
				const cv::Point2f& pt2d = has_no_depth_pts[i];
				depths_[i + (int)has_depth_pts.size()] = this->GetDepthCoPlaneCam(K_inv_arr,
					plane_map[pt2d2SPLabel[pt2d]].data(),
					pt2d);
			}

			depth_mean = std::accumulate(depths_.begin(), depths_.end(), 0.0f)
				/ float(pts2d_size);

			// 统计标准差
			for (auto depth : depths_)
			{
				depth_std += (depth - depth_mean) * (depth - depth_mean);
			}
			depth_std /= float(pts2d_size);
			depth_std = sqrtf(depth_std);

			const float& the_depth = depth_map.GetDepth(int(Pt2D.y), int(Pt2D.x));
			float energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
				+ 0.5f * (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std);

			return energy_binary + energy_unary;
		}

		void Workspace::ThFuncPts2dEnergy(const std::vector<cv::Point2f>& Pts2D,
			const int Start, const int End, const int PlaneCount,
			const std::unordered_map<int, int>& PlaneID2SPLabel,
			const std::unordered_map<int, std::vector<cv::Point2f>>& Label2Pts2d,
			const cv::Mat& labels,
			const DepthMap& depth_map,
			const float beta,
			const int radius,
			std::vector<int>& Pt2DSPLabelsRet)
		{
			printf("Start: %d, End: %d\n ", Start, End);

			for (int pt_i = Start; pt_i < End; ++pt_i)
			{
				// 处理一个2D点
				const cv::Point2f& Pt2D = Pts2D[pt_i];

				// 计算二元项搜索范围: 中心点确定了也就确定了搜索范围
				int y_begin = Pt2D.y - radius;
				y_begin = y_begin >= 0 ? y_begin : 0;
				int y_end = Pt2D.y + radius;
				y_end = y_end <= depth_map.GetHeight() - 1 ? y_end : depth_map.GetHeight() - 1;

				int x_begin = Pt2D.x - radius;
				x_begin = x_begin >= 0 ? x_begin : 0;
				int x_end = Pt2D.x + radius;
				x_end = x_end <= depth_map.GetWidth() - 1 ? x_end : depth_map.GetWidth() - 1;

				// 遍历所有label(plane), 找出能量最小的plane
				float energy_min = FLT_MAX;
				int best_pl_id = -1;
				for (int pl_i = 0; pl_i < PlaneCount; ++pl_i)
				{
					const int& SP_Label = PlaneID2SPLabel.at(pl_i);

					// ----- 计算二元能量项
					float energy_binary = 0.0f;
					for (int y = y_begin; y <= y_end; ++y)
					{
						for (int x = x_begin; x <= x_end; ++x)
						{
							if (SP_Label == labels.at<int>((int)Pt2D.y, (int)Pt2D.x))
							{
								energy_binary += -beta;
							}
							else
							{
								energy_binary += beta;
							}
						}
					}
					energy_binary /= float((y_end - y_begin + 1) * (x_end - x_begin + 1));

					// ----- 计算一元能量项
					float depth_mean = 0.0f, depth_std = 0.0f;
					const size_t pts2d_size = Label2Pts2d.at(SP_Label).size();
					std::vector<float> depths_(pts2d_size);
					const std::vector<cv::Point2f>& pts2d = Label2Pts2d.at(SP_Label);

					// 获取该label对应的深度值
					for (int i = 0; i < (int)pts2d.size(); ++i)
					{
						const cv::Point2f& pt2d = pts2d[i];
						depths_[i] = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
					}
					depth_mean = std::accumulate(depths_.begin(), depths_.end(), 0.0f)
						/ float(pts2d_size);

					// 统计标准差
					for (auto depth : depths_)
					{
						depth_std += (depth - depth_mean) * (depth - depth_mean);
					}
					depth_std /= float(pts2d_size);
					depth_std = sqrtf(depth_std);

					const float& the_depth = depth_map.GetDepth(int(Pt2D.y), int(Pt2D.x));
					float energy_unary = log2f(sqrtf(6.28f)*depth_std)  // 2.0f*3.14f
						+ 0.5f * (the_depth - depth_mean)*(the_depth - depth_mean) / (depth_std*depth_std);

					const float energy = energy_binary + energy_unary;
					if (energy < energy_min)
					{
						energy_min = energy;
						best_pl_id = pl_i;
					}
				}

				// 返回label: 可能更新过, 也可能没更新
				const int& label = PlaneID2SPLabel.at(best_pl_id);
				Pt2DSPLabelsRet[pt_i] = label;

				// 输出更新的label: 每1000次
				const int old_sp_label = labels.at<int>((int)Pt2D.y, (int)Pt2D.x);
				if (old_sp_label != PlaneID2SPLabel.at(best_pl_id))
				{
					if (0 == pt_i % 1000)
					{
						std::cout << Pt2D << ", SP Label " << old_sp_label
							<< " updated with " << label << std::endl;
					}
				}

				//std::printf("No depth Idx %d processed\n");
			}

			std::printf("Total %d Pts2D processed\n", End - Start);
		}

		// 相机坐标系
		void Workspace::TestDepth6()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";

			// 总的视角个数
			const int NumView = (int)this->m_model.m_images.size();
			for (int img_id = 0; img_id < NumView; ++img_id)  // 注意: img_id != IMAGE_ID
			{
				string depth_f_name = GetFileName(img_id, true);

				string file_name(depth_f_name);
				StringReplace(file_name, string(".geometric.bin"), string(""));

				// modify bin depth file name
				StringReplace(depth_f_name, string(".geometric.bin"), string(".geometric_win5.bin"));

				DepthMap depth_map(this->depth_ranges_.at(img_id).first,
					this->depth_ranges_.at(img_id).second);
				depth_map.ReadBinary(depth_dir + depth_f_name);
				std::printf("%s read\n", depth_f_name.c_str());

				// ----------- Test output depth map for visualization
				string viz_out_path = depth_dir + depth_f_name + ".jpg";

				float denominator = 50.0f, ratio = 0.08;  // 调参
#ifdef MRF_GPU
				for (int iter_i = 0; iter_i < NUM_MRF_ITER; ++iter_i)
#else
				for (int iter_i = 0; iter_i < NUM_ITER; ++iter_i)
#endif
				{
					// ----- 过滤斑点
					const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
						/ denominator);  // 80
					const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
					const float maxDiff = ratio * depth_range;  // 0.038

					cv::Mat depth_mat = depth_map.Depth2Mat();

					// speckle filtering for depth_mat
					this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

					// fill depth_map
					depth_map.fillDepthWithMat(depth_mat);  // 此处对depth_map进行了写入操作!!!

					// write to disk for visualization
					string filter_name = file_name + "_filtered.jpg";
					string filter_path = depth_dir + filter_name;
					cv::imwrite(filter_path, depth_map.ToBitmapGray(2, 98));
					std::printf("%s written\n", filter_name.c_str());

					// ----------- super-pixel segmentation
					cv::Mat src, mask, labels;

					// 原图读取BGR彩色图
					src = cv::imread(src_dir + file_name, cv::IMREAD_COLOR);
					if (src.empty())
					{
						std::printf("[Err]: empty src image\n");
						return;
					}

					// ----- super-pixel segmentation using SEEDS or SLIC
					// SEEDS super-pixel segmentation
					const int num_superpixels = 1000;  // 更多的初始分割保证边界
					Ptr<cv::ximgproc::SuperpixelSEEDS> superpixel = cv::ximgproc::createSuperpixelSEEDS(src.cols,
						src.rows,
						src.channels(),
						num_superpixels,  // num_superpixels
						15,  // num_levels: 5, 15
						2,
						5,
						true);
					superpixel->iterate(src);  // 迭代次数，默认为4
					superpixel->getLabels(labels);  // 获取labels
					superpixel->getLabelContourMask(mask);  // 获取超像素的边界

					//Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(src,
					//	101, 25);
					//slic->iterate();  // 迭代次数，默认为10
					//slic->enforceLabelConnectivity();
					//slic->getLabelContourMask(mask);  // 获取超像素的边界
					//slic->getLabels(labels);  // 获取labels

					// construct 2 Hashmaps for each super-pixel
					std::unordered_map<int, std::vector<cv::Point2f>> Label2Pts2d,
						has_depth_map, has_no_depth_map;

					// traverse each pxiel to put into hashmaps
					for (int y = 0; y < labels.rows; ++y)
					{
						for (int x = 0; x < labels.cols; ++x)
						{
							const int& label = labels.at<int>(y, x);

							// label -> 图像2D坐标点集
							Label2Pts2d[label].push_back(cv::Point2f(float(x), float(y)));

							if (depth_mat.at<float>(y, x) > 0.0f)
							{
								has_depth_map[label].push_back(cv::Point2f(float(x), float(y)));
							}
							else
							{
								has_no_depth_map[label].push_back(cv::Point2f(float(x), float(y)));
							}
						}
					}

#ifdef DRAW_MASK
					// ----- 为原图绘制superpixel merge之前的mask
					cv::Mat src_1 = src.clone();
					this->DrawMaskOfSuperpixels(labels, src_1);
					string mask_src1_name = file_name + "_before_merge_mask.jpg";
					viz_out_path = depth_dir + mask_src1_name;
					cv::imwrite(viz_out_path, src_1);
					std::printf("%s written\n", mask_src1_name.c_str());
#endif // DRAW_MASK

					// ----- superpixel合并(合并有效depth少于阈值的)
					std::printf("Before merging, %d superpixels\n", has_depth_map.size());
					this->MergeSuperpixels(src,
						4000,  // 2000
						labels,
						Label2Pts2d, has_depth_map, has_no_depth_map);
					std::printf("After merging, %d superpixels\n", has_depth_map.size());

#ifdef DRAW_MASK
					// ----- 为原图绘制superpixel merge之后的mask
					cv::Mat src_2 = src.clone();
					this->DrawMaskOfSuperpixels(labels, src_2);
					string mask_src2_name = file_name + "_after_merge_mask.jpg";
					viz_out_path = depth_dir + mask_src2_name;
					cv::imwrite(viz_out_path, src_2);
					std::printf("%s written\n", mask_src2_name.c_str());
#endif // DRAW_MASK

					// ----------- traverse each super-pixel for 3D plane fitting
					// project 2D points with depth back to 3D world coordinate
					const float* K_arr = this->m_model.m_images[img_id].GetK();
					const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();

					// ----- 遍历Merge后的superpixels, 平面拟合
					std::unordered_map<int, std::vector<float>> eigen_vals_map;  // superpixel的特征值
					std::unordered_map<int, std::vector<float>> eigen_vects_map;  // superpixel特征向量
					std::unordered_map<int, std::vector<float>> plane_normal_map;  // superpxiel的法向量
					std::unordered_map<int, std::vector<float>> plane_map;  // superpixel的切平面方程
					std::unordered_map<int, cv::Point3f> center_map;  // superpixel的中心点坐标
					this->FitPlaneForSPsCam(depth_map,
						K_inv_arr,
						Label2Pts2d, has_depth_map,
						center_map, eigen_vals_map, eigen_vects_map,
						plane_normal_map, plane_map);
					std::printf("Superpixel plane fitting done\n");

					// ----- 修正tagent plane
					this->CorrectPlaneCam(K_arr, K_inv_arr,
						src.cols, src.rows,
						depth_range * 0.01f,  // 调参: distance threshold
						3.0f, 5,  // 调参: fold, minimum neighbor number
						Label2Pts2d,
						plane_map, center_map, plane_normal_map,
						eigen_vals_map, eigen_vects_map);
					std::printf("Superpixel plane correction done\n");

					// ----- superpixel连接: 连接可连接的相邻superpixel,
					this->ConnectSuperpixelsCam(0.0005f, 0.15f, depth_map,
						K_inv_arr,
						plane_map,
						eigen_vals_map, plane_normal_map, center_map,
						labels, Label2Pts2d, has_depth_map, has_no_depth_map);
					std::printf("%d superpixels after connection\n", has_depth_map.size());

#ifdef DRAW_MASK
					// ----- 为原图绘制Connect之后的mask
					cv::Mat src_3 = src.clone();
					this->DrawMaskOfSuperpixels(labels, src_3);
					string mask_src3_name = file_name + "_after_connect_mask.jpg";
					viz_out_path = depth_dir + mask_src3_name;
					cv::imwrite(viz_out_path, src_3);
					printf("%s written\n", mask_src3_name.c_str());

					// ----- 在填充depth_map之前, 为depth_map绘制mask(用于算法分析)
					cv::Mat depth_mask = cv::imread(filter_path, cv::IMREAD_COLOR);
					this->DrawMaskOfSuperpixels(labels, depth_mask);
					string filter_mask = file_name + "_depth_filter_mask.jpg";
					viz_out_path = depth_dir + filter_mask;
					cv::imwrite(viz_out_path, depth_mask);
					std::printf("%s written\n", filter_mask.c_str());
#endif // DRAW_MASK


					//// ----- CPU端, 初始化depth_map无深度值2D点...
					//for (auto it = has_no_depth_map.begin();
					//	it != has_no_depth_map.end(); ++it)
					//{
					//	// 计算space_sigma
					//	float sigma_s = this->GetSigmaOfPts2D(Label2Pts2d[it->first]);
					//	for (auto pt2D : has_no_depth_map[it->first])
					//	{
					//		// 平面约束求深度值
					//		float depth = this->GetDepthCoPlaneCam(K_inv_arr, 
					//			plane_map[it->first].data(), pt2D);
					//		if (depth <= 0.0f)
					//		{
					//			// 深度值不正确, 通过JBU插值重新计算深度值
					//			depth = this->JBUSP(pt2D,
					//				has_depth_map[it->first], src, depth_map, sigma_s);
					//			//printf("Pixel[%d, %d] @Superpixel%d, depth: %.3f filled with JBU\n",
					//			//	(int)pt2D.x, (int)pt2D.y, it->first, depth);
					//		}
					//		// 更新depth
					//		depth_map.Set(pt2D.y, pt2D.x, depth);
					//	}
					//}
					//std::printf("Init depth_map done\n");

					// ----- 接下来步骤的共用计数参数
					int stride = 0, sp_count = 0, pt_count = 0;

					// ----- GPU端初始化depth_map, JBUSPGPU并行
					// ---- 构建GPU输入输出
					// 1. ---更新具有正常深度值的pt2d, 将depth < 0的放入一个数组
					// 2. ---统计depth<0的pt2d点的信息:
					// pts2d_has_no_depth_jbu, sp_labels_idx_jbu
					std::vector<cv::Point2f> pts2d_no_depth_jbu;  // 所有待处理的pt2d点
					std::vector<int> sp_labels_idx_jbu;  // 每个点的sp label index

					// ---JBUSPGPU的准备工作包括4个数组的初始化: 
					// (1). sp_labels_jbu, (2). pts2d_has_depth_jbu
					// (3). sp_has_depth_pt2ds_num (4). sigmas_s_jbu
					// pts2d_has_depth_jbu,
					// 用于GPU端JBUSP的sp label数组
					std::vector<int> sp_labels_jbu(has_no_depth_map.size(), 0);

					// ---统计有深度值pt2d点总共的点数
					pt_count = 0;
					for (auto it = has_no_depth_map.begin();
						it != has_no_depth_map.end(); ++it)
					{
						pt_count += (int)has_depth_map.at(it->first).size();
					}

					// 所有label对应的pt2d点数组: 依据has_no_depth_map的label顺序
					std::vector<cv::Point2f> pts2d_has_depth_jbu(pt_count);

					// 每个label对应的pt2d点个数
					std::vector<int> sp_has_depth_pt2ds_num(has_no_depth_map.size());

					// 构架sigma_s数组: 每个label都对应一个sigma_s值
					std::vector<float> sigmas_s_jbu(has_no_depth_map.size());

					stride = 0, sp_count = 0;  // sp_count即has_no_depth_map的label_idx
					for (auto it = has_no_depth_map.begin();
						it != has_no_depth_map.end(); ++it)
					{
						// 构建sp_labels_jbu
						sp_labels_jbu[sp_count] = it->first;

						// 构建pts2d_has_depth_jbu
						memcpy(pts2d_has_depth_jbu.data() + stride,
							has_depth_map[it->first].data(),
							sizeof(cv::Point2f) * has_depth_map[it->first].size());

						// 构建sp_has_depth_pt2ds_num
						sp_has_depth_pt2ds_num[sp_count] = int(has_depth_map[it->first].size());

						// 计算space_sigma
						sigmas_s_jbu[sp_count] = this->GetSigmaOfPts2D(Label2Pts2d[it->first]);

						// 填充pts2d_has_no_depth_jbu数组和sp_labels_idx_jbu数组
						for (auto pt2D : has_no_depth_map[it->first])
						{
							// 平面约束求深度值
							float depth = this->GetDepthCoPlaneCam(K_inv_arr,
								plane_map.at(it->first).data(), pt2D);

							// 要计算JBUSP的pt2d点
							if (depth <= 0.0f)
							{
								pts2d_no_depth_jbu.push_back(pt2D);
								sp_labels_idx_jbu.push_back(sp_count);
							}
							else
							{
								// 根据平面约束, 更新depth
								depth_map.Set(pt2D.y, pt2D.x, depth);  // 此处对depth_map进行了写入操作!!!
							}
						}

						sp_count += 1;
						stride += int(has_depth_map[it->first].size());
					}
					std::printf("GPU preparations for JBUSP built done, total %d pts for JBUSP\n",
						(int)pts2d_no_depth_jbu.size());

					assert(pts2d_has_depth_jbu.size()
						== std::accumulate(sp_has_depth_pt2ds_num.begin(), sp_has_depth_pt2ds_num.end(), 0));

					// GPU端运行JBUSP
					std::vector<float> depths_ret(pts2d_no_depth_jbu.size(), 0.0f);

					if (pts2d_no_depth_jbu.size() > 0)
					{
						std::printf("Start GPU JBUSP...\n");
						JBUSPGPU(src,
							depth_mat,
							pts2d_no_depth_jbu,  // 待处理的pt2d点数组
							sp_labels_idx_jbu,  // 每个待处理dept2d点对应的label idx
							pts2d_has_depth_jbu,
							sp_has_depth_pt2ds_num,
							sigmas_s_jbu,
							depths_ret);
						std::printf("GPU JBUSP done\n");

						// 完成depth_map填充
						for (int i = 0; i < (int)pts2d_no_depth_jbu.size(); ++i)
						{// 此处对depth_map进行了写入操作!!!
							depth_map.Set((int)pts2d_no_depth_jbu[i].y,
								(int)pts2d_no_depth_jbu[i].x, depths_ret[i]);
						}
					}
					std::printf("Depthmap completed\n");

#ifdef MRF_GPU
					// -----构建GPU的输入, 输出数组(Host端)：这一部分需要优化...
					// 构建按照sp_label_depths(按label排列的深度值数组): 按照Label2Pts2d键的排列
					// 构建sp_label_pts2d_size, 每个label的pt2d点个数: 按照Label2Pts2d键的排列
					// 构建sp_label数组: 按照Label2Pts2d键的排列
					// 构建NoDepthPts2d和NoDepthPt2DSPLabelsPre数组: 按照Label2Pts2d键的排列
					// --- 统计没有深度值的点数
					pt_count = 0;
					for (auto it = has_no_depth_map.begin(); it != has_no_depth_map.end(); ++it)
					{
						pt_count += (int)it->second.size();
					}
					const int NoDepthPt2dCount = pt_count;  // 记录没有深度值pt2d点总数
					std::vector<int> sp_labels(Label2Pts2d.size(), 0);
					std::vector<float> sp_label_depths(labels.rows*labels.cols, 0.0f);
					std::vector<int> pts2d_size(sp_labels.size());
					std::vector<cv::Point2f> NoDepthPts2d(NoDepthPt2dCount);  // 待处理的pt2d点
					std::vector<int> NoDepthPt2DSPLabelsRet(NoDepthPt2dCount, 0);  // 构建结果返回数组
					std::vector<int> NoDepthPts2dLabelIdx(NoDepthPt2dCount);  // 构建每个pt2d点的label_idx
					std::vector<int> NoDepthPts2DSPLabelsPre(NoDepthPt2dCount, 0);  // 每个pt2d点前一次的label
					std::vector<float> sp_label_plane_arrs(Label2Pts2d.size() * 4 + 9, 0.0f);

					// ---填充NoDepthPts2d(待MRF优化的pt2d点)
					// ---填充NoDepthPt2DSPLabelsPre数组
					stride = 0;
					for (auto it = has_no_depth_map.begin();
						it != has_no_depth_map.end(); ++it)
					{
						// 填充待MRF优化的pt2d点
						memcpy(NoDepthPts2d.data() + stride,
							it->second.data(),
							sizeof(cv::Point2f) * it->second.size());

						// 填充pt2d点的初始化label
						std::vector<int> tmp(it->second.size(), it->first);  // sp label
						memcpy(NoDepthPts2DSPLabelsPre.data() + stride,
							tmp.data(),
							sizeof(int) * it->second.size());

						stride += int(it->second.size());
					}  // 避免不必要的push_back

					// -----基于NeighborMap构建Neighbors数组和label idx对应的邻居数量
					const auto NeighborMap = this->GetNeighborMap(labels);
					std::vector<int> sp_label_neighs_idx;  // 按照Label2Pts2d键的排列
					std::vector<int> sp_label_neigh_num(Label2Pts2d.size(), 0);  // 按照Label2Pts2d键的排列

					stride = 0;
					sp_count = 0;
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						// Label2Pts2d键de数组
						sp_labels[sp_count] = it->first;

						// 填充pts2d_size
						pts2d_size[sp_count] = (int)it->second.size();

						// 填充平面方程数组
						memcpy(sp_label_plane_arrs.data() + sp_count * 4,
							plane_map.at(it->first).data(),
							sizeof(float) * 4);

						// 填充sp_label_neighbor_num数组
						sp_label_neigh_num[sp_count] = (int)NeighborMap.at(it->first).size();

						// 更新sp_count
						sp_count++;

						// 填充深度值数组: 按照Label2Pts2d键排列
						float* ptr_depths = sp_label_depths.data() + stride;
						int pt_i = 0;
						for (auto pt2d : it->second)
						{
							ptr_depths[pt_i++] = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
						}

						// 更新stride
						stride += (int)it->second.size();
					}

					// 将K_inv_arr添加到数组sp_label_plane_arrs的最后
					memcpy(sp_label_plane_arrs.data() + int(Label2Pts2d.size()) * 4,
						K_inv_arr,
						sizeof(float) * 9);

					// 填充sp_label_neighs_idx数组
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						for (int neigh_label : NeighborMap.at(it->first))
						{
							auto it = std::find(sp_labels.begin(), sp_labels.end(), neigh_label);
							int label_idx = std::distance(std::begin(sp_labels), it);
							sp_label_neighs_idx.push_back(label_idx);  // label_idx而非label
						}
					}

					// 填充NoDepthPts2dLabelIdx数组
					for (int i = 0; i < NoDepthPt2dCount; ++i)
					{
						const int& label = NoDepthPts2DSPLabelsPre[i];
						auto it = std::find(sp_labels.begin(), sp_labels.end(), label);
						if (it != sp_labels.end())
						{
							int label_idx = std::distance(std::begin(sp_labels), it);
							NoDepthPts2dLabelIdx[i] = label_idx;
						}
					}

					std::printf("Total %d no depth pt2ds\n", (int)NoDepthPts2d.size());
					std::printf("GPU IO for MRF built done\n");

					// ----- GPU端运行MRF优化...
					int radius = 20;
					float beta = 1.0f;
					const int WIDTH = depth_map.GetWidth();
					const int HEIGHT = depth_map.GetHeight();

					// --- MRF迭代
					//int num_bad_pt2d = INT32_MAX, iter_i = 0;
					//while (iter_i < 10 && (float)num_bad_pt2d / (float)NoDepthPts2d.size() > 0.05f)
					//for (int iter_i = 0; iter_i < NUM_MRF_ITER; ++iter_i)
					//{
					std::printf("Start GPU MRF...\n");

					// 获取初始化的depth_mat
					depth_mat = depth_map.Depth2Mat();

					//MRFGPU(labels,
					//	sp_labels, sp_label_depths,
					//	pts2d_size,
					//	NoDepthPts2d,
					//	sp_label_plane_arrs,
					//	radius, WIDTH, HEIGHT, beta,
					//	NoDepthPt2DSPLabelsRet);
					//MRFGPU2(labels,						 // 需要更新
					//	sp_labels,
					//	sp_label_depths,                 // 需要更新
					//	pts2d_size,                      // 需要更新
					//	NoDepthPts2d,
					//	NoDepthPts2dLabelIdx,			 // 需要更新
					//	sp_label_plane_arrs,             // 需不需要重新拟合, 待定?
					//	sp_label_neighs_idx,			 // 需要更新
					//	sp_label_neigh_num,				 // 需要更新
					//	radius, WIDTH, HEIGHT, beta,
					//	NoDepthPt2DSPLabelsRet);
					//MRFGPU3(depth_mat,
					//	labels,
					//	sp_labels,
					//	NoDepthPts2d,
					//	NoDepthPts2dLabelIdx,
					//	sp_label_plane_arrs,
					//	sp_label_neighs_idx,
					//	sp_label_neigh_num,
					//	radius, WIDTH, HEIGHT, beta,
					//	NoDepthPt2DSPLabelsRet);

					MRFGPU4(depth_mat,
						labels,						     // 需要更新
						sp_labels,
						sp_label_depths,                 // 需要更新
						pts2d_size,                      // 需要更新
						NoDepthPts2d,
						NoDepthPts2dLabelIdx,			 // 需要更新
						sp_label_plane_arrs,             // 需不需要重新拟合, 待定?
						sp_label_neighs_idx,			 // 需要更新
						sp_label_neigh_num,				 // 需要更新
						radius, WIDTH, HEIGHT, beta,
						NoDepthPt2DSPLabelsRet);
					std::printf("GPU MRF done\n");

#ifdef LOGGING
					// 打印label更新信息
					for (int pt_i = 0; pt_i < (int)NoDepthPts2d.size(); ++pt_i)
					{
						if (pt_i % 10000 == 0)
						{
							if (NoDepthPt2DSPLabelsRet[pt_i] != NoDepthPts2DSPLabelsPre[pt_i])
							{
								std::printf("[%d, %d] SP label %d updated with %d\n",
									(int)NoDepthPts2d[pt_i].x,
									(int)NoDepthPts2d[pt_i].y,
									NoDepthPts2DSPLabelsPre[pt_i],
									NoDepthPt2DSPLabelsRet[pt_i]);
							}
						}
					}
#endif // LOGGING

					// ----- MRF后处理
					// 更新labels数组(opencv Mat)
					for (int pt_i = 0; pt_i < (int)NoDepthPts2d.size(); ++pt_i)
					{
						const int& label = NoDepthPt2DSPLabelsRet[pt_i];
						if (Label2Pts2d.find(label) != Label2Pts2d.end())
						{
							const cv::Point2f& pt2d = NoDepthPts2d[pt_i];
							labels.at<int>((int)pt2d.y, (int)pt2d.x) = label;  // 更新pt2d点的label
						}
						else
						{
							std::printf("Wrong label: %d\n", label);
						}
					}
					std::printf("labels updated\n");

					// 更新Label2Pts2d: 依据labels重新生成Label2Pts2d, 效率更高
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						it->second.clear();  // Label2Pts2d键不变, 值需要更新
					}
					for (int y = 0; y < labels.rows; ++y)
					{
						for (int x = 0; x < labels.cols; ++x)
						{
							const int& label = labels.at<int>(y, x);

							// label -> 图像2D坐标点集
							Label2Pts2d[label].push_back(cv::Point2f(float(x), float(y)));
						}
					}
					std::printf("Labels2Pts2d updated\n");

					// 更新sp_label_depths和pts2d_size
					sp_count = 0, stride = 0;
					sp_label_depths.resize(labels.cols*labels.rows, 0.0f);
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						// 更新pts2d_size
						pts2d_size[sp_count++] = (int)it->second.size();

						// 更新深度值数组: 按照Label2Pts2d键排列
						float* ptr_depths = sp_label_depths.data() + stride;
						int pt_i = 0;
						for (auto pt2d : it->second)
						{
							ptr_depths[pt_i++] = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
						}

						// 更新stride
						stride += (int)it->second.size();
					}
					std::printf("sp_label_depths and pts2d_size updated\n");

					// 用NoDepthPt2DSPLabelsRet更新NoDepthPt2DSPLabelsPre
					memcpy(NoDepthPts2DSPLabelsPre.data(), NoDepthPt2DSPLabelsRet.data(),
						sizeof(int) * (size_t)NoDepthPt2dCount);
					std::printf("NoDepthPt2DSPLabelsPre updated\n");

					// 依据NoDepthPts2DSPLabelsPre更新NoDepthPts2dLabelIdx(每个点对应的label idx)数组
					NoDepthPts2dLabelIdx.clear();
					NoDepthPts2dLabelIdx.resize(NoDepthPt2dCount, 0);
					for (int i = 0; i < NoDepthPt2dCount; ++i)
					{
						const int& label = NoDepthPts2DSPLabelsPre[i];
						auto it = std::find(sp_labels.begin(), sp_labels.end(), label);
						int label_idx = std::distance(std::begin(sp_labels), it);
						NoDepthPts2dLabelIdx[i] = label_idx;
					}
					std::printf("NoDepthPts2dLabelIdx updated\n");

					// ---根据Label2Pts2d, 更新sp_label_neighs_idx
					// 根据labels, 重新计算neighbor_map
					auto neighbor_map = this->GetNeighborMap(labels);
					std::printf("NeighborMap updated\n");

					// 清空数据重新计算sp_label_neighs_idx
					sp_label_neighs_idx.clear();
					sp_label_neigh_num.clear();
					sp_label_neigh_num.resize(Label2Pts2d.size(), 0);

					sp_count = 0;  // 重新计数
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						// 更新sp_label_neighbor_num数组
						sp_label_neigh_num[sp_count++] = (int)neighbor_map.at(it->first).size();

						// 更新sp_label_neighs_idx数组
						for (int neigh_label : neighbor_map.at(it->first))
						{
							auto it = std::find(sp_labels.begin(), sp_labels.end(), neigh_label);
							int label_idx = std::distance(std::begin(sp_labels), it);
							sp_label_neighs_idx.push_back(label_idx);  // label_idx而非label
						}
					}
					std::printf("sp_label_neighs_idx and sp_label_neigh_num updated\n");

					// ---更新depth_mat
					// ---- GPU端运行JBUSP
					// 释放旧内存
					pts2d_no_depth_jbu.clear();
					pts2d_no_depth_jbu.shrink_to_fit();
					sp_labels_idx_jbu.clear();
					sp_labels_idx_jbu.shrink_to_fit();

					// 收集需要计算JBUSP的pt2d点
					for (int pt_i = 0; pt_i < (int)NoDepthPts2d.size(); ++pt_i)
					{
						const cv::Point2f& pt2d = NoDepthPts2d[pt_i];
						const int& label = labels.at<int>((int)pt2d.y, (int)pt2d.x);

						// 首先基于平面约束, 计算深度值
						float depth = 0.0f;
						const float* plane_arr = plane_map.at(label).data();
						depth = this->GetDepthCoPlaneCam(K_inv_arr, plane_arr, pt2d);

						if (depth <= 0.0f)
						{
							// 将需要计算JBUSP的pt2d点放进容器
							pts2d_no_depth_jbu.push_back(pt2d);

							// 计算该pt2d点的label idx
							auto it = std::find(sp_labels_jbu.begin(),
								sp_labels_jbu.end(),
								label);
							int label_idx = std::distance(std::begin(sp_labels_jbu), it);
							sp_labels_idx_jbu.push_back(label_idx);
						}
						else
						{
							// 依据平面约束更新深度值
							depth_map.Set((int)pt2d.y, (int)pt2d.x, depth);
						}
					}

					if (0 < pts2d_no_depth_jbu.size())
					{
						// 调用GPU
						depths_ret.clear();
						depths_ret.resize(pts2d_no_depth_jbu.size(), 0.0f);
						std::printf("Start GPU JBUSP...\n");
						JBUSPGPU(src, depth_mat,
							pts2d_no_depth_jbu,  // 待处理的pt2d点数组
							sp_labels_idx_jbu,  // 每个待处理dept2d点对应的label idx
							pts2d_has_depth_jbu,
							sp_has_depth_pt2ds_num,
							sigmas_s_jbu,
							depths_ret);
						std::printf("GPU JBUSP done\n");

						// 完成depth_map填充
						for (int i = 0; i < (int)pts2d_no_depth_jbu.size(); ++i)
						{// 此处对depth_map进行了写入操作!!!
							const cv::Point2f& pt2d = pts2d_no_depth_jbu[i];
							depth_map.Set((int)pt2d.y, (int)pt2d.x, depths_ret[i]);
						}

						std::printf("Depthmap completed\n");
						std::printf("Iteration %d done\n", iter_i + 1);
					}

					// 更新斑块滤除参数
					//denominator -= 10.0f;
					//ratio -= 0.005;
					denominator *= 0.95f;
					ratio *= 0.95f;
#endif // MRF_GPU
				}

				// ----- 输出Filled二进制深度图: 用于重建的最终深度图数据
				string filled_out_name = file_name + ".geometric.bin";
				viz_out_path = depth_dir + filled_out_name;
				depth_map.WriteBinary(viz_out_path);
				std::printf("%s written\n", filled_out_name.c_str());

				// 输出Filled深度图用于可视化[0, 255]
				string filled_name = file_name + "_filled.jpg";
				viz_out_path = depth_dir + filled_name;
				cv::imwrite(viz_out_path, depth_map.ToBitmapGray(2, 98));;;
				std::printf("%s written\n\n", filled_name.c_str());

				//// ----- 为Filled深度图绘制mask, 输出深度图+super-pixel mask 用于可视化[0, 255]
				//cv::Mat depth_filled = cv::imread(viz_out_path, cv::IMREAD_COLOR);
				//this->DrawMaskOfSuperpixels(labels, depth_filled);
				//// 绘制superpixel编号
				//for (auto it = has_depth_map.begin(); it != has_depth_map.end(); ++it)
				//{
				//	cv::Point2f center_2d(0.0f, 0.0f);
				//	for (auto pt2d : has_depth_map[it->first])
				//	{
				//		center_2d.x += pt2d.x;
				//		center_2d.y += pt2d.y;
				//	}
				//	center_2d.x /= float(has_depth_map[it->first].size());
				//	center_2d.y /= float(has_depth_map[it->first].size());
				//	cv::putText(depth_filled, std::to_string(it->first),
				//		cv::Point((int)center_2d.x, (int)center_2d.y),
				//		cv::FONT_HERSHEY_SIMPLEX,
				//		1.0,
				//		cv::Scalar(255, 0, 0));
				//}
				//string filled_mask_name = file_name + "_filled_mask.jpg";
				//viz_out_path = depth_dir + filled_mask_name;
				//cv::imwrite(viz_out_path, depth_filled);
				//std::printf("%s written\n\n", filled_mask_name.c_str());
			}
		}

		int Workspace::SplitDepthMat(const cv::Mat& depth_mat,
			const int blk_size,  // block size
			const float* K_inv_arr,  // 相机内参矩阵的逆
			std::vector<int>& blks_pt_cnt,  // 记录所有block的pt2d数量
			std::vector<cv::Point2f>& blks_pts2d,  // 记录所有block的pt2d点坐标
			std::vector<int>& blks_pt_cnt_has,  // 记录待处理block的有深度值点个数
			std::vector<int>& blks_pt_cnt_non,  // 记录待处理block的无深度值点个数
			std::vector<std::vector<float>>& plane_equa_arr,  // 记录作为label的blk_id对应的平面方程
			std::vector<int>& label_blk_ids,  // 记录有足够多深度值点的blk_id: 可当作label
			std::vector<int>& proc_blk_ids,  // 记录待(MRF)处理的blk_id
			std::vector<float>& proc_blks_depths_has,  // 记录待处理block的有深度值(组成的数组)
			std::vector<int>& proc_blks_pts2d_has_num,  // 记录待处理block的有深度值点个数
			std::vector<int>& proc_blks_pts2d_non_num,  // 记录待处理block无深度点个数
			std::vector<cv::Point2f>& proc_blks_pt2d_non,  // 记录待处理block的无深度值点坐标
			std::vector<int>& all_blks_labels,  // 记录每个block对应的label(blk_id): 初始label数组
			int& num_y, int& num_x)  // 记录blk总的行数, 列数
		{
			// 按块划分depth mat
			const int& HEIGHT = depth_mat.rows;
			const int& WIDTH = depth_mat.cols;
			const int remain_y = HEIGHT % blk_size;
			const int remain_x = WIDTH % blk_size;

			// block idx用作label
			int blk_id = 0;

			if (remain_x && remain_y)
			{

			}
			else if (remain_x)  // y整除, x没整除
			{

			}
			else if (remain_y)  // x整除, y没整除
			{
				// 为blks_pts2d分配内存
				blks_pts2d.resize(HEIGHT*WIDTH);

				num_y = (HEIGHT / blk_size) + 1;
				num_x = WIDTH / blk_size;

				// 为blks_labels分配内存大小
				const int NumBlk = num_x * num_y;
				all_blks_labels.resize(NumBlk);
				blks_pt_cnt.resize(NumBlk);
				blks_pt_cnt_has.resize(NumBlk);
				blks_pt_cnt_non.resize(NumBlk);

				int blk_stride = 0;
				for (int i = 0; i < num_y; ++i)
				{
					for (int j = 0; j < num_x; ++j)
					{
						// 处理每一块
						if (i == num_y - 1)  // 边界上的block
						{

						}
						else
						{
							// 统计有效深度值和无深度值的点
							const int y_start = i * blk_size;
							const int y_end = (i + 1) * blk_size;
							const int x_start = j * blk_size;
							const int x_end = (j + 1) * blk_size;

							std::vector<cv::Point2f> blk_pts2d_has;  // 每个block有深度值的点
							int blk_pt_cnt = 0, blk_pt_cnt_has = 0, blk_pt_cnt_non = 0;

							cv::Point2f* ptr_blks_pts2d = blks_pts2d.data() + blk_stride;
							for (int y = y_start; y < y_end; ++y)
							{
								for (int x = x_start; x < x_end; ++x)
								{
									// 填充ptr_blks_pts2d数组
									ptr_blks_pts2d[blk_pt_cnt++] = cv::Point2f(float(x), float(y));

									const float& depth = depth_mat.at<float>(y, x);
									if (depth > 0.0f)
									{
										// 更新block有深度点计数
										blk_pt_cnt_has++;

										// 记录所有block的有深度值的点
										blk_pts2d_has.push_back(cv::Point2f((float)x, (float)y));
									}
									else
									{
										// 更新block无深度点计数
										blk_pt_cnt_non++;
									}
								}
							}

							// 更新blk_stride
							blk_stride += blk_pt_cnt;

							// 记录每个block中pt2d点个数
							blks_pt_cnt[blk_id] = blk_pt_cnt;

							// 记录每个block对应的有深度值点数, 无深度值点数
							blks_pt_cnt_has[blk_id] = blk_pt_cnt_has;
							blks_pt_cnt_non[blk_id] = blk_pt_cnt_non;

							// 如果block存在无深度点, 即为待处理block
							// 填充待处理的block的有深度值数组
							if (blk_pt_cnt_non > 0)
							{
								// 只要block中出现1个无深度值点即需要MRF优化label
								proc_blk_ids.push_back(blk_id);

								// 填充process_blks_pts2d_num数组
								proc_blks_pts2d_has_num.push_back(blk_pt_cnt_has);

								// 填充process_blks_pts2d_non_num数组
								proc_blks_pts2d_non_num.push_back(blk_pt_cnt_non);

								for (int y = y_start; y < y_end; ++y)
								{
									for (int x = x_start; x < x_end; ++x)
									{
										// 填充blks_pts2d数组

										const float& depth = depth_mat.at<float>(y, x);
										if (depth > 0.0f)
										{
											// 填充blks_depths_has数组
											proc_blks_depths_has.push_back(depth);
										}
										else
										{
											// 填充blks_pt2d_non数组
											proc_blks_pt2d_non.push_back(cv::Point2f((float)x, (float)y));
										}
									}
								}
							}

							// 有足够比例的深度值的点, 记录label
							if (float(blk_pt_cnt_has) / float(blk_size*blk_size) >= 0.9f)
							{
								// 统计label(blk_id)
								label_blk_ids.push_back(blk_id);

								// 填充blks_labels数组: 此时blk_id不变
								all_blks_labels[blk_id] = blk_id;

								// ----- 计算3D相机坐标系下的坐标
								std::vector<cv::Point3f> Pts3D;
								Pts3D.resize(blk_pts2d_has.size());
								for (int i = 0; i < (int)blk_pts2d_has.size(); i++)
								{
									cv::Point3f Pt3D = this->m_model.BackProjTo3DCam(K_inv_arr,
										depth_mat.at<float>((int)blk_pts2d_has[i].y, (int)blk_pts2d_has[i].x),
										blk_pts2d_has[i]);
									Pts3D[i] = Pt3D;
								}

								// 3D空间平面拟合: using OLS, SVD or PCA
								RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
								ransac.RunRansac(Pts3D);
								std::vector<float> plane_equa(4);
								memcpy(plane_equa.data(), ransac.m_plane, sizeof(float) * 4);

								// 填充plane_equa_arr数组: 跟label_blk_ids的顺序一致
								plane_equa_arr.push_back(plane_equa);

							}  // 该block预处理结束

							// 更新blk_id, 预处理下一个block
							blk_id++;
						}
					}
				}

				// 继续填充all_blks_labels数组, 为待处理的blk_id随机赋值label
				// 从all_blks_labels中随机选取
				for (int blk_id : proc_blk_ids)
				{
					all_blks_labels[blk_id] = label_blk_ids[rand() % int(label_blk_ids.size())];
				}

			}
			else  // x, y完全整除
			{

			}

			return 0;
		}

		// 分块MRF
		void Workspace::TestDepth7()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";

			// 总的视角个数
			const int NumView = (int)this->m_model.m_images.size();
			for (int img_id = 0; img_id < NumView; ++img_id)  // 注意: img_id != IMAGE_ID
			{
				string depth_f_name = GetFileName(img_id, true);

				string file_name(depth_f_name);
				StringReplace(file_name, string(".geometric.bin"), string(""));

				// modify bin depth file name
				StringReplace(depth_f_name, string(".geometric.bin"), string(".geometric_win5.bin"));

				DepthMap depth_map(this->depth_ranges_.at(img_id).first,
					this->depth_ranges_.at(img_id).second);
				depth_map.ReadBinary(depth_dir + depth_f_name);
				std::printf("%s read\n", depth_f_name.c_str());

				// ----------- Test output depth map for visualization
				string viz_out_path = depth_dir + depth_f_name + ".jpg";

				// ----------- speckle filtering using CV_32F data
				int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					/ 80.0f);  // 80
				const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
				float maxDiff = 0.032f * depth_range;  // 0.038

				cv::Mat depth_mat = depth_map.Depth2Mat();

				// speckle filtering for depth_mat
				this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

				// fill depth_map
				depth_map.fillDepthWithMat(depth_mat);  // 此处对depth_map进行了写入操作!!!

				// 拷贝, 保存filtered的depth_mat
				cv::Mat depth_mat_filter = depth_mat.clone();

				// write to disk for visualization
				string filter_name = file_name + "_filtered_orig.jpg";
				string filter_path = depth_dir + filter_name;
				cv::imwrite(filter_path, depth_map.ToBitmapGray(2, 98));
				std::printf("%s written\n", filter_name.c_str());

				// ----------- super-pixel segmentation
				cv::Mat src, mask, labels;

				// 原图读取BGR彩色图
				src = cv::imread(src_dir + file_name, cv::IMREAD_COLOR);
				if (src.empty())
				{
					std::printf("[Err]: empty src image\n");
					return;
				}

				// ----- super-pixel segmentation using SEEDS or SLIC
				// SEEDS super-pixel segmentation
				const int num_superpixels = 700;  // 更多的初始分割保证边界
				Ptr<cv::ximgproc::SuperpixelSEEDS> superpixel = cv::ximgproc::createSuperpixelSEEDS(src.cols,
					src.rows,
					src.channels(),
					num_superpixels,  // num_superpixels
					15,  // num_levels: 5, 15
					2,
					5,
					true);
				superpixel->iterate(src);  // 迭代次数，默认为4
				superpixel->getLabels(labels);  // 获取labels
				superpixel->getLabelContourMask(mask);  // 获取超像素的边界

				// construct 2 Hashmaps for each super-pixel
				std::unordered_map<int, std::vector<cv::Point2f>> Label2Pts2d,
					has_depth_map, has_no_depth_map;

				// traverse each pxiel to put into hashmaps
				for (int y = 0; y < labels.rows; ++y)
				{
					for (int x = 0; x < labels.cols; ++x)
					{
						const int& label = labels.at<int>(y, x);

						// label -> 图像2D坐标点集
						Label2Pts2d[label].push_back(cv::Point2f(float(x), float(y)));

						if (depth_mat.at<float>(y, x) > 0.0f)
						{
							has_depth_map[label].push_back(cv::Point2f(float(x), float(y)));
						}
						else
						{
							has_no_depth_map[label].push_back(cv::Point2f(float(x), float(y)));
						}
					}
				}

#ifdef DRAW_MASK
				// ----- 为原图绘制superpixel merge之前的mask
				cv::Mat src_1 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_1);
				string mask_src1_name = file_name + "_before_merge_mask.jpg";
				viz_out_path = depth_dir + mask_src1_name;
				cv::imwrite(viz_out_path, src_1);
				std::printf("%s written\n", mask_src1_name.c_str());
#endif // DRAW_MASK

				// ----- superpixel合并(合并有效depth少于阈值的)
				std::printf("Before merging, %d superpixels\n", has_depth_map.size());
				this->MergeSuperpixels(src,
					2000,  // 2000
					labels,
					Label2Pts2d, has_depth_map, has_no_depth_map);
				std::printf("After merging, %d superpixels\n", has_depth_map.size());

#ifdef DRAW_MASK
				// ----- 为原图绘制superpixel merge之后的mask
				cv::Mat src_2 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_2);
				string mask_src2_name = file_name + "_after_merge_mask.jpg";
				viz_out_path = depth_dir + mask_src2_name;
				cv::imwrite(viz_out_path, src_2);
				std::printf("%s written\n", mask_src2_name.c_str());
#endif // DRAW_MASK

				// ----------- traverse each super-pixel for 3D plane fitting
				// project 2D points with depth back to 3D world coordinate
				const float* K_arr = this->m_model.m_images[img_id].GetK();
				const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();

				// ----- 遍历Merge后的superpixels, 平面拟合
				std::unordered_map<int, std::vector<float>> eigen_vals_map;  // superpixel的特征值
				std::unordered_map<int, std::vector<float>> eigen_vects_map;  // superpixel特征向量
				std::unordered_map<int, std::vector<float>> plane_normal_map;  // superpxiel的法向量
				std::unordered_map<int, std::vector<float>> plane_map;  // superpixel的切平面方程
				std::unordered_map<int, cv::Point3f> center_map;  // superpixel的中心点坐标
				this->FitPlaneForSPsCam(depth_map,
					K_inv_arr,
					Label2Pts2d, has_depth_map,
					center_map, eigen_vals_map, eigen_vects_map,
					plane_normal_map, plane_map);
				std::printf("Superpixel plane fitting done\n");

				//// ----- 修正tagent plane
				//this->CorrectPCPlaneCam(K_arr, K_inv_arr,
				//	src.cols, src.rows,
				//	depth_range * 0.05f,
				//	3.0f,
				//	Label2Pts2d,
				//	plane_map, center_map, plane_normal_map,
				//	eigen_vals_map, eigen_vects_map);
				//std::printf("Superpixel plane correction done\n");

				// ----- superpixel连接: 连接可连接的相邻superpixel,
				this->ConnectSuperpixelsCam(0.0002f, 0.2f, depth_map,
					K_inv_arr,
					plane_map,
					eigen_vals_map, plane_normal_map, center_map,
					labels, Label2Pts2d, has_depth_map, has_no_depth_map);
				printf("Connect done, after connect, %d superpixels\n",
					has_depth_map.size());

#ifdef DRAW_MASK
				// ----- 为原图绘制Connect之后的mask
				cv::Mat src_3 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_3);
				string mask_src3_name = file_name + "_after_connect_mask.jpg";
				viz_out_path = depth_dir + mask_src3_name;
				cv::imwrite(viz_out_path, src_3);
				printf("%s written\n", mask_src3_name.c_str());

				// ----- 在填充depth_map之前, 为depth_map绘制mask(用于算法分析)
				cv::Mat depth_mask = cv::imread(filter_path, cv::IMREAD_COLOR);
				this->DrawMaskOfSuperpixels(labels, depth_mask);
				string filter_mask = file_name + "_depth_filter_mask.jpg";
				viz_out_path = depth_dir + filter_mask;
				cv::imwrite(viz_out_path, depth_mask);
				std::printf("%s written\n", filter_mask.c_str());
#endif // DRAW_MASK

				// ----- 接下来步骤的共用计数参数
				int stride = 0, sp_count = 0, pt_count = 0;

				// ----- GPU端初始化depth_map, JBUSPGPU并行
				// ---- 构建GPU输入输出
				// 1. ---更新具有正常深度值的pt2d, 将depth < 0的放入一个数组
				// 2. ---统计depth<0的pt2d点的信息:
				// pts2d_has_no_depth_jbu, sp_labels_idx_jbu
				std::vector<cv::Point2f> pts2d_has_no_depth_jbu;  // 所有待处理的pt2d点
				std::vector<int> sp_labels_idx_jbu;  // 每个点的sp label index

				// ---JBUSPGPU的准备工作包括4个数组的初始化: 
				// (1). sp_labels_jbu, (2). pts2d_has_depth_jbu
				// (3). sp_has_depth_pt2ds_num (4). sigmas_s_jbu
				// pts2d_has_depth_jbu,
				// 用于GPU端JBUSP的sp label数组
				std::vector<int> sp_labels_jbu(has_no_depth_map.size(), 0);

				// ---统计有深度值pt2d点总共的点数
				pt_count = 0;
				for (auto it = has_no_depth_map.begin();
					it != has_no_depth_map.end(); ++it)
				{
					pt_count += (int)has_depth_map.at(it->first).size();
				}

				// 所有label对应的pt2d点数组: 依据has_no_depth_map的label顺序
				std::vector<cv::Point2f> pts2d_has_depth_jbu(pt_count);

				// 每个label对应的pt2d点个数
				std::vector<int> sp_has_depth_pt2ds_num(has_no_depth_map.size());

				// 构架sigma_s数组: 每个label都对应一个sigma_s值
				std::vector<float> sigmas_s_jbu(has_no_depth_map.size());

				stride = 0, sp_count = 0;  // sp_count即has_no_depth_map的label_idx
				for (auto it = has_no_depth_map.begin();
					it != has_no_depth_map.end(); ++it)
				{
					// 构建sp_labels_jbu
					sp_labels_jbu[sp_count] = it->first;

					// 构建pts2d_has_depth_jbu
					memcpy(pts2d_has_depth_jbu.data() + stride,
						has_depth_map[it->first].data(),
						sizeof(cv::Point2f) * has_depth_map[it->first].size());

					// 构建sp_has_depth_pt2ds_num
					sp_has_depth_pt2ds_num[sp_count] = int(has_depth_map[it->first].size());

					// 计算space_sigma
					sigmas_s_jbu[sp_count] = this->GetSigmaOfPts2D(Label2Pts2d[it->first]);

					// 填充pts2d_has_no_depth_jbu数组和sp_labels_idx_jbu数组
					for (auto pt2D : has_no_depth_map[it->first])
					{
						// 平面约束求深度值
						float depth = this->GetDepthCoPlaneCam(K_inv_arr,
							plane_map.at(it->first).data(), pt2D);

						// 要计算JBUSP的pt2d点
						if (depth <= 0.0f)
						{
							pts2d_has_no_depth_jbu.push_back(pt2D);
							sp_labels_idx_jbu.push_back(sp_count);
						}
						else
						{
							// 根据平面约束, 更新depth
							depth_map.Set(pt2D.y, pt2D.x, depth);  // 此处对depth_map进行了写入操作!!!
					}
				}

					sp_count += 1;
					stride += int(has_depth_map[it->first].size());
			}
				std::printf("GPU preparations for JBUSP built done, total %d pts for JBUSP\n",
					(int)pts2d_has_no_depth_jbu.size());

				assert(pts2d_has_depth_jbu.size()
					== std::accumulate(sp_has_depth_pt2ds_num.begin(), sp_has_depth_pt2ds_num.end(), 0));

				// GPU端运行JBUSP
				std::vector<float> depths_ret(pts2d_has_no_depth_jbu.size(), 0.0f);
				std::printf("Start GPU JBUSP...\n");
				JBUSPGPU(src,
					depth_mat,
					pts2d_has_no_depth_jbu,  // 待处理的pt2d点数组
					sp_labels_idx_jbu,  // 每个待处理dept2d点对应的label idx
					pts2d_has_depth_jbu,
					sp_has_depth_pt2ds_num,
					sigmas_s_jbu,
					depths_ret);
				std::printf("GPU JBUSP done\n");

				// 完成depth_map填充
				for (int i = 0; i < (int)pts2d_has_no_depth_jbu.size(); ++i)
				{// 此处对depth_map进行了写入操作!!!
					depth_map.Set((int)pts2d_has_no_depth_jbu[i].y,
						(int)pts2d_has_no_depth_jbu[i].x, depths_ret[i]);
				}

				// 更新初始化之后的depth_mat
				cv::Mat init_depth_mat = depth_map.Depth2Mat();

				std::printf("Depthmap initialized\n");

				//----- 分块MRF优化
				int blk_size = 5, radius = 50;
				const float beta = 1.0f;
				std::vector<int> blks_pt_cnt;  // 记录所有block的pt2d数量
				std::vector<cv::Point2f> blks_pts2d;  // 记录所有block的pt2d点坐标
				std::vector<int> blks_pt_cnt_has;  // 记录待处理block的有深度值点个数
				std::vector<int> blks_pt_cnt_non;  // 记录待处理block的无深度值点个数
				std::vector<std::vector<float>> plane_equa_arr;  // 记录作为label的blk_id对应的平面方程
				std::vector<int> label_blk_ids;  // 记录有足够多深度值点的blk_id: 可当作label
				std::vector<int> proc_blk_ids;  // 记录待(MRF)处理的blk_id
				std::vector<float> proc_blks_depths_has;  // 记录待处理block的有深度值(组成的数组)
				std::vector<int> proc_blks_pts2d_has_num;  // 记录待处理block的有深度值点个数
				std::vector<int> proc_blks_pts2d_non_num;  // 记录待处理block无深度点个数
				std::vector<cv::Point2f> proc_blks_pt2d_non;  // 记录待处理block的无深度值点坐标
				std::vector<int> all_blks_labels;  // 记录每个block对应的label(blk_id): 初始label数组
				int num_y, num_x;  // y, x方向block块数
				std::vector<int> label_ids_ret;  // 返回结果数组

				maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					/ 10.0f);
				maxDiff = 0.04f * depth_range;

				const int NumIter = 3;
				for (int iter_i = 0; iter_i < NumIter; ++iter_i)
				{
					std::printf("Block size: %d\n", blk_size);

					// 分割depth_mat
					this->SplitDepthMat(depth_mat_filter,  // 使用filtered之后的depth_mat
						blk_size,
						K_inv_arr,
						blks_pt_cnt,
						blks_pts2d,
						blks_pt_cnt_has,
						blks_pt_cnt_non,
						plane_equa_arr,
						label_blk_ids,
						proc_blk_ids,
						proc_blks_depths_has,
						proc_blks_pts2d_has_num,
						proc_blks_pts2d_non_num,
						proc_blks_pt2d_non,
						all_blks_labels,
						num_y, num_x);
					std::printf("Split depth mat done\n");

					// 初始化label_ids_ret数组
					label_ids_ret.resize(proc_blk_ids.size(), 0);

					// 调用GPU block MRF
					BlockMRF(init_depth_mat,  // 为一元能量项准备
						blk_size, K_inv_arr,
						blks_pt_cnt, blks_pts2d,
						blks_pt_cnt_has, blks_pt_cnt_non,
						plane_equa_arr,
						label_blk_ids,
						proc_blk_ids,
						proc_blks_depths_has,
						proc_blks_pts2d_has_num,
						proc_blks_pts2d_non_num,
						proc_blks_pt2d_non,
						all_blks_labels,
						num_x, num_y,
						radius, beta, depth_range,  // radius, beta, depth_range
						label_ids_ret);
					std::printf("GPU BlockMRF done\n");

					// 更新all_blks_labels
					for (int proc_i = 0; proc_i < (int)proc_blk_ids.size(); ++proc_i)
					{
						const int& blk_id_old = proc_blk_ids[proc_i];
						const int& blk_id_new = label_blk_ids[label_ids_ret[proc_i]];
						all_blks_labels[blk_id_old] = blk_id_new;
					}

					// 更新depth_map, 处理每一个MRF优化的block
					for (int proc_i = 0; proc_i < (int)proc_blk_ids.size(); ++proc_i)
					{
						// 取旧的block idx
						const int& blk_id_old = proc_blk_ids[proc_i];

						// 取新label对应的plane equation
						const float* pl_arr = plane_equa_arr[label_ids_ret[proc_i]].data();

						// 判断该block是全空还是部分空
						if (proc_blks_pts2d_non_num[proc_i] == blks_pt_cnt[blk_id_old])  // 全空block
						{
							// ---该block每个pt2d点都通过新label的平面约束插值
							// --遍历该block的每一个点
							// 计算offset
							int offset = 0;
							for (int idx = 0; idx < blk_id_old; ++idx)
							{
								offset += blks_pt_cnt[idx];
							}

							// 计算block pt2d点集初始指针
							const cv::Point2f* ptr_blks_pts2d = blks_pts2d.data() + offset;

							// 遍历block每一个pt2d点
							for (int k = 0; k < blks_pt_cnt[blk_id_old]; ++k)
							{
								const cv::Point2f& pt2d = ptr_blks_pts2d[k];

								// 平面约束插值
								const float depth = this->GetDepthCoPlaneCam(K_inv_arr, pl_arr, pt2d);
								depth_map.Set((int)pt2d.y, (int)pt2d.x, depth);
							}
						}
						else if (proc_blks_pts2d_non_num[proc_i] < blks_pt_cnt[blk_id_old])  // 非全空block
						{
							// 非空部分不做处理, 找出深度值为空的pt2d点,
							// 深度为空部分通过新label平面约束插值
							// 计算无深度点数组的offset
							int offset = 0;
							for (int j = 0; j < proc_i; ++j)
							{
								offset += proc_blks_pts2d_non_num[j];
							}

							// 计算无深度点数组开始指针
							const cv::Point2f* ptr_proc_blks_pt2d_non = proc_blks_pt2d_non.data() + offset;

							for (int k = 0; k < proc_blks_pts2d_non_num[proc_i]; ++k)
							{
								const cv::Point2f& pt2d = ptr_proc_blks_pt2d_non[k];

								// 平面约束插值
								const float depth = this->GetDepthCoPlaneCam(K_inv_arr, pl_arr, pt2d);
								depth_map.Set((int)pt2d.y, (int)pt2d.x, depth);
							}
						}
						else
						{
							std::printf("[Err]: block pt2d number wrong.\n");
							continue;
						}
					}

					// 更新blk_size, radius
					blk_size = blk_size * 2;

					// 清除内存
					blks_pt_cnt.clear(); blks_pt_cnt.shrink_to_fit();
					blks_pts2d.clear(); blks_pts2d.shrink_to_fit();
					blks_pt_cnt_has.clear(); blks_pt_cnt_has.shrink_to_fit();
					blks_pt_cnt_non.clear(); blks_pt_cnt_non.shrink_to_fit();
					plane_equa_arr.clear(); plane_equa_arr.shrink_to_fit();
					label_blk_ids.clear(); label_blk_ids.shrink_to_fit();
					proc_blk_ids.clear(); proc_blk_ids.shrink_to_fit();
					proc_blks_depths_has.clear(); proc_blks_depths_has.shrink_to_fit();
					proc_blks_pts2d_has_num.clear(); proc_blks_pts2d_has_num.shrink_to_fit();
					proc_blks_pts2d_non_num.clear(); proc_blks_pts2d_non_num.shrink_to_fit();
					proc_blks_pt2d_non.clear(); proc_blks_pt2d_non.shrink_to_fit();
					all_blks_labels.clear(); all_blks_labels.shrink_to_fit();

					// ----- for debug... 保存中文件
					// 保存depth_map
					char buff[100];
					sprintf(buff, "_iter%d_filled.jpg", iter_i + 1);
					string iter_filled_name = file_name + string(buff);
					viz_out_path = depth_dir + iter_filled_name;
					cv::imwrite(viz_out_path, depth_map.ToBitmapGray(2, 98));
					std::printf("%s written\n", iter_filled_name.c_str());

					if (iter_i == NumIter - 1)
					{
						std::printf("Iter %d done\n", iter_i + 1);
						break;
					}

					// ----- 为下一轮迭代做准备
					// --- 更新depth_mat
					depth_mat = depth_map.Depth2Mat();

					// --- 更新depth_mat_filter
					depth_mat_filter = depth_mat.clone();

					// --- 滤除斑块, speckle filtering for depth_mat
					this->FilterSpeckles<float>(depth_mat_filter, 0.0f, maxSpeckleSize, maxDiff);

					// 保存filtered
					depth_map.fillDepthWithMat(depth_mat_filter);
					sprintf(buff, "_iter%d_filterSpecke_32F.jpg", iter_i + 1);
					string iter_filter_out_name = file_name + string(buff);
					viz_out_path = depth_dir + iter_filter_out_name;
					cv::imwrite(viz_out_path, depth_map.ToBitmapGray(2, 98));
					std::printf("%s written\n", iter_filter_out_name.c_str());

					std::printf("Iter %d done\n", iter_i + 1);
		}

				// ----- 输出Filled二进制深度图: 用于重建的最终深度图数据
				string filled_out_name = file_name + ".geometric.bin";
				viz_out_path = depth_dir + filled_out_name;
				depth_map.WriteBinary(viz_out_path);  // 写出depth_map为bin文件
				std::printf("%s written\n", filled_out_name.c_str());

				// 输出Filled深度图用于可视化[0, 255]
				string filled_name = file_name + "_filled.jpg";
				viz_out_path = depth_dir + filled_name;
				cv::imwrite(viz_out_path, depth_map.ToBitmapGray(2, 98));
				std::printf("%s written\n\n", filled_name.c_str());
	}
}

		//void Workspace::TestDepth2()
		//{
		//	const string depth_dir = this->options_.workspace_path \
			//		+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			//	const string consistency_dir = this->options_.workspace_path \
			//		+ "/dense/stereo/consistency_graphs/dslr_images_undistorted/";
			//	const string src_dir = this->options_.workspace_path \
			//		+ "/dense/images/dslr_images_undistorted/";

			//	typedef ahc::PlaneFitter<OrganizedImage3D> PlaneFitter;

			//	PlaneFitter pf;
			//	pf.minSupport = 2000;
			//	pf.windowWidth = 7;
			//	pf.windowHeight = 7;
			//	pf.doRefine = true;

			//	// 遍历每个视角
			//	int NumView = (int)this->m_model.m_images.size();
			//	for (int img_id = 0; img_id < NumView; ++img_id)
			//	{
			//		string depth_f_name = GetFileName(img_id, true);
			//		string consistency_f_name(depth_f_name);

			//		string file_name(depth_f_name);
			//		StringReplace(file_name, string(".geometric.bin"), string(""));

			//		// modify bin depth file name
			//		StringReplace(depth_f_name, string(".geometric.bin"), string(".geometric_old.bin"));

			//		DepthMap depth_map(this->depth_ranges_.at(img_id).first,
			//			this->depth_ranges_.at(img_id).second);
			//		depth_map.ReadBinary(depth_dir + depth_f_name);
			//		printf("%s read\n", depth_f_name.c_str());

			//		// 换换成opencv Mat
			//		cv::Mat& depth_mat = depth_map.Depth2Mat();

			//		const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
			//			/ 100.0f);

			//		// using depth range
			//		const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
			//		const float maxDiff = 0.1f * depth_range;

			//		// ----- speckle filtering for depth_mat
			//		this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

			//		// 提取ROI
			//		cv::Mat ROI(depth_mat, cv::Rect2i(20, 1, 1960, 1330));
			//		printf("ROI: %d×%d\n", ROI.cols, ROI.rows);

			//		// 获取相机参数
			//		//const float* P_arr = this->m_model.m_images[img_id].GetP();
			//		//const float* K_arr = this->m_model.m_images[img_id].GetK();
			//		//const float* R_arr = this->m_model.m_images[img_id].GetR();
			//		const float* T_arr = this->m_model.m_images[img_id].GetT();
			//		const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();
			//		const float* R_inv_arr = this->m_model.m_images[img_id].GetInvR();

			//		// 提取ROI点云
			//		cv::Mat_<cv::Vec3f> cloud(ROI.rows, ROI.cols);
			//		for (int r = 0; r < ROI.rows; r++)
			//		{
			//			// 深度图一行的指针
			//			const float* depth_ptr = ROI.ptr<float>(r);

			//			// 点云一行的指针
			//			cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
			//			for (int c = 0; c < ROI.cols; c++)
			//			{
			//				// 提取3D坐标
			//				const float& depth = depth_ptr[c];

			//				// 判断深度值是否有效


			//				cv::Point3f pt3d = this->m_model.BackProjTo3D(K_inv_arr, R_inv_arr, T_arr, 
			//					depth, cv::Point2f(c, r));
			//				//printf("%.3f, %.3f, %.3f\n", pt3d.x, pt3d.y, pt3d.z);

			//				// m -> mm
			//				pt_ptr[c][0] = pt3d.x * 1000.0f;
			//				pt_ptr[c][1] = pt3d.y * 1000.0f;
			//				pt_ptr[c][2] = pt3d.z * 1000.0f;
			//			}
			//		}

			//		// 平面检测
			//		cv::Mat seg(ROI.rows, ROI.cols, CV_8UC3);
			//		OrganizedImage3D Ixyz(cloud);
			//		pf.run(&Ixyz, 0, &seg);

			//		// 可视化原始深度图和平面检测结果
			//		//depth_mat.convertTo(depth_mat, cv::)
			//		cv::Mat depth_rs, seg_rs;
			//		cv::resize(seg, seg_rs, cv::Size(int(seg.cols*0.5f), int(seg.rows*0.5f)));

			//		char buff[100];
			//		sprintf(buff, "%s_%dplanes", file_name.c_str(), (int)pf.extractedPlanes.size());
			//		cv::imshow(buff, seg_rs);
			//		cv::waitKey();
			//	}

			//}

		PlaneDetection plane_detection;

		//-----------------------------------------------------------------
		// MRF energy functions
		MRF::CostVal dCost(int pix, int label)
		{
			return plane_detection.dCost(pix, label);
		}

		MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
		{
			return plane_detection.fnCost(pix1, pix2, i, j);
		}

		void Workspace::runMRFOptimization()
		{
			DataCost *data = new DataCost(dCost);
			SmoothnessCost *smooth = new SmoothnessCost(fnCost);
			EnergyFunction *energy = new EnergyFunction(data, smooth);
			int width = kDepthWidth, height = kDepthHeight;
			MRF* mrf = new Expansion(width * height, plane_detection.plane_num_ + 1, energy);

			// Set neighbors for the graph
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					int pix = y * width + x;
					if (x < width - 1)  // horizontal neighbor
					{
						mrf->setNeighbors(pix, pix + 1, 1);
					}
					if (y < height - 1)  // vertical
					{
						mrf->setNeighbors(pix, pix + width, 1);
					}
					if (y < height - 1 && x < width - 1)  // diagonal
					{
						mrf->setNeighbors(pix, pix + width + 1, 1);
					}
				}
			}

			mrf->initialize();
			mrf->clearAnswer();

			float t;
			mrf->optimize(5, t);  // run for 5 iterations, store time t it took 

			MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
			MRF::EnergyVal E_data = mrf->dataEnergy();
			cout << "Optimized Energy: smooth = " << E_smooth << ", data = " << E_data << endl;
			cout << "Time consumed in MRF: " << t << endl;

			// Get MRF result
			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < width; col++)
				{
					int pix = row * width + col;
					plane_detection.opt_seg_img_.at<cv::Vec3b>(row, col) = plane_detection.plane_colors_[mrf->getLabel(pix)];
					plane_detection.opt_membership_img_.at<int>(row, col) = mrf->getLabel(pix);
				}
			}

			delete mrf; mrf = nullptr;
			delete energy; energy = nullptr;
			delete smooth; smooth = nullptr;
			delete data; data = nullptr;
		}

		//typedef ahc::PlaneFitter<OrganizedImage3D> PlaneFitter;
		//PlaneFitter pf;
		//pf.minSupport = 2000;
		//pf.windowWidth = 7;
		//pf.windowHeight = 7;
		//pf.doRefine = true;
		void Workspace::TestDepth3()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";
			const string out_dir = string("C:/output/");

			// 遍历每个视角
			int NumView = (int)this->m_model.m_images.size();
			for (int img_id = 0; img_id < NumView; ++img_id)
			{
				string depth_f_name = GetFileName(img_id, true);
				string consistency_f_name(depth_f_name);

				string file_name(depth_f_name);
				StringReplace(file_name, string(".geometric.bin"), string(""));

				// modify bin depth file name
				StringReplace(depth_f_name, string(".geometric.bin"), string(".geometric_win5.bin"));

				DepthMap depth_map(this->depth_ranges_.at(img_id).first,
					this->depth_ranges_.at(img_id).second);
				depth_map.ReadBinary(depth_dir + depth_f_name);
				printf("%s read\n", depth_f_name.c_str());

				// 换换成opencv Mat
				cv::Mat& depth_mat = depth_map.Depth2Mat();

				//// ----- speckle filtering for depth_mat
				//const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					//	/ 100.0f);
					//const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
					//const float maxDiff = 0.1f * depth_range;
					//this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

				float depth_max = depth_map.GetDepthMax();
				float depth_min = depth_map.GetDepthMin();

				// 提取ROI
				cv::Mat ROI(depth_mat, cv::Rect2i(20, 1, 1960, 1330));
				printf("ROI: %d*%d\n", ROI.cols, ROI.rows);

				// 获取相机参数
				//const float* P_arr = this->m_model.m_images[img_id].GetP();
				//const float* K_arr = this->m_model.m_images[img_id].GetK();
				//const float* R_arr = this->m_model.m_images[img_id].GetR();
				const float* T_arr = this->m_model.m_images[img_id].GetT();
				const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();
				const float* R_inv_arr = this->m_model.m_images[img_id].GetInvR();
				//for (int i = 0; i < 9; ++i)
				//{
				//	printf("%.3f\n", K_arr[i]);
				//}

				// 提取ROI点云
				//cv::Mat depth_out(ROI.rows, ROI.cols, CV_16UC1);

				cv::Mat_<cv::Vec3f> cloud(ROI.rows, ROI.cols);
				for (int r = 0; r < ROI.rows; r++)
				{
					// 深度图一行的指针
					const float* depth_ptr = ROI.ptr<float>(r);

					// 点云一行的指针
					cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
					for (int c = 0; c < ROI.cols; c++)
					{
						// 提取3D坐标
						const float& depth = depth_ptr[c];
						cv::Point3f pt3d = this->m_model.BackProjTo3D(K_inv_arr, R_inv_arr, T_arr,
							depth, cv::Point2f(c, r));
						//printf("%.3f, %.3f, %.3f\n", pt3d.x, pt3d.y, pt3d.z);

						// m -> mm
						pt_ptr[c][0] = pt3d.x;
						pt_ptr[c][1] = pt3d.y;
						pt_ptr[c][2] = pt3d.z;

						//depth_out.at<unsigned short>(r, c) = unsigned short(depth * 1000.0f);
					}
				}
				printf("Depth min: %.3f, depth max: %.3f\n", depth_min, depth_max);

				// run plane extraction
				string color_filename = src_dir + file_name;
				plane_detection.readColorImage(color_filename);
				plane_detection.readCloud(cloud);
				plane_detection.runPlaneDetection();

				//// 输出u16深度图
				//int pos = color_filename.find_last_of("/\\");
				//string frame_name = color_filename.substr(pos + 1);
				//string out_depth_path = out_dir + frame_name + string("_16u.png");
				//cv::imwrite(out_depth_path, depth_out);

				//// 输出彩色图
				//cv::Mat src = cv::imread(color_filename, cv::IMREAD_COLOR);
				//cv::Mat color_out(src, cv::Rect2i(20, 1, 1960, 1330));
				//string color_out_path = out_dir + frame_name + string("_color.png");
				//cv::imwrite(color_out_path, color_out);

				bool run_mrf = false;
				if (run_mrf)
				{
					plane_detection.prepareForMRF();
					runMRFOptimization();
				}

				//// 输出深度图jpg
				//string depth_map_path = out_dir + depth_f_name + ".jpg";
				//cv::imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				//printf("%s written\n", depth_map_path.c_str());

				// 输出平面检测结果
				int pos = color_filename.find_last_of("/\\");
				string frame_name = color_filename.substr(pos + 1);
				frame_name += string("_nofilter_") + std::to_string(plane_detection.plane_num_) + string("planes");
				plane_detection.writeOutputFiles(out_dir, frame_name, run_mrf);
				printf("%s written @ %s\n\n", frame_name.c_str(), out_dir);
			}
		}

		void Workspace::OutputU16Depthmap()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";
			const string out_dir = string("C:/output/");

			// 遍历每个视角
			int NumView = (int)this->m_model.m_images.size();
			for (int img_id = 0; img_id < NumView; ++img_id)
			{
				string depth_f_name = GetFileName(img_id, true);
				string consistency_f_name(depth_f_name);

				string file_name(depth_f_name);
				StringReplace(file_name, string(".geometric.bin"), string(""));

				// modify bin depth file name
				StringReplace(depth_f_name, string(".geometric.bin"), string(".geometric_win5.bin"));

				DepthMap depth_map(this->depth_ranges_.at(img_id).first,
					this->depth_ranges_.at(img_id).second);
				depth_map.ReadBinary(depth_dir + depth_f_name);
				printf("%s read\n", depth_f_name.c_str());

				// 换换成opencv Mat
				cv::Mat& depth_mat = depth_map.Depth2Mat();

				// 获取相机参数
				const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();

				// 输出的Mat
				cv::Mat depth_out(depth_mat.rows, depth_mat.cols, CV_16UC1);

				// 遍历原始深度图
				for (int r = 0; r < depth_mat.rows; r++)
				{
					// 深度图一行的指针
					const float* depth_ptr = depth_mat.ptr<float>(r);

					// 点云一行的指针
					for (int c = 0; c < depth_mat.cols; c++)
					{
						// 提取3D坐标
						depth_out.at<unsigned short>(r, c) = unsigned short(depth_ptr[c] * 1000.0f);
					}
				}

				// 输出*1000的uint16深度图
				const string color_filename = src_dir + file_name;
				int pos = color_filename.find_last_of("/\\");
				string frame_name = color_filename.substr(pos + 1);
				string out_depth_path = out_dir + frame_name + string("_16u.png");
				cv::imwrite(out_depth_path, depth_out);
				std::printf("%s written\n", out_depth_path.c_str());
			}
		}

		void Workspace::TestDepth4()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";
			const string out_dir = string("C:/output/");

			// 遍历每个视角
			int NumView = (int)this->m_model.m_images.size();
			for (int img_id = 0; img_id < NumView; ++img_id)
			{
				string depth_f_name = GetFileName(img_id, true);

				string file_name(depth_f_name);
				StringReplace(file_name, string(".geometric.bin"), string(""));

				// modify bin depth file name
				StringReplace(depth_f_name, string(".geometric.bin"), string(".geometric_win5.bin"));

				DepthMap depth_map(this->depth_ranges_.at(img_id).first,
					this->depth_ranges_.at(img_id).second);
				depth_map.ReadBinary(depth_dir + depth_f_name);
				printf("%s read\n", depth_f_name.c_str());

				// 换换成opencv Mat
				cv::Mat& depth_mat = depth_map.Depth2Mat();

				//// ----- speckle filtering for depth_mat
				//const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					//	/ 100.0f);
					//const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
					//const float maxDiff = 0.1f * depth_range;
					//this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

					// 提取ROI
				cv::Mat ROI(depth_mat, cv::Rect2i(20, 1, 1960, 1330));
				printf("ROI: %d*%d\n", ROI.cols, ROI.rows);

				// 获取相机参数
				const float* K_arr = this->m_model.m_images[img_id].GetK();

				// run plane extraction
				string color_filename = src_dir + file_name;
				plane_detection.readColorImage(color_filename);
				plane_detection.readDepthMat(ROI);
				plane_detection.runPlaneDetection();

				bool run_mrf = false;
				if (run_mrf)
				{
					plane_detection.prepareForMRF();
					runMRFOptimization();
				}

				// 输出平面检测结果
				int pos = color_filename.find_last_of("/\\");
				string frame_name = color_filename.substr(pos + 1);
				frame_name += string("_nofilter_")
					+ std::to_string(plane_detection.plane_num_) + string("planes");
				plane_detection.writeOutputFiles(out_dir, frame_name, run_mrf);
				printf("%s written @ %s\n\n", frame_name.c_str(), out_dir);
			}
		}

		// superpixel对应的点云拟合切平面
		int Workspace::FitPlaneForSuperpixel(const DepthMap& depth_map,
			const float* K_inv_arr,
			const float* R_inv_arr,
			const float* T_arr,
			const std::vector<cv::Point2f>& Pts2D,
			std::vector<float>& plane_normal,
			std::vector<float>& eigen_vals,
			std::vector<float>& center_arr)
		{
			if (Pts2D.size() < 3)
			{
				printf("[Err]: points less than 3\n");
				return -1;
			}

			// ----- 计算3D世界坐标系下的坐标
			std::vector<cv::Point3f> Pts3D;
			Pts3D.reserve(Pts2D.size());
			Pts3D.resize(Pts2D.size());

			for (int i = 0; i < (int)Pts2D.size(); i++)
			{
				cv::Point3f Pt3D = this->m_model.BackProjTo3D(K_inv_arr,
					R_inv_arr,
					T_arr,
					depth_map.GetDepth((int)Pts2D[i].y, (int)Pts2D[i].x),
					Pts2D[i]);

				Pts3D[i] = Pt3D;
			}

			// ----- 计算每个superpixel的中心
			float center[3] = { 0.0f };
			for (auto pt3D : Pts3D)
			{
				center[0] += pt3D.x;
				center[1] += pt3D.y;
				center[2] += pt3D.z;
			}
			center[0] /= float(Pts3D.size());
			center[1] /= float(Pts3D.size());
			center[2] /= float(Pts3D.size());

			// 返回点云中心坐标
			center_arr.reserve(3);
			center_arr.resize(3);
			memcpy(center_arr.data(), center, sizeof(float) * 3);

			// ----- PCA平面拟合RANSAC
			if (Pts3D.size() <= 5)
			{
				// using OLS, SVD or PCA
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
				ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

				// 返回平面法向量
				plane_normal.reserve(3);
				plane_normal.resize(3);
				memcpy(plane_normal.data(), ransac.m_plane, sizeof(float) * 3);

				// 返回特征值
				eigen_vals.reserve(3);
				eigen_vals.resize(3);
				memcpy(eigen_vals.data(), ransac.m_eigen_vals, sizeof(float) * 3);
			}
			else
			{
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);  // using OLS, SVD or PCA
				ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

				// 返回平面法向量
				plane_normal.reserve(3);
				plane_normal.resize(3);
				memcpy(plane_normal.data(), ransac.m_plane, sizeof(float) * 3);

				// 返回特征值
				eigen_vals.reserve(3);
				eigen_vals.resize(3);
				memcpy(eigen_vals.data(), ransac.m_eigen_vals, sizeof(float) * 3);
			}

			return 0;
		}

		int Workspace::FitPlaneForSuperpixelCam(const DepthMap& depth_map,
			const float* K_inv_arr,
			const std::vector<cv::Point2f>& Pts2D,
			std::vector<float>& plane_arr,
			std::vector<float>& plane_normal,
			std::vector<float>& eigen_vals,
			std::vector<float>& center_arr)
		{
			if (Pts2D.size() < 3)
			{
				printf("[Err]: points less than 3\n");
				return -1;
			}

			// ----- 计算3D空间(世界坐标系或者相机坐标系)的坐标
			std::vector<cv::Point3f> Pts3D;
			Pts3D.resize(Pts2D.size());

			for (int i = 0; i < (int)Pts2D.size(); i++)
			{
				cv::Point3f Pt3D = this->m_model.BackProjTo3DCam(K_inv_arr,
					depth_map.GetDepth((int)Pts2D[i].y, (int)Pts2D[i].x),
					Pts2D[i]);
				Pts3D[i] = Pt3D;
			}

			// ----- 计算每个superpixel的中心
			float center[3] = { 0.0f };
			for (auto pt3D : Pts3D)
			{
				center[0] += pt3D.x;
				center[1] += pt3D.y;
				center[2] += pt3D.z;
			}
			center[0] /= float(Pts3D.size());
			center[1] /= float(Pts3D.size());
			center[2] /= float(Pts3D.size());

			// 返回点云中心坐标
			center_arr.resize(3, 0.0f);
			memcpy(center_arr.data(), center, sizeof(float) * 3);

			// ----- PCA平面拟合RANSAC
			//if (Pts3D.size() <= 5)
			{
				// using OLS, SVD or PCA
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
				ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

				// 返回平面方程
				plane_arr.resize(4, 0.0f);
				memcpy(plane_arr.data(), ransac.m_plane, sizeof(float) * 4);

				// 返回平面法向量
				plane_normal.resize(3, 0.0f);
				memcpy(plane_normal.data(), ransac.m_plane, sizeof(float) * 3);

				// 返回特征值
				eigen_vals.resize(3, 0.0f);
				memcpy(eigen_vals.data(), ransac.m_eigen_vals, sizeof(float) * 3);
			}
			//else
			//{
			//	RansacRunner ransac(0.08f, (int)Pts3D.size(), 3);  // using OLS, SVD or PCA
			//	ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

			//	// 返回平面方程
			//	plane_arr.resize(4, 0.0f);
			//	memcpy(plane_arr.data(), ransac.m_plane, sizeof(float) * 4);

			//	// 返回平面法向量
			//	plane_normal.resize(3, 0.0f);
			//	memcpy(plane_normal.data(), ransac.m_plane, sizeof(float) * 3);

			//	// 返回特征值
			//	eigen_vals.resize(3, 0.0f);
			//	memcpy(eigen_vals.data(), ransac.m_eigen_vals, sizeof(float) * 3);
			//}

			return 0;
		}

		int Workspace::FitPlaneForSuperpixel(
			const std::vector<cv::Point3f>& Pts3D,
			float* plane_normal,
			float* eigen_vals,
			float* eigen_vects,
			float* center_arr)
		{
			if (Pts3D.size() < 3)
			{
				printf("[Err]: points less than 3\n");
				return -1;
			}

			// ----- 计算每个superpixel的中心
			float center[3] = { 0.0f };
			for (auto pt3D : Pts3D)
			{
				center[0] += pt3D.x;
				center[1] += pt3D.y;
				center[2] += pt3D.z;
			}
			center[0] /= float(Pts3D.size());
			center[1] /= float(Pts3D.size());
			center[2] /= float(Pts3D.size());

			// ----- 3D plane fittin using RANSAC
			if (Pts3D.size() <= 5)
			{
				// using OLS, SVD or PCA
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
				ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

				// 返回点云中心坐标
				memcpy(center_arr, center, sizeof(float) * 3);

				// 返回平面法向量
				memcpy(plane_normal, ransac.m_plane, sizeof(float) * 3);

				// 返回特征值
				memcpy(eigen_vals, ransac.m_eigen_vals, sizeof(float) * 3);

				// 返回特征向量
				memcpy(eigen_vects, ransac.m_eigen_vect, sizeof(float) * 9);
			}
			else
			{
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
				ransac.RunRansac(Pts3D);

				// 返回点云中心坐标
				memcpy(center_arr, center, sizeof(float) * 3);

				// 返回平面法向量
				memcpy(plane_normal, ransac.m_plane, sizeof(float) * 3);

				// 返回特征值
				memcpy(eigen_vals, ransac.m_eigen_vals, sizeof(float) * 3);

				// 返回特征向量
				memcpy(eigen_vects, ransac.m_eigen_vect, sizeof(float) * 9);
			}

			return 0;
		}

		int Workspace::FitPlaneForSuperpixel(const std::vector<cv::Point3f>& Pts3D,
			std::vector<float>& plane_arr,
			std::vector<float>& eigen_vals,
			std::vector<float>& eigen_vects,
			cv::Point3f& center_pt)
		{
			// ----- 计算superpixel的中心坐标
			float center[3] = { 0.0f };
			for (auto pt3D : Pts3D)
			{
				center[0] += pt3D.x;
				center[1] += pt3D.y;
				center[2] += pt3D.z;
			}
			center[0] /= float(Pts3D.size());
			center[1] /= float(Pts3D.size());
			center[2] /= float(Pts3D.size());

			// 返回点云中心坐标
			center_pt.x = center[0];
			center_pt.y = center[1];
			center_pt.z = center[2];

			if (Pts3D.size() < 3)
			{
				printf("[Err]: points less than 3\n");
				return -1;
			}

			// ----- 3D plane fittin using RANSAC
			else
			{
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
				ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

				// 返回平面方程
				plane_arr.resize(4, 0.0f);
				memcpy(plane_arr.data(), ransac.m_plane, sizeof(float) * 4);

				// 返回特征值
				eigen_vals.resize(3, 0.0f);
				memcpy(eigen_vals.data(), ransac.m_eigen_vals, sizeof(float) * 3);

				// 返回特征向量
				eigen_vects.resize(9, 0.0f);
				memcpy(eigen_vects.data(), ransac.m_eigen_vect, sizeof(float) * 9);
			}

			return 0;
		}

		int Workspace::FitPlaneForSuperpixels(const DepthMap& depth_map,
			const float* K_inv_arr,
			const float* R_inv_arr,
			const float* T_arr,
			const std::unordered_map<int, std::vector<cv::Point2f>>& labels_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
			std::unordered_map<int, cv::Point3f>& center_map,
			std::unordered_map<int, std::vector<float>>& eigen_vals_map,
			std::unordered_map<int, std::vector<float>>& eigen_vects_map,
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, std::vector<float>>& plane_map)
		{
			// 遍历每一个superpixel
			for (auto it = labels_map.begin();
				it != labels_map.end(); it++)
			{
				if ((int)it->second.size() < 3)
				{
					printf("[Warning]: Not enough valid depth within super-pixel %d\n", it->first);
					continue;
				}

				// 取存在深度值的2D点
				const std::vector<cv::Point2f>& Pts2D = has_depth_map[it->first];

				// ----- 计算3D世界坐标系下的坐标
				std::vector<cv::Point3f> Pts3D;
				Pts3D.reserve(Pts2D.size());
				Pts3D.resize(Pts2D.size());

				for (int i = 0; i < (int)Pts2D.size(); i++)
				{
					cv::Point3f Pt3D = this->m_model.BackProjTo3D(K_inv_arr,
						R_inv_arr,
						T_arr,
						depth_map.GetDepth((int)Pts2D[i].y, (int)Pts2D[i].x),
						Pts2D[i]);

					Pts3D[i] = Pt3D;
				}

				// ----- 计算每个superpixel的中心
				float center_x = 0.0f, center_y = 0.0f, center_z = 0.0f;
				for (auto pt3D : Pts3D)
				{
					center_x += pt3D.x;
					center_y += pt3D.y;
					center_z += pt3D.z;
				}
				center_x /= float(Pts3D.size());
				center_y /= float(Pts3D.size());
				center_z /= float(Pts3D.size());

				// 填充superpixel对应3D点云的中心点坐标
				center_map[it->first] = cv::Point3f(center_x, center_y, center_z);

				// ----- 3D plane fittin using RANSAC
				if (Pts3D.size() <= 5)
				{
					// using OLS, SVD or PCA
					RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
					ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

					// 填充特征值eigen_vals_map
					eigen_vals_map[it->first].reserve(3);
					eigen_vals_map[it->first].resize(3);
					memcpy(eigen_vals_map[it->first].data(), ransac.m_eigen_vals, sizeof(float) * 3);

					// 填充特征向量: 9个元素
					eigen_vects_map[it->first].reserve(9);
					eigen_vects_map[it->first].resize(9);
					memcpy(eigen_vects_map[it->first].data(), ransac.m_eigen_vect, sizeof(float) * 9);

					// 填充平面方程: 4个元素
					plane_map[it->first].reserve(4);
					plane_map[it->first].resize(4);
					memcpy(plane_map[it->first].data(), ransac.m_plane, sizeof(float) * 4);

					// 填充法向量: 切平面方程的前三项是法向量
					plane_normal_map[it->first].reserve(3);
					plane_normal_map[it->first].resize(3);
					memcpy(plane_normal_map[it->first].data(), plane_map[it->first].data(), sizeof(float) * 3);
				}
				else
				{
					RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
					ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

					// 填充特征值eigen_vals_map
					eigen_vals_map[it->first].reserve(3);
					eigen_vals_map[it->first].resize(3);
					memcpy(eigen_vals_map[it->first].data(), ransac.m_eigen_vals, sizeof(float) * 3);

					// 填充特征向量: 9个元素
					eigen_vects_map[it->first].reserve(9);
					eigen_vects_map[it->first].resize(9);
					memcpy(eigen_vects_map[it->first].data(), ransac.m_eigen_vect, sizeof(float) * 9);

					// 填充平面方程: 4个元素
					plane_map[it->first].reserve(4);
					plane_map[it->first].resize(4);
					memcpy(plane_map[it->first].data(), ransac.m_plane, sizeof(float) * 4);

					// 填充法向量: 切平面方程的前三项是法向量
					plane_normal_map[it->first].reserve(3);
					plane_normal_map[it->first].resize(3);
					memcpy(plane_normal_map[it->first].data(), plane_map[it->first].data(), sizeof(float) * 3);
				}

				//printf("Superpixel %d tagent plane fitted\n", it->first);
			}

			return 0;
		}

		int Workspace::FitPlaneForSPsCam(const DepthMap& depth_map,
			const float* K_inv_arr,
			const std::unordered_map<int, std::vector<cv::Point2f>>& labels_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
			std::unordered_map<int, cv::Point3f>& center_map,
			std::unordered_map<int, std::vector<float>>& eigen_vals_map,
			std::unordered_map<int, std::vector<float>>& eigen_vects_map,
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, std::vector<float>>& plane_map)
		{
			// 遍历每一个superpixel
			for (auto it = labels_map.begin();
				it != labels_map.end(); it++)
			{
				if ((int)it->second.size() < 3)
				{
					std::printf("[Warning]: Not enough valid depth within super-pixel %d\n", it->first);
					continue;
				}

				// 取存在深度值的2D点
				const std::vector<cv::Point2f>& Pts2D = has_depth_map[it->first];

				// ----- 计算3D相机坐标系下的坐标
				std::vector<cv::Point3f> Pts3D;
				Pts3D.reserve(Pts2D.size());
				Pts3D.resize(Pts2D.size());

				for (int i = 0; i < (int)Pts2D.size(); i++)
				{
					cv::Point3f Pt3D = this->m_model.BackProjTo3DCam(K_inv_arr,
						depth_map.GetDepth((int)Pts2D[i].y, (int)Pts2D[i].x),
						Pts2D[i]);

					Pts3D[i] = Pt3D;
				}

				// ----- 计算每个superpixel的中心
				float center_x = 0.0f, center_y = 0.0f, center_z = 0.0f;
				for (auto pt3D : Pts3D)
				{
					center_x += pt3D.x;
					center_y += pt3D.y;
					center_z += pt3D.z;
				}
				center_x /= float(Pts3D.size());
				center_y /= float(Pts3D.size());
				center_z /= float(Pts3D.size());

				// 填充superpixel对应3D点云的中心点坐标
				center_map[it->first] = cv::Point3f(center_x, center_y, center_z);

				// ----- 3D plane fittin using RANSAC
				//if (Pts3D.size() <= 5)
				{
					// using OLS, SVD or PCA
					RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
					ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

					// 填充特征值eigen_vals_map
					eigen_vals_map[it->first].resize(3, 0.0f);
					memcpy(eigen_vals_map[it->first].data(), ransac.m_eigen_vals, sizeof(float) * 3);

					// 填充特征向量: 9个元素
					eigen_vects_map[it->first].resize(9, 0.0f);
					memcpy(eigen_vects_map[it->first].data(), ransac.m_eigen_vect, sizeof(float) * 9);

					// 填充平面方程: 4个元素
					plane_map[it->first].resize(4, 0.0f);
					memcpy(plane_map[it->first].data(), ransac.m_plane, sizeof(float) * 4);

					// 填充法向量: 切平面方程的前三项是法向量
					plane_normal_map[it->first].reserve(3);
					plane_normal_map[it->first].resize(3);
					memcpy(plane_normal_map[it->first].data(), plane_map[it->first].data(), sizeof(float) * 3);
				}
				//else
				//{
				//	RansacRunner ransac(0.08f, (int)Pts3D.size(), 3);
				//	ransac.RunRansac(Pts3D);  // 估算的ETH3D: 焦距~0.0204m(20.4mm)

				//	// 填充特征值eigen_vals_map
				//	eigen_vals_map[it->first].resize(3, 0.0f);
				//	memcpy(eigen_vals_map[it->first].data(), ransac.m_eigen_vals, sizeof(float) * 3);

				//	// 填充特征向量: 9个元素
				//	eigen_vects_map[it->first].resize(9, 0.0f);
				//	memcpy(eigen_vects_map[it->first].data(), ransac.m_eigen_vect, sizeof(float) * 9);

				//	// 填充平面方程: 4个元素
				//	plane_map[it->first].resize(4, 0.0f);
				//	memcpy(plane_map[it->first].data(), ransac.m_plane, sizeof(float) * 4);

				//	// 填充法向量: 切平面方程的前三项是法向量
				//	plane_normal_map[it->first].resize(3, 0.0f);
				//	memcpy(plane_normal_map[it->first].data(), plane_map[it->first].data(), sizeof(float) * 3);
				//}

				//printf("Superpixel %d tagent plane fitted\n", it->first);
			}

			return 0;
		}

		// 依据邻接表和plane_normal_map和center_map计算(光滑)可连接的superpixel
		int Workspace::ConnectSuperpixels(const float THRESH_1,
			const float THRESH_2,
			const DepthMap& depth_map,
			const float* K_inv_arr,
			const float* R_inv_arr,
			const float* T_arr,
			std::unordered_map<int, std::vector<float>>& eigen_vals_map,
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, cv::Point3f>& center_map,
			cv::Mat& labels,
			std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map)
		{
			// 遍历superpxiel
			for (auto it = label_map.begin(); it != label_map.end(); ++it)
			{
				// 获取superpixel拟合平面的维度
				int dim_1 = this->GetPlaneDimForSuperpixel(eigen_vals_map[it->first], THRESH_1);

				// TODO: 对于dim < 2的superpixel扩展区域重新计算切平面...

				// 排除"维度"为3的
				if (3 == dim_1)
				{
					continue;
				}
				else
				{
					const cv::Point3f& center_1 = center_map[it->first];
					const float* normal_1 = plane_normal_map[it->first].data();

					// 更新superpixel邻接表
					auto NeighborMap = GetNeighborMap(labels);

					// 连接所有"可连接"的邻接superpixel
					for (int neighbor : NeighborMap[it->first])
					{
						int dim_2 = this->GetPlaneDimForSuperpixel(eigen_vals_map[neighbor], THRESH_1);
						if (3 == dim_2)
						{
							continue;
						}
						else  // ----- 符合可连接的第一个条件
						{
							const cv::Point3f& center_2 = center_map[neighbor];
							const float* normal_2 = plane_normal_map[neighbor].data();

							float abs_normal_dot = fabs(
								normal_1[0] * normal_2[0] +
								normal_1[1] * normal_2[1] +
								normal_1[2] * normal_2[2]);
							float abs_dot_1 = fabsf(
								normal_1[0] * (center_1.x - center_2.x) +
								normal_1[1] * (center_1.y - center_2.y) +
								normal_1[2] * (center_1.z - center_2.z));
							float abs_dot_2 = fabsf(
								normal_2[0] * (center_1.x - center_2.x) +
								normal_2[1] * (center_1.y - center_2.y) +
								normal_2[2] * (center_1.z - center_2.z));

							// 取较大值
							float numerator = abs_dot_1 >= abs_dot_2 ? abs_dot_1 : abs_dot_2;

							float connectivity = numerator / abs_normal_dot;
							if (connectivity > THRESH_2)  // 如果"连接度"超过阈值
							{
								continue;
							}

							// ----- 符合可连接的第二个条件, 连接这两个superpixel
							else  // 连接这两个superpixel: neighbor -> it->first 
							{
								// 合并更新labels
								for (auto pt : label_map[neighbor])
								{
									labels.at<int>((int)pt.y, (int)pt.x) = it->first;
								}
								// 合并label_map
								for (auto pt : label_map[neighbor])
								{
									label_map[it->first].push_back(pt);
								}
								// 合并更新has_depth_map
								for (auto pt : has_depth_map[neighbor])
								{
									has_depth_map[it->first].push_back(pt);
								}
								// 合并has_no_depth_map
								for (auto pt : has_no_depth_map[neighbor])
								{
									has_no_depth_map[it->first].push_back(pt);
								}

								// ----- 更新这个superpixel的拟合平面
								std::vector<float> center_arr(3, 0.0f);

								// --- 更新plane_normal_map和eigen_vals_map
								// 提取有深度值的2D点
								const std::vector<cv::Point2f>& Pts2D = has_depth_map[it->first];

								// 重新拟合切平面
								this->FitPlaneForSuperpixel(depth_map,
									K_inv_arr, R_inv_arr, T_arr,
									Pts2D,
									plane_normal_map[it->first],
									eigen_vals_map[it->first],
									center_arr);

								// 更新center_map
								center_map[it->first].x = center_arr[0];
								center_map[it->first].y = center_arr[1];
								center_map[it->first].z = center_arr[2];

								// ---删除neighbor
								// 删除plane_normal_map中的neighbor
								if (plane_normal_map.find(neighbor) != plane_normal_map.end())
								{
									plane_normal_map.erase(neighbor);
								}

								// 删除eigen_vals_map中的neighbor
								if (eigen_vals_map.find(neighbor) != eigen_vals_map.end())
								{
									eigen_vals_map.erase(neighbor);
								}

								// 删除center_map中的neighbor
								if (center_map.find(neighbor) != center_map.end())
								{
									center_map.erase(neighbor);
								}

								// 删除label_map中的neighbor
								if (label_map.find(neighbor) != label_map.end())
								{
									label_map.erase(neighbor);
								}

								// 删除has_depth_map中的neighbor
								if (has_depth_map.find(neighbor) != has_depth_map.end())
								{
									has_depth_map.erase(neighbor);
								}

								// 删除has_no_depth_map中的neighbor
								if (has_no_depth_map.find(neighbor) != has_no_depth_map.end())
								{
									has_no_depth_map.erase(neighbor);
								}

								//printf("Superpixel %d connected to superpixel %d\n", neighbor, it->first);
							}
						}
					}
				}
			}

			return 0;
		}

		int Workspace::ConnectSuperpixelsCam(const float THRESH_1,
			const float THRESH_2,
			const DepthMap& depth_map,
			const float* K_inv_arr,
			std::unordered_map<int, std::vector<float>>& plane_map,
			std::unordered_map<int, std::vector<float>>& eigen_vals_map,
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, cv::Point3f>& center_map,
			cv::Mat& labels,
			std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map)
		{
			// 遍历superpxiel
			for (auto it = label_map.begin(); it != label_map.end(); ++it)
			{
				// 获取superpixel拟合平面的维度
				int dim_1 = this->GetPlaneDimForSuperpixel(eigen_vals_map[it->first], THRESH_1);

				// TODO: 对于dim < 2的superpixel扩展区域重新计算切平面...

				// 排除"维度"为3的
				if (3 == dim_1)
				{
					continue;
				}
				else
				{
					const cv::Point3f& center_1 = center_map[it->first];
					const float* normal_1 = plane_normal_map[it->first].data();

					// 更新superpixel邻接表
					auto NeighborMap = GetNeighborMap(labels);

					// 连接所有"可连接"的邻接superpixel
					for (int neighbor : NeighborMap[it->first])
					{
						int dim_2 = this->GetPlaneDimForSuperpixel(eigen_vals_map[neighbor],
							THRESH_1);
						if (3 == dim_2)
						{
							continue;
						}
						else  // ----- 符合可连接的第一个条件
						{
							const cv::Point3f& center_2 = center_map[neighbor];
							const float* normal_2 = plane_normal_map[neighbor].data();

							float abs_normal_dot = fabs(
								normal_1[0] * normal_2[0] +
								normal_1[1] * normal_2[1] +
								normal_1[2] * normal_2[2]);
							float abs_dot_1 = fabsf(
								normal_1[0] * (center_1.x - center_2.x) +
								normal_1[1] * (center_1.y - center_2.y) +
								normal_1[2] * (center_1.z - center_2.z));
							float abs_dot_2 = fabsf(
								normal_2[0] * (center_1.x - center_2.x) +
								normal_2[1] * (center_1.y - center_2.y) +
								normal_2[2] * (center_1.z - center_2.z));

							// 取较大值
							float numerator = abs_dot_1 >= abs_dot_2 ? abs_dot_1 : abs_dot_2;

							float connectivity = numerator / abs_normal_dot;
							if (connectivity > THRESH_2)  // 如果"连接度"超过阈值
							{
								continue;
							}

							// ----- 符合可连接的第二个条件, 连接这两个superpixel
							else  // 连接这两个superpixel: neighbor -> it->first 
							{
								// 合并更新labels
								for (auto pt : label_map[neighbor])
								{
									labels.at<int>((int)pt.y, (int)pt.x) = it->first;
								}
								// 合并label_map
								for (auto pt : label_map[neighbor])
								{
									label_map[it->first].push_back(pt);
								}
								// 合并更新has_depth_map
								for (auto pt : has_depth_map[neighbor])
								{
									has_depth_map[it->first].push_back(pt);
								}
								// 合并has_no_depth_map
								for (auto pt : has_no_depth_map[neighbor])
								{
									has_no_depth_map[it->first].push_back(pt);
								}

								// ----- Connect之后, 更新这个superpixel的拟合平面
								std::vector<float> center_arr(3, 0.0f);

								// --- 更新plane_normal_map和eigen_vals_map
								// 提取有深度值的2D点
								const std::vector<cv::Point2f>& Pts2D = has_depth_map[it->first];

								// connect之后重新拟合切平面
								this->FitPlaneForSuperpixelCam(depth_map,
									K_inv_arr,
									Pts2D,
									plane_map[it->first],
									plane_normal_map[it->first],
									eigen_vals_map[it->first],
									center_arr);

								// 更新center_map
								center_map[it->first].x = center_arr[0];
								center_map[it->first].y = center_arr[1];
								center_map[it->first].z = center_arr[2];

								// ---删除neighbor
								// 删除plane_arr的neighbor
								if (plane_map.find(neighbor) != plane_map.end())
								{
									plane_map.erase(neighbor);
								}

								// 删除plane_normal_map中的neighbor
								if (plane_normal_map.find(neighbor) != plane_normal_map.end())
								{
									plane_normal_map.erase(neighbor);
								}

								// 删除eigen_vals_map中的neighbor
								if (eigen_vals_map.find(neighbor) != eigen_vals_map.end())
								{
									eigen_vals_map.erase(neighbor);
								}

								// 删除center_map中的neighbor
								if (center_map.find(neighbor) != center_map.end())
								{
									center_map.erase(neighbor);
								}

								// 删除label_map中的neighbor
								if (label_map.find(neighbor) != label_map.end())
								{
									label_map.erase(neighbor);
								}

								// 删除has_depth_map中的neighbor
								if (has_depth_map.find(neighbor) != has_depth_map.end())
								{
									has_depth_map.erase(neighbor);
								}

								// 删除has_no_depth_map中的neighbor
								if (has_no_depth_map.find(neighbor) != has_no_depth_map.end())
								{
									has_no_depth_map.erase(neighbor);
								}

								//printf("Superpixel %d connected to superpixel %d\n", neighbor, it->first);
							}
						}
					}
				}
			}

			return 0;
		}

		// 依据filtered depth map合并superpixel
		int Workspace::MergeSuperpixels(const cv::Mat& src,
			const int MinNum,
			cv::Mat& labels,
			std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map)
		{
			// 遍历superpixel提取有效深度值个数少于阈值的superpixel
			// hash_map一边迭代一边删除
			//for (auto it = has_depth_map.begin(); it != has_depth_map.end(); it++)
			for (auto it = label_map.begin(); it != label_map.end(); it++)
			{
				// 如果superpixel存在有效的深度值
				if (has_depth_map.find(it->first) != has_depth_map.end())
				{
					while ((int)has_depth_map[it->first].size() < MinNum)
					{
						// 依据labels更新邻接表
						std::unordered_map<int, std::set<int>> NeighborMap = GetNeighborMap(labels);

						// 遍历邻接superpixel, 搜索巴氏距离最小的邻接superpixel
						float dist_min = FLT_MAX;  // 距离初始化为最大值
						int best_neigh = -1;
						for (int neigh : NeighborMap[it->first])  // 依据邻接表
						{
							// 取superpixel的像素值, 计算巴氏距离
							float dist = BaDistOf2Superpixel(src, label_map[it->first], label_map[neigh]);
							if (dist < dist_min)
							{
								dist_min = dist;
								best_neigh = neigh;
							}
						}

						// -----合并it->first和best_neigh
						// ---将best_neigh的数据转移到it->first
						// 合并labels
						for (auto pt : label_map[best_neigh])
						{
							labels.at<int>((int)pt.y, (int)pt.x) = it->first;
						}
						// 合并label_map
						for (auto pt : label_map[best_neigh])
						{
							label_map[it->first].push_back(pt);
						}
						// 合并has_Depth_map
						for (auto pt : has_depth_map[best_neigh])
						{
							has_depth_map[it->first].push_back(pt);  // 更新it->first的有效深度点
						}
						// 合并has_no_depth_map
						for (auto pt : has_no_depth_map[best_neigh])
						{
							has_no_depth_map[it->first].push_back(pt);
						}

						// ---删除best_neigh
						// 删除label_map中的best_neigh
						if (label_map.find(best_neigh) != label_map.end())
						{
							label_map.erase(best_neigh);
						}
						// 删除has_depth_map中的best_neigh
						if (has_depth_map.find(best_neigh) != has_depth_map.end())
						{
							has_depth_map.erase(best_neigh);
						}
						// 删除has_no_depth_map中的best_neigh
						if (has_no_depth_map.find(best_neigh) != has_no_depth_map.end())
						{
							has_no_depth_map.erase(best_neigh);
						}

						//printf("Superpixel %d merged to superpixel %d\n", best_neigh, it->first);
					}
				}
				else if (has_no_depth_map.find(it->first) != has_no_depth_map.end())
				{
					while (has_depth_map.find(it->first) == has_depth_map.end()
						|| (int)has_depth_map[it->first].size() < MinNum)
					{
						// 依据labels更新邻接表
						std::unordered_map<int, std::set<int>> NeighborMap = GetNeighborMap(labels);

						// 遍历邻接superpixel, 搜索巴氏距离最小的邻接superpixel
						float dist_min = FLT_MAX;  // 距离初始化为最大值
						int best_neigh = -1;
						for (int neigh : NeighborMap[it->first])  // 依据邻接表
						{
							// 取superpixel的像素值, 计算巴氏距离
							float dist = BaDistOf2Superpixel(src, label_map[it->first], label_map[neigh]);
							if (dist < dist_min)
							{
								dist_min = dist;
								best_neigh = neigh;
							}
						}

						// -----合并it->first和best_neigh
						// ---将best_neigh的数据转移到it->first
						// 合并labels
						for (auto pt : label_map[best_neigh])
						{
							labels.at<int>((int)pt.y, (int)pt.x) = it->first;
						}
						// 合并label_map
						for (auto pt : label_map[best_neigh])
						{
							label_map[it->first].push_back(pt);
						}
						// 合并has_Depth_map
						for (auto pt : has_depth_map[best_neigh])
						{
							has_depth_map[it->first].push_back(pt);  // 更新it->first的有效深度点
						}
						// 合并has_no_depth_map
						for (auto pt : has_no_depth_map[best_neigh])
						{
							has_no_depth_map[it->first].push_back(pt);
						}

						// ---删除best_neigh
						// 删除label_map中的best_neigh
						if (label_map.find(best_neigh) != label_map.end())
						{
							label_map.erase(best_neigh);
						}
						// 删除has_depth_map中的best_neigh
						if (has_depth_map.find(best_neigh) != has_depth_map.end())
						{
							has_depth_map.erase(best_neigh);
						}
						// 删除has_no_depth_map中的best_neigh
						if (has_no_depth_map.find(best_neigh) != has_no_depth_map.end())
						{
							has_no_depth_map.erase(best_neigh);
						}

						//printf("Superpixel %d merged to superpixel %d\n", best_neigh, it->first);
					}

				}
				else
				{
					printf("[Err]: superpixel %d not included \
							in has_depth_map and has_no_depth_map\n", it->first);
					return -1;
				}
			}

			return 0;
		}

		int Workspace::DrawMaskOfSuperpixels(const cv::Mat& labels, cv::Mat& Input)
		{
			// 遍历labels
			for (int y = 0; y < labels.rows; ++y)
			{
				for (int x = 0; x < labels.cols; ++x)
				{
					const int& label = labels.at<int>(y, x);

					int label_up = label;
					if (y > 0)
					{
						label_up = labels.at<int>(y - 1, x);
					}

					int label_down = label;
					if (y < labels.rows - 1)
					{
						label_down = labels.at<int>(y + 1, x);
					}

					int label_left = label;
					if (x > 0)
					{
						label_left = labels.at<int>(y, x - 1);
					}

					int label_right = label;
					if (x < labels.cols - 1)
					{
						label_right = labels.at<int>(y, x + 1);
					}

					// 根据四邻域, 判断是否是边界
					if (label_up != label || label_down != label
						|| label_left != label || label_right != label)
					{
						cv::Vec3b& bgr = Input.at<cv::Vec3b>(y, x);
						bgr[0] = 255;  // 将边界处绘制为蓝色
						bgr[1] = 0;
						bgr[2] = 0;
					}
				}
			}

			return 0;
		}

		std::unordered_map<int, std::set<int>> Workspace::GetNeighborMap(const cv::Mat& labels)
		{
			std::unordered_map<int, std::set<int>> NeighborMap;

			// 为每一个像素统计四邻域是否是mask border
			// 生成无向邻接表
			for (int y = 0; y < labels.rows; ++y)
			{
				for (int x = 0; x < labels.cols; ++x)
				{
					const int& center = labels.at<int>(y, x);

					// 中间部分
					if (y > 0 && x > 0 && y < labels.rows - 1 && x < labels.cols - 1)
					{
						const int& up = labels.at<int>(y - 1, x);
						const int& down = labels.at<int>(y + 1, x);
						const int& left = labels.at<int>(y, x - 1);
						const int& right = labels.at<int>(y, x + 1);

						// 根据标签一致性生成无向邻接表
						if (up != center)
						{
							NeighborMap[center].insert(up);
						}
						if (down != center)
						{
							NeighborMap[center].insert(down);
						}
						if (left != center)
						{
							NeighborMap[center].insert(left);
						}
						if (right != center)
						{
							NeighborMap[center].insert(right);
						}
					}
					else if (y == 0 && x == 0)  // 左上角
					{
						const int& right = labels.at<int>(y, x + 1);
						const int& down = labels.at<int>(y + 1, x);
						if (right != center)
						{
							NeighborMap[center].insert(right);
						}
						if (down != center)
						{
							NeighborMap[center].insert(down);
						}
					}
					else if (y == 0 && x == labels.cols - 1)  // 右上角
					{
						const int& left = labels.at<int>(y, x - 1);
						const int& down = labels.at<int>(y + 1, x);
						if (left != center)
						{
							NeighborMap[center].insert(left);
						}
						if (down != center)
						{
							NeighborMap[center].insert(down);
						}
					}
					else if (y == 0 && x > 0 && x < labels.cols - 1)  // 第一行中间
					{
						const int& left = labels.at<int>(y, x - 1);
						const int& right = labels.at<int>(y, x + 1);
						const int& down = labels.at<int>(y + 1, x);
						if (left != center)
						{
							NeighborMap[center].insert(left);
						}
						if (right != center)
						{
							NeighborMap[center].insert(right);
						}
						if (down != center)
						{
							NeighborMap[center].insert(down);
						}
					}
					else if (y == labels.rows - 1 && x > 0 && x < labels.cols - 1)  // 最后一行中间
					{
						const int& up = labels.at<int>(y - 1, x);
						const int& left = labels.at<int>(y, x - 1);
						const int& right = labels.at<int>(y, x + 1);
						if (up != center)
						{
							NeighborMap[center].insert(up);
						}
						if (left != center)
						{
							NeighborMap[center].insert(left);
						}
						if (right != center)
						{
							NeighborMap[center].insert(right);
						}
					}
					else if (x == 0 && y == labels.rows - 1)  // 左下角
					{
						const int& right = labels.at<int>(y, x + 1);
						const int& up = labels.at<int>(y - 1, x);
						if (right != center)
						{
							NeighborMap[center].insert(right);
						}
						if (up != center)
						{
							NeighborMap[center].insert(up);
						}
					}
					else if (x == 0 && y > 0 && y < labels.rows - 1)  // 第一列中间
					{
						const int& up = labels.at<int>(y - 1, x);
						const int& down = labels.at<int>(y + 1, x);
						const int& right = labels.at<int>(y, x + 1);
						if (up != center)
						{
							NeighborMap[center].insert(up);
						}
						if (down != center)
						{
							NeighborMap[center].insert(down);
						}
						if (right != center)
						{
							NeighborMap[center].insert(right);
						}
					}
					else if (x == labels.cols - 1 && y == labels.rows - 1)  // 右下角
					{
						const int& left = labels.at<int>(y, x - 1);
						const int& up = labels.at<int>(y - 1, x);
						if (left != center)
						{
							NeighborMap[center].insert(left);
						}
						if (up != center)
						{
							NeighborMap[center].insert(up);
						}
					}
					else if (x == labels.cols - 1 && y > 0 && y < labels.rows - 1)  // 最后一列中间
					{
						const int& up = labels.at<int>(y - 1, x);
						const int& down = labels.at<int>(y + 1, x);
						const int& left = labels.at<int>(y, x - 1);
						if (up != center)
						{
							NeighborMap[center].insert(up);
						}
						if (down != center)
						{
							NeighborMap[center].insert(down);
						}
						if (left != center)
						{
							NeighborMap[center].insert(left);
						}
					}
				}
			}

			return NeighborMap;
		}

		float Workspace::BaDistOf2Superpixel(const cv::Mat& src,
			const std::vector<cv::Point2f>& superpix1,
			const std::vector<cv::Point2f>& superpix2,
			const int num_bins)
		{
			// 统计三个通道BGR的直方图
			std::vector<int> b1, g1, r1, b2, g2, r2;
			b1.reserve(superpix1.size());
			g1.reserve(superpix1.size());
			r1.reserve(superpix1.size());
			b1.resize(superpix1.size());
			g1.resize(superpix1.size());
			r1.resize(superpix1.size());

			b2.reserve(superpix2.size());
			g2.reserve(superpix2.size());
			r2.reserve(superpix2.size());
			b2.resize(superpix2.size());
			g2.resize(superpix2.size());
			r2.resize(superpix2.size());

			// ----- 提取BGR通道intensity
			for (int i = 0; i < (int)superpix1.size(); ++i)
			{
				const cv::Point2f& pt = superpix1[i];
				const cv::Vec3b& bgr = src.at<cv::Vec3b>((int)pt.y, (int)pt.x);
				b1[i] = bgr[0];
				g1[i] = bgr[1];
				r1[i] = bgr[2];
			}
			for (int i = 0; i < (int)superpix2.size(); ++i)
			{
				const cv::Point2f& pt = superpix2[i];
				const cv::Vec3b& bgr = src.at<cv::Vec3b>((int)pt.y, (int)pt.x);
				b2[i] = bgr[0];
				g2[i] = bgr[1];
				r2[i] = bgr[2];
			}

			// ---- 计算直方图的等差数列
			std::vector<float> linspace_b1(num_bins),
				linspace_g1(num_bins),
				linspace_r1(num_bins),
				linspace_b2(num_bins),
				linspace_g2(num_bins),
				linspace_r2(num_bins);

			Linspace(b1, num_bins, linspace_b1);
			Linspace(g1, num_bins, linspace_g1);
			Linspace(r1, num_bins, linspace_r1);

			Linspace(b2, num_bins, linspace_b2);
			Linspace(g2, num_bins, linspace_g2);
			Linspace(r2, num_bins, linspace_r2);

			// ----- 计算直方图频数
			std::vector<float> histgram_b1(num_bins),
				histgram_g1(num_bins),
				histgram_r1(num_bins),
				histgram_b2(num_bins),
				histgram_g2(num_bins),
				histgram_r2(num_bins);

			for (int i = 0; i < (int)superpix1.size(); ++i)
			{
				for (int j = 0; j < num_bins; ++j)
				{
					if (b1[i] >= linspace_b1[j] && b1[i] < linspace_b1[j + 1])
					{
						histgram_b1[j] += 1.0f;
					}
					else if (b1[i] == linspace_b1[j + 1])
					{
						histgram_b1[j] += 1.0f;
					}

					if (g1[i] >= linspace_g1[j] && g1[i] < linspace_g1[j + 1])
					{
						histgram_g1[j] += 1.0f;
					}
					else if (g1[i] == linspace_g1[j + 1])
					{
						histgram_g1[j] += 1.0f;
					}

					if (r1[i] >= linspace_r1[j] && r1[i] < linspace_r1[j + 1])
					{
						histgram_r1[j] += 1.0f;
					}
					else if (r1[i] == linspace_r1[j + 1])
					{
						histgram_r1[j] += 1.0f;
					}
				}
			}

			for (int i = 0; i < (int)superpix2.size(); ++i)
			{
				for (int j = 0; j < num_bins; ++j)
				{
					if (b2[i] >= linspace_b2[j] && b2[i] < linspace_b2[j + 1])
					{
						histgram_b2[j] += 1.0f;
					}
					else if (b2[i] == linspace_b2[j + 1])
					{
						histgram_b2[j] += 1.0f;
					}

					if (g2[i] >= linspace_g2[j] && g2[i] < linspace_g2[j + 1])
					{
						histgram_g2[j] += 1.0f;
					}
					else if (g2[i] == linspace_g2[j + 1])
					{
						histgram_g2[j] += 1.0f;
					}

					if (r2[i] >= linspace_r2[j] && r2[i] < linspace_r2[j + 1])
					{
						histgram_r2[j] += 1.0f;
					}
					else if (r2[i] == linspace_r2[j + 1])
					{
						histgram_r2[j] += 1.0f;
					}
				}
			}

			// 计算直方图频率
			for (int i = 0; i < num_bins; ++i)
			{
				histgram_b1[i] /= float(superpix1.size());
				histgram_g1[i] /= float(superpix1.size());
				histgram_r1[i] /= float(superpix1.size());

				histgram_b2[i] /= float(superpix1.size());
				histgram_g2[i] /= float(superpix1.size());
				histgram_r2[i] /= float(superpix1.size());
			}

			// 直方图归一化
			float sum_hist_b1 = std::accumulate(histgram_b1.begin(), histgram_b1.end(), 0.0f);
			float sum_hist_g1 = std::accumulate(histgram_g1.begin(), histgram_g1.end(), 0.0f);
			float sum_hist_r1 = std::accumulate(histgram_r1.begin(), histgram_r1.end(), 0.0f);

			float sum_hist_b2 = std::accumulate(histgram_b2.begin(), histgram_b2.end(), 0.0f);
			float sum_hist_g2 = std::accumulate(histgram_g2.begin(), histgram_g2.end(), 0.0f);
			float sum_hist_r2 = std::accumulate(histgram_r2.begin(), histgram_r2.end(), 0.0f);

			for (int i = 0; i < num_bins; ++i)
			{
				histgram_b1[i] /= sum_hist_b1;
				histgram_g1[i] /= sum_hist_g1;
				histgram_r1[i] /= sum_hist_r1;

				histgram_b2[i] /= sum_hist_b2;
				histgram_g2[i] /= sum_hist_g2;
				histgram_r2[i] /= sum_hist_r2;
			}

			// ----- 计算三通道巴氏距离
			float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f;
			for (int i = 0; i < num_bins; ++i)
			{
				sum_b += sqrtf(histgram_b1[i] * histgram_b2[i]);
				sum_g += sqrtf(histgram_g1[i] * histgram_g2[i]);
				sum_r += sqrtf(histgram_r1[i] * histgram_r2[i]);
			}
			float B_b = -log2f(sum_b);
			float B_g = -log2f(sum_g);
			float B_r = -log2f(sum_r);

			// 返回三通道巴氏距离均值
			return (B_b + B_g + B_r) / 3.0f;
		}

		// 合并两种depth, normal maps(src and enhance)，并且进行后续的有选择性联合双边传播插值
		void Workspace::MergeDepthNormalMaps(const bool is_merged, const bool is_sel_JBPF)
		{
			////指定几种深度图和法向量图的路径
			// 原始图像
			const string DepthPath_src = options_.workspace_path + "/SrcMVS/depth_maps/dslr_images_undistorted/";
			const string NormalPath_src = options_.workspace_path + "/SrcMVS/normal_maps/dslr_images_undistorted/";

			// 细节增强图像
			//const string DepthPath_detailEnhance = options_.workspace_path + "/detailEnhance/depth_maps/dslr_images_undistorted/";
			//const string NormalPath_detailEnhance = options_.workspace_path + "/detailEnhance/normal_maps/dslr_images_undistorted/";

			// 结构增强图像
			const string DepthPath_structEnhance = options_.workspace_path + "/EnhanceMVS/depth_maps/dslr_images_undistorted/";
			const string NormalPath_structEnhance = options_.workspace_path + "/EnhanceMVS/normal_maps/dslr_images_undistorted/";

			// 合并几种深度和法向量图的结果路径
			const string resultDepthPath = options_.workspace_path + "/result/depth_maps/";
			const string resultNormalPath = options_.workspace_path + "/result/normal_maps/";

			// 对合并结果，进行有选择性联合双边传播插值结果路径
			const string resultProDepthPath = options_.workspace_path + "/resultPro/depth_maps/";
			const string resultProNormalPath = options_.workspace_path + "/resultPro/normal_maps/";

			// 原始彩色图像路径
			const string srcColorImgPath = options_.workspace_path + "/SrcMVS/images/dslr_images_undistorted/";

			clock_t T_start, T_end;

			// 遍历每一张图
			for (int img_id = 0; img_id < m_model.m_images.size(); img_id++)
			{
				const string DepthAndNormalName = GetFileName(img_id, true);

				// 如果还没有合并过，那么合并
				if (!is_merged)
				{
					T_start = clock();

					// 分别读取深度图和法向量图
					DepthMap depthMap_src(depth_ranges_.at(img_id).first,
						depth_ranges_.at(img_id).second),
						//depthMap_detailEnhance(depth_ranges_.at(image_id).first,
						//depth_ranges_.at(image_id).second),
						depthMap_structEnhance(depth_ranges_.at(img_id).first,
							depth_ranges_.at(img_id).second);

					depthMap_src.ReadBinary(DepthPath_src + DepthAndNormalName);
					//depthMap_detailEnhance.ReadBinary(DepthPath_detailEnhance + DepthAndNormalName);
					depthMap_structEnhance.ReadBinary(DepthPath_structEnhance + DepthAndNormalName);

					// normal maps
					NormalMap normalMap_src, normalMap_detailEnhance, normalMap_structEnhance;

					normalMap_src.ReadBinary(NormalPath_src + DepthAndNormalName);
					//normalMap_detailEnhance.ReadBinary(NormalPath_detailEnhance + DepthAndNormalName);
					normalMap_structEnhance.ReadBinary(NormalPath_structEnhance + DepthAndNormalName);

					// write BitMap to local
					const auto& depthMap_path = DepthPath_src + DepthAndNormalName + ".jpg";
					const auto& normalMap_path = NormalPath_src + DepthAndNormalName + ".jpg";

					imwrite(depthMap_path, depthMap_src.ToBitmapGray(2, 98));
					imwrite(normalMap_path, normalMap_src.ToBitmap());
					//imwrite(DepthPath_detailEnhance + DepthAndNormalName + ".jpg", depthMap_detailEnhance.ToBitmapGray(2, 98));
					//imwrite(NormalPath_detailEnhance + DepthAndNormalName + ".jpg", normalMap_detailEnhance.ToBitmap());

					const auto& depthMap_path_struct_enhance = DepthPath_structEnhance + DepthAndNormalName + ".jpg";
					const auto& normalMap_path_struct_enhance = NormalPath_structEnhance + DepthAndNormalName + ".jpg";

					imwrite(depthMap_path_struct_enhance, depthMap_structEnhance.ToBitmapGray(2, 98));
					imwrite(normalMap_path_struct_enhance, normalMap_structEnhance.ToBitmap());

					//depthMaps_.at(image_id) = depthMap_structureEnhance;
					//normalMaps_.at(image_id) = normalMap_structureEnhance;
					//hasReadMapsGeom_= true;

					// @even Fusion深度图, 法向图选择宽高较小的作为自己的宽高
					const int src_width = depthMap_src.GetWidth();
					const int enhance_width = depthMap_structEnhance.GetWidth();
					const int src_height = depthMap_src.GetHeight();
					const int enhance_height = depthMap_structEnhance.GetHeight();

					const int Fusion_Width = std::min(src_width, enhance_width);
					const int Fusion_Height = std::min(src_height, enhance_height);

					//const int width = depthMap_src.GetWidth();
					//const int height = depthMap_src.GetHeight();

					// 初始化Fusion的深度图, 法向图为0
					DepthMap depthMap_result(Fusion_Width, Fusion_Height,
						depth_ranges_.at(img_id).first,
						depth_ranges_.at(img_id).second);
					NormalMap normalMap_result(Fusion_Width, Fusion_Height);

					const float NON_VALUE = 0.0f;

					for (int row = 0; row < Fusion_Height; row++)
					{
						for (int col = 0; col < Fusion_Width; col++)
						{
							const float depth_src = depthMap_src.GetDepth(row, col);

							//const float depth_detailEnhance = 
							// depthMap_detailEnhance.Get(row, col);
							const float depth_structEnhance = depthMap_structEnhance.GetDepth(row, col);

							// 初始化法向值为0
							float normal_src[3],
								//normal_detailEnhance[3],
								normal_structEnhance[3],
								normal_result[3] = { 0.0f };

							normalMap_src.GetSlice(row, col, normal_src);
							//normalMap_detailEnhance.GetSlice(row, col, normal_detailEnhance);
							normalMap_structEnhance.GetSlice(row, col, normal_structEnhance);

							// 收集有用的深度和法向信息
							vector<float> depths;
							vector<float*> normals;

							// 收集有用的src深度,法向
							if (depth_src != NON_VALUE)
							{
								depths.push_back(depth_src);
								normals.push_back(normal_src);
							}

							// 收集有用的enhance深度, 法向
							int flags_se = 1;
							if (flags_se == 1 && depth_structEnhance != NON_VALUE)
							{
								depths.push_back(depth_structEnhance);
								normals.push_back(normal_structEnhance);
							}

							//int flags_de = -1;
							//if (flags_de == 1 && depth_detailEnhance != NON_VALUE)
							//{
							//	depths.push_back(depth_detailEnhance);
							//	normals.push_back(normal_detailEnhance);
							//}

							const float num_valid = depths.size();

							if (num_valid > NON_VALUE)
							{
								//// average
								if (0)
								{
									depthMap_result.Set(row,
										col,
										accumulate(depths.begin(), depths.end(), 0.0) / num_valid);

									for (int i = 0; i < num_valid; i++)
									{
										normal_result[0] += normals[i][0];
										normal_result[1] += normals[i][1];
										normal_result[2] += normals[i][2];
									}

									NormVec3(normal_result);
									normalMap_result.SetSlice(row, col, normal_result);
								}
								//// the first
								if (0)
								{
									depthMap_result.Set(row, col, depths[0]);
									normalMap_result.SetSlice(row, col, normals[0]);
								}
								//// evalution
								if (1)
								{
									if (num_valid == 1)
									{
										depthMap_result.Set(row, col, depths[0]);
										normalMap_result.SetSlice(row, col, normals[0]);
									}
									if (num_valid == 2)
									{
										// 深度误差比小于阈值：取深度值较小的
										if (abs(depths[0] - depths[1]) / depths[0] > 0.01)
										{
											depthMap_result.Set(row, col, depths[0] < depths[1]
												? depths[0] : depths[1]);
											normalMap_result.SetSlice(row, col, depths[0] < depths[1]
												? normals[0] : normals[1]);
										}
										else  // 深度取均值, 法向量求和再求L2 norm
										{
											depthMap_result.Set(row,
												col,
												(depths[0] + depths[1]) / 2.0f);

											normal_result[0] = normals[0][0] + normals[1][0];
											normal_result[1] = normals[0][1] + normals[1][1];
											normal_result[2] = normals[0][2] + normals[1][2];

											NormVec3(normal_result);
											normalMap_result.SetSlice(row, col, normal_result);
										}
									}
									if (num_valid == 3)
									{
										depthMap_result.Set(row, col, depths[0] >= depths[1] ?
											(depths[1] >= depths[2] ? depths[1] : (depths[0] >= depths[2]
												? depths[2] : depths[0])) :
												(depths[0] >= depths[2] ? depths[0] : (depths[1] >= depths[2]
													? depths[2] : depths[1])));
										normalMap_result.SetSlice(row, col, depths[0] >= depths[1] ?
											(depths[1] >= depths[2] ? normals[1] : (depths[0] >= depths[2]
												? normals[2] : normals[0])) :
												(depths[0] >= depths[2] ? normals[0] : (depths[1] >= depths[2]
													? normals[2] : normals[1])));
									}
								}
							}

						}  // col
					}  // row

					// 设置工作空间的depth, normal maps为合并、过滤后的值
					m_depth_maps.at(img_id) = depthMap_result;
					m_normal_maps.at(img_id) = normalMap_result;

					// 设置"已读": 更新几何一致性depth, normal maps的读取状态
					hasReadMapsGeom_ = true;

					// 将合并后的depth, normal maps写入workspace
					imwrite(resultDepthPath + DepthAndNormalName + ".jpg",
						depthMap_result.ToBitmapGray(2, 98));
					imwrite(resultNormalPath + DepthAndNormalName + ".jpg",
						normalMap_result.ToBitmap());

					depthMap_result.WriteBinary(resultDepthPath + DepthAndNormalName);
					normalMap_result.WriteBinary(resultNormalPath + DepthAndNormalName);

					T_end = clock();
					std::cout << "Merge image:" << img_id << " Time:" << (float)(T_end - T_start) / CLOCKS_PER_SEC << "s" << endl;
				}

				// 有选择性联合双边传播插值
				if (is_sel_JBPF)
				{
					T_start = clock();

					// 如果之前合并过Map图了，直接从文件中读取就行了
					if (is_merged)
					{
						// 读取深度图和法向量图
						DepthMap depthMap(depth_ranges_.at(img_id).first,
							depth_ranges_.at(img_id).second);
						depthMap.ReadBinary(resultDepthPath + DepthAndNormalName);

						NormalMap normalMap;
						normalMap.ReadBinary(resultNormalPath + DepthAndNormalName);

						m_depth_maps.at(img_id) = depthMap;
						m_normal_maps.at(img_id) = normalMap;
					}

					// 传播后结果
					DepthMap depthMap_pro = m_depth_maps.at(img_id);
					NormalMap normalMap_pro = m_normal_maps.at(img_id);

					// 读取原彩色图像并resize
					const auto& src_img_path = srcColorImgPath + m_model.GetImageName(img_id);
					cv::Mat src_img = imread(src_img_path);
					resize(src_img,
						src_img,
						Size(m_depth_maps.at(img_id).GetWidth(),
							m_depth_maps.at(img_id).GetHeight()));

					// 选择双边传播滤波
					this->selJointBilateralPropagateFilter(src_img,
						this->m_depth_maps.at(img_id),
						this->m_normal_maps.at(img_id),
						this->m_model.m_images.at(img_id).GetK(),
						25, 10,  // 25, 10
						-1, 16,
						depthMap_pro, normalMap_pro);

					//// 迭代SelJointBilateralPropagateFilter
					//int sigma_color = 23, sigma_space = 7;
					//for (int iter_i = 0; iter_i < 3; ++iter_i)
					//{
					//	// 选择双边传播滤波
					//	this->selJointBilateralPropagateFilter(src_img,
					//		this->m_depth_maps.at(img_id),
					//		this->m_normal_maps.at(img_id),
					//		model_.m_images.at(img_id).GetK(),
					//		sigma_color, sigma_space,  // 25, 10
					//		-1, 16,
					//		depthMap_pro, normalMap_pro);

					//	// 动态调整
					//	sigma_color += 1;
					//	sigma_space += 1;

					//	// 设置工作空间的depth, normal maps为合并、过滤后的值
					//	this->m_depth_maps.at(img_id) = depthMap_pro;
					//	this->m_normal_maps.at(img_id) = normalMap_pro;
					//}

					const int num_iter = 1;  // 迭代次数
					const double sigma_space = 5.0, sigma_color = 5.0, sigma_depth = 5.0;
					const float THRESH = 0.00f, eps = 1.0f, tau = 0.3f;   // 超参数 
					const bool is_propagate = false;   // 是否使用传播深度值
					for (int iter_i = 0; iter_i < num_iter; ++iter_i)
					{
						this->NoiseAwareFilter(src_img,
							this->m_depth_maps.at(img_id),
							this->m_normal_maps.at(img_id),
							m_model.m_images.at(img_id).GetK(),
							sigma_space, sigma_color, sigma_depth,
							THRESH,
							eps, tau,
							is_propagate,
							25,  // radius, window_size: 2*d + 1
							depthMap_pro, normalMap_pro);

						this->m_depth_maps.at(img_id) = depthMap_pro;
						this->m_normal_maps.at(img_id) = normalMap_pro;

						//// 写入中间结果..
						//if (iter_i % 10 == 0 || iter_i == NUM_ITER - 1)
						//{
						//	char buff[100];
						//	sprintf(buff, "_iter%d.jpg", iter_i);
						//	imwrite(std::move(resultProDepthPath + DepthAndNormalName + string(buff)),
						//		depthMap_pro.ToBitmapGray(2, 98));
						//}
					}

					//const int NUM_ITER = 1;  // 迭代次数
					//const double sigma_space = 1.5, sigma_color = 0.09;
					//double sigma_depth = 0.02;
					//const float THRESH = 0.06f;   // 超参数 
					//const bool is_propagate = false;   // 是否使用传播深度值
					//for (int iter_i = 0; iter_i < NUM_ITER; ++iter_i)
					//{
					//	this->JTU(src_img,
					//		this->m_depth_maps.at(img_id),
					//		this->m_normal_maps.at(img_id),
					//		model_.m_images.at(img_id).GetK(),
					//		sigma_space, sigma_color, sigma_depth,
					//		THRESH,
					//		is_propagate,
					//		25,  // radius, window_size: 2*d + 1
					//		depthMap_pro, normalMap_pro);

					//	this->m_depth_maps.at(img_id) = depthMap_pro;
					//	this->m_normal_maps.at(img_id) = normalMap_pro;

					//	//// 写入中间结果..
					//	//if (iter_i % 10 == 0 || iter_i == NUM_ITER - 1)
					//	//{
					//	//	char buff[100];
					//	//	sprintf(buff, "_iter%d.jpg", iter_i);
					//	//	imwrite(std::move(resultProDepthPath + DepthAndNormalName + string(buff)),
					//	//		depthMap_pro.ToBitmapGray(2, 98));
					//	//}
					//}

					// 设置"已读": 更新几何一致性depth, normal maps的读取状态
					hasReadMapsGeom_ = true;

					// 将depth, normal maps转成bitmap并写入磁盘
					imwrite(resultProDepthPath + DepthAndNormalName + ".jpg",
						depthMap_pro.ToBitmapGray(2, 98));
					imwrite(resultProNormalPath + DepthAndNormalName + ".jpg",
						normalMap_pro.ToBitmap());

					depthMap_pro.WriteBinary(resultProDepthPath + DepthAndNormalName);
					normalMap_pro.WriteBinary(resultProNormalPath + DepthAndNormalName);

					T_end = clock();

					cout << "SelectiveJBPF image:" << img_id << " Time:"
						<< (float)(T_end - T_start) / CLOCKS_PER_SEC << "s" << endl;
				}

			}  // end of image_id
		}

		// 有选择性的联合双边传播滤波
		void Workspace::selJointBilateralPropagateFilter(const cv::Mat& joint,
			const DepthMap& depthMap,
			const NormalMap& normalMap,
			const float* refK,
			const double sigma_color, const double sigma_space,
			int radius, const int topN,
			DepthMap& outDepthMap, NormalMap& outNormalMap) const
		{
			const int MapWidth = depthMap.GetWidth();
			const int MapHeight = depthMap.GetHeight();

			if (radius <= 0)
			{
				radius = round(sigma_space * 1.5);  // original parameters, 根据 sigma_space 计算 radius  
			}

			//assert(radius % 2 == 1);  // 确保窗口尺寸是奇数
			const int d = 2 * radius + 1;

			// 原联合图像的通道数
			const int channels = joint.channels();

			//float *color_weight = new float[cnj * 256];
			//float *space_weight = new float[d*d];
			//int *space_ofs_row = new int[d*d];  // 坐标的差值
			//int *space_ofs_col = new int[d*d];

			vector<float> color_weight(channels * 256);
			vector<float> space_weight(d * d);
			vector<int> space_offsets_row(d * d), space_offsets_col(d * d);

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
			// initialize color-related bilateral filter coefficients

			// 色差的高斯权重  
			for (int i = 0; i < 256 * channels; i++)
			{
				color_weight[i] = std::expf(i * i * gauss_color_coeff);
			}

			int MAX_K = 0;   // 0 ~ (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			//空间差的高斯权重
			// 统计满足距离的像素数量:计算方形的最大内切圆形区域
			for (int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					double r = std::sqrt((double)i * i + (double)j * j);

					if (r > radius)
					{
						continue;
					}

					space_weight[MAX_K] = (float)std::exp(r * r * gauss_space_coeff);
					space_offsets_row[MAX_K] = i;
					space_offsets_col[MAX_K++] = j;  // update MAX_K
				}
			}

			//selective joint bilataral propagation filter
			for (int y = 0; y < MapHeight; y++)
			{
				for (int x = 0; x < MapWidth; x++)
				{
					// 跳过有深度值(深度值非零)的像素
					if (depthMap.GetDepth(y, x) != 0.0f)
					{
						continue;
					}

					// bgr
					const cv::Vec3b color_0 = joint.ptr<cv::Vec3b>(y)[x];

					// 储存权重和所在位置的索引
					vector<pair<float, int>> weightAndIndex;
					weightAndIndex.clear();
					for (int k = 0; k < MAX_K; k++)
					{
						const int yy = y + space_offsets_row[k];
						const int xx = x + space_offsets_col[k];

						// 判断q, 需要q也有深度值
						if (yy < 0 || yy >= MapHeight || xx < 0
							|| xx >= MapWidth || depthMap.GetDepth(yy, xx) == 0.0f)
						{
							continue;
						}

						//颜色距离权重，是作用在高分辨率图像上的
						cv::Vec3b color_1 = joint.ptr<cv::Vec3b>(yy)[xx];

						// 根据joint当前像素和邻域像素的 距离权重 和 色差权重，计算综合的权重
						const float& the_color_weight = color_weight[abs(color_0[0] - color_1[0]) +
							abs(color_0[1] - color_1[1]) + abs(color_0[2] - color_1[2])];
						float w = space_weight[k] * the_color_weight;

						//只利用space距离作为权重!!!!!!
						//float w = space_weight[k];

						weightAndIndex.push_back(make_pair(w, k));
					}

					// 如果权重值为空
					if (weightAndIndex.size() == 0)
					{
						continue;
					}
					//if (weightAndIndex.size() < int(0.1f * (float)space_offsets_row.size()))
					//{
					//	continue;
					//}

					//对存储的权重进行从大到小排序
					if (topN < weightAndIndex.size())
					{
						partial_sort(weightAndIndex.begin(),
							weightAndIndex.begin() + topN,
							weightAndIndex.end(),
							pairIfDescend);
					}
					else
					{
						sort(weightAndIndex.begin(), weightAndIndex.end(), pairIfDescend);
					}

					//if (weightAndIndex[0].first < 0.3)
					//	continue;

					// 按照从大到小的权重，进行深度传播
					float sum_w = 0.0f;
					float sum_value_depth = 0.0f;
					float sum_value_normal[3] = { 0.0f };

					const int EffNum = std::min(topN, (int)weightAndIndex.size());
					for (int i = 0; i < EffNum; i++)
					{
						//if (weightAndIndex[i].first < 0.3)
						//	continue;

						int yy = y + space_offsets_row[weightAndIndex[i].second];
						int xx = x + space_offsets_col[weightAndIndex[i].second];

						const float src_depth = depthMap.GetDepth(yy, xx);

						float src_normal[3];
						normalMap.GetSlice(yy, xx, src_normal);

						/****************深度值传播方案****************/

						// 计算深度传播值
						float propagated_depth = PropagateDepth(refK,
							src_depth, src_normal,
							yy, xx, y, x);

						// 不传播，直接用原深度值
						//const float propagated_depth = src_depth;

						sum_value_depth += propagated_depth * weightAndIndex[i].first;

						sum_value_normal[0] += src_normal[0] * weightAndIndex[i].first;
						sum_value_normal[1] += src_normal[1] * weightAndIndex[i].first;
						sum_value_normal[2] += src_normal[2] * weightAndIndex[i].first;

						sum_w += weightAndIndex[i].first;
					}

					if (sum_w < 1e-8)
					{
						//cout << "[Warning]: very small sum_w: " << sum_w << endl;
						sum_w += float(1e-8);
						//continue;
					}

					sum_w = 1.0f / sum_w;

					// 设置深度值
					const float out_depth = sum_value_depth * sum_w;

					//// @even DEBUG: to check for Nan dpeth
					//if (isnan(out_depth))
					//{
					//	cout << "\n[Nan out depth]: " << out_depth << endl;
					//}

					outDepthMap.Set(y, x, out_depth);

					// 设置法向值
					sum_value_normal[0] *= sum_w;
					sum_value_normal[1] *= sum_w;
					sum_value_normal[2] *= sum_w;

					// 法向向量
					SuitNormal(y, x, refK, sum_value_normal);
					outNormalMap.SetSlice(y, x, sum_value_normal);

				}  // end of x
			}  // end of y

		}

		void Workspace::NoiseAwareFilter(const cv::Mat& joint,
			DepthMap& depthMap, const NormalMap& normalMap,
			const float* refK,
			const double& sigma_space, const double& sigma_color, const double& sigma_depth,
			const float& THRESH,
			const float& eps, const float& tau,
			const bool is_propagate,
			int radius,
			DepthMap& outDepthMap, NormalMap& outNormalMap) const
		{
			const int MapWidth = depthMap.GetWidth();
			const int MapHeight = depthMap.GetHeight();

			// original parameters, 根据 sigma_space 计算 radius 
			if (radius <= 0)
			{
				radius = (int)round(sigma_space * 1.5 + 0.5);
			}

			//assert(radius % 2 == 1);  // 确保窗口尺寸是奇数
			const int d = 2 * radius + 1;

			// 原联合图像的通道数
			const int channels = joint.channels();
			const int& color_levels = 256 * channels;

			// ------------ RGB原图色差, 空间距离高斯权重
			vector<float> color_weights(color_levels);
			vector<float> space_weights(d * d);
			vector<int> space_offsets_row(d * d), space_offsets_col(d * d);

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
			// initialize color-related bilateral filter coefficients

			// 色差的高斯权重  
			for (int i = 0; i < color_levels; ++i)
			{
				color_weights[i] = (float)std::exp(i * i * gauss_color_coeff);
			}

			int MAX_K = 0;   // 0 ~ (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			// 空间差的高斯权重
			// 统计满足距离的像素数量：求正方形内切圆区域
			for (int i = -radius; i <= radius; ++i)
			{
				for (int j = -radius; j <= radius; ++j)
				{
					const double r = std::sqrt((double)i * i + (double)j * j);
					if (r > radius)
					{
						continue;
					}

					space_weights[MAX_K] = (float)std::exp(r * r * gauss_space_coeff);
					space_offsets_row[MAX_K] = i;
					space_offsets_col[MAX_K++] = j;  // update MAX_K
				}
			}

			//// 计算原始深度图高斯平滑结果 
			//cv::Mat depth_mat, depth_blur;
			//depth_mat = depthMap.Depth2Mat();
			//cv::GaussianBlur(depth_mat, depth_blur, cv::Size(3, 3), 0);

			// 遍历每一个像素
			//printf("eps: %.3f, tau: %.3f\n", eps, tau);
			for (int y = 0; y < MapHeight; y++)
			{
				for (int x = 0; x < MapWidth; x++)
				{
					// 跳过有深度值(深度值非零)的像素
					if (depthMap.GetDepth(y, x) != 0.0f)
					{
						continue;
					}

					//// 计算半径区域内的omega_depth
					//double depth_min = DBL_MAX;
					//double depth_max = -1.0f;
					//for (int k = 0; k < MAX_K; ++k)
					//{
					//	const int yy = y + space_offsets_row[k];
					//	const int xx = x + space_offsets_col[k];

					//	if (yy < 0 || yy >= MapHeight || xx < 0
					//		|| xx >= MapWidth)
					//	{
					//		continue;
					//	}
					//	float depth = depth_blur.at<float>(yy, xx);
					//	if (depth == 0.0f)
					//	{
					//		continue;
					//	}

					//	if (depth > depth_max)
					//	{
					//		depth_max = depth;
					//	}
					//	else if (depth < depth_min)
					//	{
					//		depth_min = depth;
					//	}
					//}

					//// 如果不存在最小,最大深度值,没必要接下来的计算,跳过此像素
					//if (depth_min == DBL_MAX || depth_max == -1.0f)
					//{
					//	continue;
					//}

					//const double omega_depth = depth_max - depth_min;

					// p像素bgr颜色值
					const cv::Vec3b& color_0 = joint.ptr<cv::Vec3b>(y)[x];

					// p像素的深度值
					const double& depth_0 = (double)depthMap.GetDepth(y, x);

					// 统计p为中心的圆形窗口, 有效的权重及其所在位置的索引
					vector<pair<float, int>> WeightAndIndex;
					WeightAndIndex.clear();
					for (int k = 0; k < MAX_K; ++k)
					{
						const int yy = y + space_offsets_row[k];
						const int xx = x + space_offsets_col[k];

						// 判断q, 需要q也有深度值
						if (yy < 0 || yy >= MapHeight || xx < 0
							|| xx >= MapWidth || depthMap.GetDepth(yy, xx) == 0.0f)
						{
							// 跳过没有深度值的neighbor
							continue;
						}

						// q像素bgr颜色值
						cv::Vec3b color_1 = joint.ptr<cv::Vec3b>(yy)[xx];

						// q像素的深度值
						const double depth_1 = (double)depthMap.GetDepth(yy, xx);

						// 计算原始深度图深度差值的高斯函数值
						double delta_depth = depth_0 - depth_1;
						const double depth_weight = std::exp(-0.5 * delta_depth * delta_depth
							/ sigma_depth);

						// 根据joint当前像素和邻域像素的距离权重和色差权重，计算综合的权重
						const int delta_color = abs(color_0[0] - color_1[0]) +
							abs(color_0[1] - color_1[1]) + abs(color_0[2] - color_1[2]);
						const float color_weight = color_weights[delta_color];

						// 计算Alpha
						//double alpha = depthMap.CalculateAlpha(eps, tau, omega_depth);

						// 考虑根据color_weight和depth_weight的相似度确定Alpha值....
						const double delta_color_ratio = double(delta_color) / double(color_levels);
						const double delta_depth_ratio = std::abs(delta_depth) / double(depthMap.depth_max_);
						double diff = std::abs(delta_color_ratio - delta_depth_ratio);
						double alpha = std::exp(-0.5 * diff * diff / 0.2);  // to reduce sigma_alpha: 0.2

						const float compound_weight = float(alpha * color_weight + \
							(1.0f - alpha) * depth_weight);

						float weight = space_weights[k] * compound_weight;
						WeightAndIndex.push_back(make_pair(weight, k));
					}

					// 对WeightAndIndex的Size大小进行过滤
					//if (WeightAndIndex.size() == 0)
					//{
					//	continue;
					//}
					if (WeightAndIndex.size() < size_t(THRESH * (float)space_offsets_row.size()))
					{
						continue;
					}

					// 计算加权深度值和法向量
					float sum_w = 0.0f, sum_value_depth = 0.0f;
					float sum_value_normal[3] = { 0.0f };
					for (int i = 0; i < (int)WeightAndIndex.size(); ++i)
					{
						int yy = y + space_offsets_row[WeightAndIndex[i].second];
						int xx = x + space_offsets_col[WeightAndIndex[i].second];

						// neighbor q's depth
						const float src_depth = depthMap.GetDepth(yy, xx);

						// neighbor q's normal
						float src_normal[3];
						normalMap.GetSlice(yy, xx, src_normal);

						/****************深度值传播方案****************/
						float depth_val = 0.0f;
						if (is_propagate)
						{
							// 计算深度传播值
							depth_val = PropagateDepth(refK,
								src_depth, src_normal,
								yy, xx, y, x);
						}
						else
						{
							//不传播，直接用原深度值
							depth_val = src_depth;
						}

						// weighting depth
						sum_value_depth += depth_val * WeightAndIndex[i].first;

						// weighting normal
						sum_value_normal[0] += src_normal[0] * WeightAndIndex[i].first;
						sum_value_normal[1] += src_normal[1] * WeightAndIndex[i].first;
						sum_value_normal[2] += src_normal[2] * WeightAndIndex[i].first;

						sum_w += WeightAndIndex[i].first;
					}

					if (sum_w < 1e-8)
					{
						//cout << "[Warning]: very small sum_w: " << sum_w << endl;
						sum_w += float(1e-8);
						//continue;
					}

					sum_w = 1.0f / sum_w;

					// 设置深度值
					const float out_depth = sum_value_depth * sum_w;
					outDepthMap.Set(y, x, out_depth);

					// 设置法向值
					sum_value_normal[0] *= sum_w;
					sum_value_normal[1] *= sum_w;
					sum_value_normal[2] *= sum_w;

					// 法向向量
					SuitNormal(y, x, refK, sum_value_normal);
					outNormalMap.SetSlice(y, x, sum_value_normal);
				}
			}
		}

		void Workspace::JTU(const cv::Mat& joint,
			DepthMap& depthMap, const NormalMap& normalMap,
			const float* refK,
			const double& sigma_space, const double& sigma_color, double& sigma_depth,
			const float& THRESH,
			const bool is_propagate,
			int radius,
			DepthMap& outDepthMap, NormalMap& outNormalMap) const
		{
			const int MapWidth = depthMap.GetWidth();
			const int MapHeight = depthMap.GetHeight();

			// original parameters, 根据 sigma_space 计算 radius 
			if (radius <= 0)
			{
				radius = (int)round(sigma_space * 1.5 + 0.5);
			}

			//assert(radius % 2 == 1);  // 确保窗口尺寸是奇数
			const int d = 2 * radius + 1;

			// 原联合图像的通道数
			const int channels = joint.channels();
			const int& color_levels = 256 * channels;

			// ------------ RGB原图色差, 空间距离高斯权重
			vector<float> color_weights(color_levels);
			vector<float> space_weights(d * d);
			vector<int> space_offsets_row(d * d), space_offsets_col(d * d);

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
			// initialize color-related bilateral filter coefficients

			//// 色差的高斯权重  
			//for (int i = 0; i < color_levels; ++i)
			//{
			//	color_weights[i] = (float)std::exp(i * i * gauss_color_coeff);
			//}

			int MAX_K = 0;   // 0 ~ (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			// 空间差的高斯权重
			// 统计满足距离的像素数量：求正方形内切圆区域
			for (int i = -radius; i <= radius; ++i)
			{
				for (int j = -radius; j <= radius; ++j)
				{
					const double r = std::sqrt((double)i * i + (double)j * j);
					if (r > radius)
					{
						continue;
					}

					//space_weights[MAX_K] = (float)std::exp(r * r * gauss_space_coeff);
					space_offsets_row[MAX_K] = i;
					space_offsets_col[MAX_K++] = j;  // update MAX_K
				}
			}

			// 遍历每一个像素
			//printf("eps: %.3f, tau: %.3f\n", eps, tau);
			for (int y = 0; y < MapHeight; y++)
			{
				for (int x = 0; x < MapWidth; x++)
				{
					// 跳过有深度值(深度值非零)的像素
					if (depthMap.GetDepth(y, x) != 0.0f)
					{
						continue;
					}

					// p像素bgr颜色值
					const cv::Vec3b& color_0 = joint.ptr<cv::Vec3b>(y)[x];

					// p像素的深度值
					const double depth_0 = (double)depthMap.GetDepth(y, x);

					// 统计p为中心的圆形窗口, 有效的权重及其所在位置的索引
					vector<pair<float, int>> WeightAndIndex;
					WeightAndIndex.clear();
					for (int k = 0; k < MAX_K; ++k)
					{
						const int yy = y + space_offsets_row[k];
						const int xx = x + space_offsets_col[k];

						// 判断q, 需要q也有深度值
						if (yy < 0 || yy >= MapHeight || xx < 0
							|| xx >= MapWidth || depthMap.GetDepth(yy, xx) == 0.0f)
						{
							// 跳过没有深度值的neighbor
							continue;
						}

						// q像素bgr颜色值
						cv::Vec3b color_1 = joint.ptr<cv::Vec3b>(yy)[xx];

						// q像素的深度值
						const double depth_1 = (double)depthMap.GetDepth(yy, xx);

						// 计算原始深度图深度差值的高斯函数值
						double delta_depth = depth_0 - depth_1;
						delta_depth /= depthMap.depth_max_;
						if (float(color_0[0] + color_0[1] + color_0[2])
							/ float(color_levels) < 0.16f)  // threshold of sigma_depth
						{
							sigma_depth = 0.06;
						}
						const double depth_weight = std::exp(-0.5 * delta_depth * delta_depth
							/ sigma_depth);

						// 计算色差权重
						double delta_color = abs(color_0[0] - color_1[0]) +
							abs(color_0[1] - color_1[1]) + abs(color_0[2] - color_1[2]);
						delta_color /= double(color_levels);
						const double color_weight = std::exp(-0.5 * delta_color * delta_color
							/ sigma_color);
						//const float color_weight = color_weights[delta_color];

						// 计算距离权重
						//float& space_weight = space_weights[k];
						double delta_space = sqrt((x - xx) * (x - xx) + (y - yy) * (y - yy));
						//delta_space /= double(radius);
						const double space_weight = std::exp(-0.5 * delta_space * delta_space
							/ sigma_space);

						// 计算综合权重
						const float weight = space_weight * color_weight * depth_weight;
						WeightAndIndex.push_back(make_pair(weight, k));
					}

					// 对weightAndIndex的Size大小进行过滤
					//if (WeightAndIndex.size() == 0)
					//{
					//	continue;
					//}
					if (WeightAndIndex.size() < size_t(THRESH * (float)space_offsets_row.size()))
					{
						continue;
					}

					// 计算加权深度值和法向量
					float sum_w = 0.0f, sum_value_depth = 0.0f;
					float sum_value_normal[3] = { 0.0f };
					for (int i = 0; i < (int)WeightAndIndex.size(); ++i)
					{
						int yy = y + space_offsets_row[WeightAndIndex[i].second];
						int xx = x + space_offsets_col[WeightAndIndex[i].second];

						// neighbor q's depth
						const float src_depth = depthMap.GetDepth(yy, xx);

						// neighbor q's normal
						float src_normal[3];
						normalMap.GetSlice(yy, xx, src_normal);

						/****************深度值传播方案****************/
						float depth_val = 0.0f;
						if (is_propagate)
						{
							// 计算深度传播值
							depth_val = PropagateDepth(refK,
								src_depth, src_normal,
								yy, xx, y, x);
						}
						else
						{
							//不传播，直接用原深度值
							depth_val = src_depth;
						}

						// weighting depth
						sum_value_depth += depth_val * WeightAndIndex[i].first;

						// weighting normal
						sum_value_normal[0] += src_normal[0] * WeightAndIndex[i].first;
						sum_value_normal[1] += src_normal[1] * WeightAndIndex[i].first;
						sum_value_normal[2] += src_normal[2] * WeightAndIndex[i].first;

						sum_w += WeightAndIndex[i].first;
					}

					if (sum_w < 1e-8)
					{
						sum_w += float(1e-8);
					}

					sum_w = 1.0f / sum_w;

					// 设置深度值
					const float out_depth = sum_value_depth * sum_w;
					outDepthMap.Set(y, x, out_depth);

					// 设置法向值
					sum_value_normal[0] *= sum_w;
					sum_value_normal[1] *= sum_w;
					sum_value_normal[2] *= sum_w;

					// 法向向量
					SuitNormal(y, x, refK, sum_value_normal);
					outNormalMap.SetSlice(y, x, sum_value_normal);
				}
			}
		}

		//对深度图和法向量map图进行联合双边滤波
		void Workspace::jointBilateralFilter_depth_normal_maps(const cv::Mat& joint,
			const DepthMap& depthMap, const NormalMap& normalMap,
			const float *refK, const double sigma_color, const double sigma_space, int radius,
			DepthMap& outDepthMap, NormalMap& outNormalMap) const
		{
			const int mapWidth = depthMap.GetWidth();
			const int mapHeight = depthMap.GetHeight();

			if (radius <= 0)
				radius = round(sigma_space * 1.5);  // 根据 sigma_space 计算 radius  

			//assert(radius % 2 == 1);//确保窗口半径是奇数
			const int d = 2 * radius + 1;

			//原联合图像的通道数
			const int cnj = joint.channels();
			vector<float> color_weight(cnj * 256);
			vector<float> space_weight(d*d);
			vector<int> space_ofs_row(d*d), space_ofs_col(d*d);

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

			// initialize color-related bilateral filter coefficients  
			// 色差的高斯权重  
			for (int i = 0; i < 256 * cnj; i++)
				color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

			int maxk = 0;   // 0 - (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			//空间差的高斯权重
			for (int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					double r = std::sqrt((double)i * i + (double)j * j);
					if (r > radius)
						continue;
					space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
					space_ofs_row[maxk] = i;
					space_ofs_col[maxk++] = j;
				}
			}

			//selective joint bilataral propagation filter
			for (int r = 0; r < mapHeight; r++)
			{
				for (int l = 0; l < mapWidth; l++)
				{
					if (depthMap.GetDepth(r, l) != 0.0f)//如果有深度点了，就跳过
						continue;

					const cv::Vec3b color0 = joint.ptr<cv::Vec3b>(r)[l];
					float sum_w = 0;
					float sum_value_depth = 0;
					float sum_value_normal[3] = { 0.0f };
					for (int k = 0; k < maxk; k++)
					{
						const int rr = r + space_ofs_row[k];
						const int ll = l + space_ofs_col[k];

						if (rr < 0 || rr >= mapHeight || ll < 0
							|| ll >= mapWidth || depthMap.GetDepth(rr, ll) == 0)
							continue;

						//颜色距离权重，是作用在高分辨率图像上的
						cv::Vec3b color1 = joint.ptr<cv::Vec3b>(rr)[ll];

						//// 根据joint当前像素和邻域像素的 距离权重 和 色差权重，计算综合的权重  
						float w = space_weight[k] * color_weight[abs(color0[0] - color1[0]) +
							abs(color0[1] - color1[1]) + abs(color0[2] - color1[2])];

						const float srcDepth = depthMap.GetDepth(rr, ll);
						float srcNormal[3]; normalMap.GetSlice(rr, ll, srcNormal);

						sum_value_depth += srcDepth * w;
						sum_value_normal[0] += srcNormal[0] * w;
						sum_value_normal[1] += srcNormal[1] * w;
						sum_value_normal[2] += srcNormal[2] * w;
						sum_w += w;
					}
					if (sum_w == 0)
						continue;

					sum_w = 1 / sum_w;
					outDepthMap.Set(r, l, sum_value_depth*sum_w);
					sum_value_normal[0] *= sum_w;
					sum_value_normal[1] *= sum_w;
					sum_value_normal[2] *= sum_w;
					SuitNormal(r, l, refK, sum_value_normal);
					outNormalMap.SetSlice(r, l, sum_value_normal);
				}//end of l
			}//end of r

		}

		//只利用距离权重进行滤波插值
		void Workspace::distanceWeightFilter(const DepthMap &depthMap, const NormalMap &normalMap,
			const float *refK, const double sigma_color, const double sigma_space, int radius,
			DepthMap &outDepthMap, NormalMap &outNormalMap) const
		{
			const int mapWidth = depthMap.GetWidth();
			const int mapHeight = depthMap.GetHeight();

			if (radius <= 0)
				radius = round(sigma_space * 1.5);  // 根据 sigma_space 计算 radius  

			//assert(radius % 2 == 1);//确保窗口半径是奇数
			const int d = 2 * radius + 1;

			//原联合图像的通道数
			vector<float> space_weight(d*d);
			vector<int> space_ofs_row(d*d), space_ofs_col(d*d);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

			int maxk = 0;   // 0 - (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			//空间差的高斯权重
			for (int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					double r = std::sqrt((double)i * i + (double)j * j);
					if (r > radius)
						continue;
					space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
					space_ofs_row[maxk] = i;
					space_ofs_col[maxk++] = j;
				}
			}

			//use sapce distance wight only
			for (int r = 0; r < mapHeight; r++)
			{
				for (int l = 0; l < mapWidth; l++)
				{
					if (depthMap.GetDepth(r, l) != 0.0f)  // 如果有深度点了，就跳过
						continue;

					float sum_w = 0;
					float sum_value_depth = 0;
					float sum_value_normal[3] = { 0.0f };
					for (int k = 0; k < maxk; k++)
					{
						const int rr = r + space_ofs_row[k];
						const int ll = l + space_ofs_col[k];

						if (rr < 0 || rr >= mapHeight || ll < 0 || ll >= mapWidth || depthMap.GetDepth(rr, ll) == 0)
							continue;

						// 距离权重   
						float w = space_weight[k];

						const float srcDepth = depthMap.GetDepth(rr, ll);
						float srcNormal[3]; normalMap.GetSlice(rr, ll, srcNormal);

						sum_value_depth += srcDepth * w;
						sum_value_normal[0] += srcNormal[0] * w;
						sum_value_normal[1] += srcNormal[1] * w;
						sum_value_normal[2] += srcNormal[2] * w;
						sum_w += w;
					}
					if (sum_w == 0)
						continue;

					sum_w = 1 / sum_w;
					outDepthMap.Set(r, l, sum_value_depth*sum_w);
					sum_value_normal[0] *= sum_w;
					sum_value_normal[1] *= sum_w;
					sum_value_normal[2] *= sum_w;
					SuitNormal(r, l, refK, sum_value_normal);
					outNormalMap.SetSlice(r, l, sum_value_normal);
				}//end of l
			}//end of r
		}

	}  // namespace mvs	
}  // namespace colmap


					// @even: 计算lapulace边缘, 用于深度图的联合双边滤波
					//cv::Mat blured, guide;
					//cv::blur(src_img, blured, Size(9, 9));
					//cv::Laplacian(blured, guide, -1, 9);
					//selJointBilataralPropagateFilter(guide,
					//	m_depth_maps.at(img_id),
					//	m_normal_maps.at(img_id),
					//	model_.m_images.at(img_id).GetK(),
					//	25, 10,  // 25, 10
					//	-1, 16,
					//	depthMap_pro, normalMap_pro);

					// @even: 对selJB之后的深度图进行联合双边滤波
					//cv::Mat dst, mat = depthMap_pro.Depth2Mat();
					//src_img.convertTo(src_img, CV_32FC1);
					//int k = 10;
					//cv::ximgproc::jointBilateralFilter(src_img, mat, dst,
					//-1, 2 * k - 1, 2 * k - 1);
					//depthMap_pro.fillDepthWithMat(dst);

					// @even: 对selJB之后的深度图进行引导滤波
					//cv::Mat dst, mat = depthMap_pro.Depth2Mat();
					//double eps = 1e-6;
					//eps *= 255.0 * 255.0;
					//dst = guidedFilter(src_img, mat, 10, eps);
					//depthMap_pro.fillDepthWithMat(dst);

						// 联合双边滤波
						//jointBilateralFilter_depth_normal_maps(srcImage, depthMaps_.at(image_id), normalMaps_.at(image_id),
						//	model_.images.at(image_id).GetK(), 25, 10, -1, depthMap_pro, normalMap_pro);

						// 只利用空间距离权重插值
						//distanceWeightFilter(depthMaps_.at(image_id), normalMaps_.at(image_id),
						//	model_.images.at(image_id).GetK(), 25, 10, -1, depthMap_pro, normalMap_pro);