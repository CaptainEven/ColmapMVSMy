#include "workspace.h"
#include "RansacRunner.h"
#include "MyPatchMatch.h"
#include "JointBilateralFilter.h"
#include "upsampler.h"//˫�������ϲ���
#include "BilateralGrid.h"//˫������
#include "FastBilateralSolverMe.h"
#include "guidedfilter.h"
#include "consistency_graph.h"

//#include "./peac/AHCPlaneFitter.hpp"

#include <io.h>
#include <ctime>
#include <Eigen/Dense>
#include <opencv2/ximgproc.hpp>


//#include "BilateralTextureFilter.h";//ϸ�ںͽṹ��ǿ
//#include <Eigen/core>

#include "Utils.h"

// �Ƿ���MRR GPU�Ż�
#define MRF_GPU

#ifdef MRF_GPU
#define NUM_MRF_ITER 1
#else
#define NUM_ITER 1
#endif // MRF_GPU

// �Ƿ����mask
//#define DRAW_MASK

// �Ƿ����log
//#define LOGGING

extern int MRFGPU(const cv::Mat& labels,  // ÿ��pt2d���label
	const std::vector<int>& SPLabels,  // ��ѡ�ĵ�labels
	const std::vector<float>& SPLabelDepths,  // ����label���е�����ֵ
	const std::vector<int>& pts2d_size, // ����label���е��pt2d�����
	const std::vector<cv::Point2f>& NoDepthPts2d,  // �����ֵpt2d��
	const std::vector<float>& sp_label_plane_arrs,  // ����label���е��ƽ�淽��
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet);

extern int MRFGPU2(const cv::Mat& labels,  // ÿ��pt2d���label
	const std::vector<int>& SPLabels,  // ��ѡ�ĵ�labels
	const std::vector<float>& SPLabelDepths,  // ����label���е�����ֵ
	const std::vector<int>& pts2d_size, // ����label���е��pt2d�����
	const std::vector<cv::Point2f>& NoDepthPts2d,  // �����ֵpt2d��
	const std::vector<int>& NoDepthPts2dLabelIdx,  // ÿ��pt2d���label idx
	const std::vector<float>& sp_label_plane_arrs,  // ����label���е��ƽ�淽��
	const std::vector<int>& sp_label_neighs_idx,  // ÿ��label_idx��Ӧ��sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet);

extern int MRFGPU3(
	const cv::Mat& depth_mat,
	const cv::Mat& labels,  // ÿ��pt2d���label
	const std::vector<int>& SPLabels,  // ��ѡ�ĵ�labels
	const std::vector<cv::Point2f>& NoDepthPts2d,  // �����ֵpt2d��
	const std::vector<int>& NoDepthPts2dLabelIdx,  // ÿ��pt2d���label idx
	const std::vector<float>& sp_label_plane_arrs,  // ����label���е��ƽ�淽��
	const std::vector<int>& sp_label_neighs_idx,  // ÿ��label_idx��Ӧ��sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet);

extern int MRFGPU4(const cv::Mat& depth_mat,
	const cv::Mat& labels,  // ÿ��pt2d���label
	const std::vector<int>& SPLabels,  // ��ѡ�ĵ�labels
	const std::vector<float>& SPLabelDepths,  // ����label���е�����ֵ
	const std::vector<int>& pts2d_size, // ����label���е��pt2d�����
	const std::vector<cv::Point2f>& NoDepthPts2d,  // �����ֵpt2d��
	const std::vector<int>& NoDepthPts2dLabelIdx,  // ÿ��pt2d���label idx
	const std::vector<float>& sp_label_plane_arrs,  // ����label���е��ƽ�淽��
	const std::vector<int>& sp_label_neighs_idx,  // ÿ��label_idx��Ӧ��sp_label idx
	const std::vector<int>& sp_label_neigh_num,  // ÿ��label_idx��Ӧ��neighbor����
	const int Radius, const int WIDTH, const int HEIGHT, const float Beta,
	std::vector<int>& NoDepthPt2DSPLabelsRet);

extern int BlockMRF(const cv::Mat& depth_mat,
	const int blk_size,  // block size
	const float* K_inv_arr,  // ����ڲξ������
	const std::vector<int>& blks_pt_cnt,  // ��¼����block��pt2d����
	const std::vector<cv::Point2f>& blks_pts2d,  // ��¼����block��pt2d������
	const std::vector<int>& blks_pt_cnt_has,  // ��¼������block�������ֵ�����
	const std::vector<int>& blks_pt_cnt_non,  // ��¼������block�������ֵ�����
	const std::vector<std::vector<float>>& plane_equa_arr,  // ��¼��Ϊlabel��blk_id��Ӧ��ƽ�淽��
	const std::vector<int>& label_blk_ids,  // ��¼���㹻�����ֵ���blk_id: �ɵ���label
	const std::vector<int>& process_blk_ids,  // ��¼��(MRF)�����blk_id
	const std::vector<float>& process_blks_depths_has,  // ��¼������block�������ֵ(��ɵ�����)
	const std::vector<int>& process_blks_pts2d_has_num,  // ��¼������block�������ֵ�����
	const std::vector<int>& process_blks_pts2d_non_num,  // ��¼������block����ȵ����
	const std::vector<cv::Point2f>& process_blks_pt2d_non,  // ��¼������block�������ֵ������
	const std::vector<int>& all_blks_labels,  // ��¼ÿ��block��Ӧ��label(blk_id): ��ʼlabel����
	const int num_x, const int num_y,  // y����block����, x����block����
	const int radius, const float beta, const float depth_range,
	std::vector<int>& labels_ret);

extern int JBUSPGPU(const cv::Mat& src,
	const cv::Mat& depth_mat,
	const std::vector<cv::Point2f>& pts2d_has_no_depth_jbu,  // �������pt2d��
	const std::vector<int>& sp_labels_idx_jbu,  // ÿ���������pt2d���Ӧ��label_idx
	const std::vector<cv::Point2f>& pts2d_has_depth_jbu,  // ��������JBU�����ֵ��pt2d��
	const std::vector<int>& sp_has_depth_pt2ds_num,  // ÿ��label_idx��Ӧ�������ֵpt2d����
	const std::vector<float>& sigmas_s_jbu, // // ÿ��label_idx��Ӧ��sigma_s
	std::vector<float>& depths_ret);


namespace colmap {
	namespace mvs {

		//��� * elem1 Ӧ������ * elem2 ǰ�棬��������ֵ�Ǹ��������κθ��������У���
		//��� * elem1 ��* elem2 �ĸ�����ǰ�涼�У���ô��������0
		//��� * elem1 Ӧ������ * elem2 ���棬��������ֵ�����������κ����������У���
		bool pairIfAscend(pair<float, int> &a, pair<float, int> &b)
		{
			//if (a.second >= b.second)//���aҪ����b���棬����������
			//{
			//	return 1;
			//}
			//else
			//{
			//	return -1;
			//}

			return a.first < b.first;
		}

		//�����ɴ�С
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

		// ���캯��: ��ʼ��
		Workspace::Workspace(const Options& options)
			: options_(options)
		{
			// ��bundler�����ж�ȡϡ�������Ϣ
			// StringToLower(&options_.input_type);

			// ����ԭͼ���·��
			this->m_model.SetSrcImgRelDir(std::string(options.src_img_dir));

			// ��ȡ�����ռ�(SFMϡ���ؽ����)
			m_model.Read(options_.workspace_path,
				options_.workspace_format,
				options_.newPath);

			if (options_.max_image_size != -1)
			{
				for (auto& image : m_model.m_images)
				{
					// ������ͼ��ߴ�
					image.Downsize(options_.max_image_size, options_.max_image_size);
				}
			}
			if (options_.bDown_sampling)  // �Ƿ���н���������
			{
				for (auto& image : m_model.m_images)
				{
					// ͼ���������������ű���
					image.Rescale(options_.fDown_scale);
				}
			}

			// ������ͼ�����ȥ���䴦��
			//model_.RunUndistortion(options_.undistorte_path);

			// ����bundler����ά��ͶӰ����ά��
			m_model.ProjectToImage();

			// ������ȷ�Χ
			depth_ranges_ = m_model.ComputeDepthRanges();

			// ��ʼ״̬����û���������map����
			hasReadMapsPhoto_ = false;
			hasReadMapsGeom_ = false;
			hasBitMaps_.resize(m_model.m_images.size(), false);
			bitMaps_.resize(m_model.m_images.size());

			// ��ʼ�����ͼ������С
			this->m_depth_maps.resize(m_model.m_images.size());
			this->m_normal_maps.resize(m_model.m_images.size());
		}

		void Workspace::runSLIC(const std::string &path)
		{
			std::cout << "\t" << "=> Begin SLIC..." << std::endl;

			slicLabels_.resize(m_model.m_images.size());

			int k = 1500;
			int m = 10;
			float ss = 15;  // �����صĲ���
			for (int i = 0; i < m_model.m_images.size(); i++)
			{
				SLIC *slic = new SLIC(ss, m, m_model.m_images[i].GetPath(), path);
				slic->run(i).copyTo(slicLabels_[i]);//�ѳ����طָ������labelͼ������slicLabels_
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
			//		//��������Ϣ�����Լ��ͶӰ���Ƿ��bundler������һ��
			//		float x = xyz(0) / xyz(2) - image.GetWidth() / 2;
			//		float y = -xyz(1) / xyz(2) + image.GetHeight() / 2;
			//
			//	}
			//}

			///// ����ά��ͶӰ��ͼ������
			for (int img_id = 0; img_id < m_model.m_images.size(); img_id++)
			{
				const auto &image = m_model.m_images.at(img_id);
				cv::Mat img = cv::imread(image.GetPath());

				// ��Сͼ�񱥺Ͷȣ��ǵ���άͶӰ����ͼ�����濴�ĸ����
				cv::Mat whiteImg(img.size(), img.type(), cv::Scalar::all(255));
				cv::addWeighted(img, 0.6, whiteImg, 0.4, 0.0, img);

				// ��ʼ��ͼ������Բ
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

			// ��ͶӰ�㻭��������ͼ������
			for (int img_id = 0; img_id < m_model.m_images.size(); img_id++)
			{
				const auto &image = m_model.m_images.at(img_id);
				const auto &label = slicLabels_.at(img_id);
				cv::Mat img = cv::imread(image.GetPath());

				// ����ͼ�񱥺Ͷȣ�ʹ�ö�ά����ͼ���Ͽ�������
				cv::Mat whiteImg(img.rows, img.cols, CV_8UC3, cv::Scalar::all(255));
				cv::addWeighted(img, 0.8, whiteImg, 0.2, 0.0, img);

				//// ��������
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
						if (np > 1)  // ����ɼ�ϸ�����طָ���
						{
							img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);//����
							//img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);//����
							istaken.at<bool>(i, j) = true;
						}
					}
				}

				//// ��ʼ��ͼ������Բ
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
		//����Ⱥͷ�����ͼ�����ϲ���
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

				if (!options_.image_as_rgb)  // �������Ҫrgbͼ����ôת��Ϊ�Ҷ�ͼ��
				{
					cv::cvtColor(bitmap, bitmap, CV_BGR2GRAY);
				}

				if (options_.bDetailEnhance)  // �Ƿ�ϸ����ǿ
				{
					//DentailEnhance(bitmap, bitmap);
					//detailEnhance(bitmap, bitmap);//opencv
					const string &tempFileName = "/" + m_model.m_images.at(img_id).GetfileName() + ".jpg";
					imwrite(options_.workspace_path + options_.newPath + tempFileName, bitmap);
				}
				else if (options_.bStructureEnhance)//�Ƿ�ṹ��ǿ
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

		// ����photometirc����geometric��Ⱥͷ���mapͼ
		const void Workspace::ReadDepthAndNormalMaps(const bool isGeometric)
		{
			// ���Ҫ��Geom�����Ѿ����룬����Ҫ��photo�����Ѿ����룬�򷵻�
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

			// ��ȡ������Ⱥͷ���map
			for (int img_id = 0; img_id < m_model.m_images.size(); img_id++)
			{
				//DepthMap depth_map(model_.images.at(image_id).GetWidth(), model_.images.at(image_id).GetHeight(),
				//	depth_ranges_.at(image_id).first, depth_ranges_.at(image_id).second);

				// ��ʼ��depth map
				DepthMap depth_map(depth_ranges_.at(img_id).first,
					depth_ranges_.at(img_id).second);

				string& depth_map_path = this->GetDepthMapPath(img_id, isGeometric);

				depth_map.ReadBinary(depth_map_path);

				// ��Ϊͼ��ߴ�������+-1, ��˾ͼ򵥵İ�ͼ��ߴ��޸�һ��
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
			if (!isGeom)  // ������Ǽ���һ���Եģ�
			{
				fileName = image_name + "." + options_.input_type + file_type;
			}
			else  // ����Ǽ���һ���Ե�
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

		//����˫���ϲ���
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

			// ԭ����ͼ���ͨ����
			const int cnj = joint.channels();

			float *color_weight = new float[cnj * 256];
			float *space_weight = new float[d*d];
			int *space_ofs_row = new int[d*d];  // ����Ĳ�ֵ
			int *space_ofs_col = new int[d*d];

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

			// initialize color-related bilateral filter coefficients  
			// ɫ��ĸ�˹Ȩ��  
			for (int i = 0; i < 256 * cnj; i++)
				color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

			int maxk = 0;   // 0 - (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			// �ռ��ĸ�˹Ȩ��
			for (int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					double r = std::sqrt((double)i * i + (double)j * j);
					if (r > radius)
						continue;

					// �ռ�Ȩ����������Сͼ���ϵ�
					space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff / (upscale*upscale));
					space_ofs_row[maxk] = i;
					space_ofs_col[maxk++] = j;
				}
			}

			for (int r = 0; r < highRow; r++)
			{
				for (int l = 0; l < highCol; l++)
				{
					int px = l, py = r;  // ������������
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

						float fqx = (float)qx / upscale;//�ͷֱ���ͼ���Ӧ����
						float fqy = (float)qy / upscale;
						int iqx = roundf(fqx);//��������
						int iqy = roundf(fqy);
						if (iqx >= lowCol || iqy >= lowRow)
							continue;

						// ��ɫ����Ȩ�أ��������ڸ߷ֱ���ͼ���ϵ�
						cv::Vec3b color1 = joint.ptr<cv::Vec3b>(qy)[qx];

						// ����joint��ǰ���غ��������ص� ����Ȩ�� �� ɫ��Ȩ�أ������ۺϵ�Ȩ��  
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
					//		int qx = px + j, qy = py + i;//����������
					//		if (qx < 0 || qx >= highCol || qy < 0 || qy >= highRow)
					//			continue;
					//
					//		float fqx = (float)qx / upscale;//�ͷֱ���ͼ���Ӧ����
					//		float fqy = (float)qy / upscale;
					//		int iqx = roundf(fqx);//��������
					//		int iqy = roundf(fqy);
					//		if (iqx >= lowCol || iqy >= lowRow)
					//			continue;
					//
					//		//�ռ����Ȩ�أ��������ڵͷֱ���ͼ���ϵ�
					//		float spaceDis = (i*i + j*j) / (upscale*upscale);
					//		float space_w = (float)std::exp(spaceDis * gauss_space_coeff);
					//		//��ɫ����Ȩ�أ��������ڸ߷ֱ���ͼ���ϵ�
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

		// ����˫�ߴ����ϲ���
		void Workspace::jointBilateralPropagationUpsampling(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
			const float upscale, const double sigma_color, const double sigma_space, const int radius, cv::Mat &highDepthMat) const
		{

		}

		// ����˫�ߴ����˲�
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

			// ������ǵ�λ��������ô��һ��: ����һ����Сֵ�������
			const float inv_norm = 1.0f / norm;
			if (inv_norm != 1.0f)
			{
				normal[0] *= inv_norm;
				normal[1] *= inv_norm;
				normal[2] *= inv_norm;
			}
		}

		// �Է�����ͼ��������ֵ�˲�
		void Workspace::NormalMapMediaFilter(const cv::Mat& InNormalMapMat,
			cv::Mat& OutNormalMapMat, const int windowRadis) const
		{

		}

		// �Է�����ͼ��������ֵ�˲����޳�Ϊ0������
		void Workspace::NormalMapMediaFilter1(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat, const int windowRadis) const
		{

		}

		// �Է����������ͼ��������ֵ�˲��������޳�Ϊ0������
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

		// CV_32F����FilterSpeckles
		typedef cv::Point_<short> Point2s;
		//typedef cv::Point_<float> Point2s;

		template <typename T>
		void Workspace::FilterSpeckles(cv::Mat& img, T newVal, int maxSpeckleSize, T maxDiff)
		{
			using namespace cv;

			cv::Mat _buf;

			int width = img.cols, height = img.rows, npixels = width * height;

			// each pixel contains: pixel coordinate(Point2S), label(int), �Ƿ���blob(uchar)
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

		// ����superpixel���Ƶ���ƽ��
		int Workspace::CorrectPCPlane(const float* P_arr,
			const float* K_inv_arr,
			const float* R_inv_arr,
			const float* T_arr,
			const int IMG_WIDTH,   // 2Dͼ����
			const int IMG_HEIGHT,  // 2Dͼ��߶�
			const float DIST_THRESH,  // ��ƽ��ռ������ֵ
			const float fold,
			const std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<float>>& plane_map,
			std::unordered_map<int, cv::Point3f>& center_map,  // superpixel, 3D������������
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, std::vector<float>>& eigen_vals_map,
			std::unordered_map<int, std::vector<float>>& eigen_vects_map)
		{
			for (auto it_1 = label_map.begin(); it_1 != label_map.end(); ++it_1)
			{
				// ����ÿһ��superpixel, ��ȡ��neighbors
				std::set<int> Neighbors;

				for (auto it_2 = label_map.begin(); it_2 != label_map.end(); ++it_2)
				{
					if (it_2->first != it_1->first)
					{
						// Ray tracing: ������������ƽ�������ཻ��2��3D���
						std::vector<std::pair<cv::Point3f, cv::Point3f>> In_2_Plane_3DPtPairs;

						// ----- �Լ���plane֮��ľ���(ȡ2����֮��������ֵ)
						// ����superpixel1��2D���귶Χ
						float x_range_1[2], y_range_1[2], x_range_2[2], y_range_2[2];

						const auto& center_1 = center_map[it_1->first];
						const auto& center_2 = center_map[it_2->first];

						const float* ei_vals_1 = eigen_vals_map[it_1->first].data();
						const float* ei_vals_2 = eigen_vals_map[it_2->first].data();

						const float* ei_vects_1 = eigen_vects_map[it_1->first].data();  // ��һ��plane��3����������
						const float* ei_vects_2 = eigen_vects_map[it_2->first].data();  // �ڶ���plane��3����������

						const float* ei_vects_1_tagent_1 = &ei_vects_1[3];  // ��һ��plane��������� 1
						const float* ei_vects_1_tagent_2 = &ei_vects_1[6];  // ��һ��plane��������� 2

						const float* ei_vects_2_tagent_1 = &ei_vects_2[3];  // �ڶ���plane��������� 1
						const float* ei_vects_2_tagent_2 = &ei_vects_2[6];  // �ڶ���plane��������� 2 

						int ret = this->GetSearchRange(center_1,
							P_arr,
							ei_vals_1, ei_vects_1,
							fold,
							x_range_1, y_range_1);
						if (ret < 0)
						{
							continue;
						}

						// ����superpixel2��2D���귶Χ
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

						// ���2�����������ཻ
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

						// ----- ����plane_1, plane_2�غϵ���������(����ȫ��2D�������)
						for (int y = Y_MIN; y <= Y_MAX; ++y)
						{
							for (int x = X_MIN; x <= X_MAX; ++x)
							{
								cv::Point2f pt2D((float)x, (float)y);

								// ������plane�Ľ�������ֵ
								float depth_1 = this->GetDepthCoPlane(K_inv_arr, R_inv_arr, T_arr,
									plane_map[it_1->first].data(),
									pt2D);
								float depth_2 = this->GetDepthCoPlane(K_inv_arr, R_inv_arr, T_arr,
									plane_map[it_2->first].data(),
									pt2D);

								// ��Ч���ֵ, ����
								if (isnan(depth_1) || isnan(depth_2))
								{
									continue;
								}

								// ������plane�Ľ���(3D)
								cv::Point3f pt3D_1 = this->m_model.BackProjTo3D(K_inv_arr,
									R_inv_arr,
									T_arr,
									depth_1, pt2D);
								cv::Point3f pt3D_2 = this->m_model.BackProjTo3D(K_inv_arr,
									R_inv_arr,
									T_arr,
									depth_2, pt2D);

								// �������3D����, ͬʱ�ڸ�����ƽ�淶Χ��
								if (
									this->IsPt3DInPlaneRange(center_1, pt3D_1,
										ei_vals_1[1], ei_vals_1[2],  // plane_1����ƽ������ֵ����
										ei_vects_1_tagent_1, ei_vects_1_tagent_2,  // plane_1����������,��ƽ�����
										fold)  // ������ú������ƽ�淶Χ
									&&
									this->IsPt3DInPlaneRange(center_2, pt3D_2,
										ei_vals_2[1], ei_vals_2[2],  // plane_2����ƽ������ֵ����
										ei_vects_2_tagent_1, ei_vects_2_tagent_2,  // plane_2����������,��ƽ�����
										fold))
								{
									In_2_Plane_3DPtPairs.push_back(std::make_pair(pt3D_1, pt3D_2));
								}

								//In_2_Plane_3DPtPairs.push_back(std::make_pair(pt3D_1, pt3D_2));
							}
						}

						// 3D���Ϊ��, ����it_2->first
						if (In_2_Plane_3DPtPairs.size() < 3)
						{
							continue;
						}

						// ͳ�ƾ������ֵ
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

						//// ͳ�ƾ����ֵ
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

						// �ж�superpixel it_2->first �Ƿ���neighbor
						if (dist_max < DIST_THRESH)
							//if (dist_mean < DIST_THRESH)
						{
							Neighbors.insert(it_2->first);
						}
					}
				}

				// ���Neighbor̫��, ������
				if (Neighbors.size() < 4)  // ������СNeighbor����
				{
					//printf("[Note]: Superpixel %d, neighbor less than 2\n", 
					//	it_1->first);
					continue;
				}

				// ----- ����Neighbors, ����superpixel it_1->first ����ƽ��
				// ��ȡNeighbor superpixel�ĵ�������
				std::vector<cv::Point3f> neighbor_3dpts(Neighbors.size() + 1);
				int k = 0;
				for (int neigh : Neighbors)
				{
					neighbor_3dpts[k++] = center_map[neigh];
				}
				neighbor_3dpts[k] = center_map[it_1->first];

				// ����Neighbor���ĵĵ���, ���������ƽ��
				this->FitPlaneForSuperpixel(neighbor_3dpts,
					plane_map[it_1->first],                // ����superpixel��ƽ�淽��
					eigen_vals_map[it_1->first],           // ����superpixel����ֵ
					eigen_vects_map[it_1->first],          // ����superpixel��������
					center_map[it_1->first]);              // ����superpixel��������

				// ����superpixel������
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
				// ����ÿһ��superpixel, ��ȡ��neighbors
				std::set<int> Neighbors;

				for (auto it_2 = label_map.begin(); it_2 != label_map.end(); ++it_2)
				{
					if (it_2->first != it_1->first)
					{
						// Ray tracing: ������������ƽ�������ཻ��2��3D���
						std::vector<std::pair<cv::Point3f, cv::Point3f>> In_2_Plane_3DPtPairs;

						// ----- �Լ���plane֮��ľ���(ȡ2����֮��������ֵ)
						// ����superpixel1��2D���귶Χ
						float x_range_1[2], y_range_1[2], x_range_2[2], y_range_2[2];

						const auto& center_1 = center_map[it_1->first];
						const auto& center_2 = center_map[it_2->first];

						const float* ei_vals_1 = eigen_vals_map[it_1->first].data();
						const float* ei_vals_2 = eigen_vals_map[it_2->first].data();

						const float* ei_vects_1 = eigen_vects_map[it_1->first].data();  // ��һ��plane3����������
						const float* ei_vects_2 = eigen_vects_map[it_2->first].data();  // �ڶ���plane3����������

						const float* ei_vects_1_tagent_1 = &ei_vects_1[3];  // ��һ��plane������� 1
						const float* ei_vects_1_tagent_2 = &ei_vects_1[6];  // ��һ��plane������� 2

						const float* ei_vects_2_tagent_1 = &ei_vects_2[3];  // �ڶ���plane������� 1
						const float* ei_vects_2_tagent_2 = &ei_vects_2[6];  // �ڶ���plane������� 2 

						int ret = this->GetSearchRangeCam(center_1,
							K_arr,
							ei_vals_1, ei_vects_1,
							fold,
							x_range_1, y_range_1);
						if (ret < 0)
						{
							continue;
						}

						// ����superpixel2��2D���귶Χ
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

						// ���2�����������ཻ
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

						// ----- ����plane_1, plane_2�غϵ���������(����ȫ��2D�������)
						for (int y = Y_MIN; y <= Y_MAX; ++y)
						{
							for (int x = X_MIN; x <= X_MAX; ++x)
							{
								cv::Point2f pt2D((float)x, (float)y);

								// ������plane�Ľ�������ֵ
								float depth_1 = this->GetDepthCoPlaneCam(K_inv_arr,
									plane_map[it_1->first].data(),
									pt2D);
								float depth_2 = this->GetDepthCoPlaneCam(K_inv_arr,
									plane_map[it_2->first].data(),
									pt2D);

								// ��Ч���ֵ, ����
								if (isnan(depth_1) || isnan(depth_2))
								{
									continue;
								}

								// ������plane�Ľ���(3D)
								cv::Point3f pt3D_1 = this->m_model.BackProjTo3DCam(K_inv_arr,
									depth_1, pt2D);
								cv::Point3f pt3D_2 = this->m_model.BackProjTo3DCam(K_inv_arr,
									depth_2, pt2D);

								// �������3D����, ͬʱ�ڸ�����ƽ�淶Χ��
								if (
									this->IsPt3DInPlaneRange(center_1, pt3D_1,
										ei_vals_1[1], ei_vals_1[2],  // plane_1����ƽ������ֵ����
										ei_vects_1_tagent_1, ei_vects_1_tagent_2,  // plane_1����������,��ƽ�����
										fold)  // ������ú������ƽ�淶Χ
									&&
									this->IsPt3DInPlaneRange(center_2, pt3D_2,
										ei_vals_2[1], ei_vals_2[2],  // plane_2����ƽ������ֵ����
										ei_vects_2_tagent_1, ei_vects_2_tagent_2,  // plane_2����������,��ƽ�����
										fold))
								{
									In_2_Plane_3DPtPairs.push_back(std::make_pair(pt3D_1, pt3D_2));
								}

								//In_2_Plane_3DPtPairs.push_back(std::make_pair(pt3D_1, pt3D_2));
							}
						}

						// 3D���Ϊ��, ����it_2->first
						if (In_2_Plane_3DPtPairs.size() < 3)
						{
							continue;
						}

						// ͳ�ƾ������ֵ
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

						//// ͳ�ƾ����ֵ
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

						// �ж�superpixel it_2->first �Ƿ���neighbor
						if (dist_max < DIST_THRESH)
							//if (dist_mean < DIST_THRESH)
						{
							Neighbors.insert(it_2->first);
						}
					}
				}

				// ���Neighbor̫��, ������
				if (Neighbors.size() < TH_Num_Neigh)  // ������СNeighbor����
				{
					//printf("[Note]: Superpixel %d, neighbor less than 5\n", 
					//	it_1->first);
					continue;
				}

				// ----- ����Neighbors, ����superpixel it_1->first ����ƽ��
				// ��ȡNeighbor superpixel�ĵ�������
				std::vector<cv::Point3f> neighbor_3dpts(Neighbors.size() + 1);
				int k = 0;
				for (int neigh : Neighbors)
				{
					neighbor_3dpts[k++] = center_map[neigh];
				}
				neighbor_3dpts[k] = center_map[it_1->first];

				// ����Neighbor���ĵĵ���, ���������ƽ��
				this->FitPlaneForSuperpixel(neighbor_3dpts,
					plane_map[it_1->first],                // ����superpixel��ƽ�淽��
					eigen_vals_map[it_1->first],           // ����superpixel����ֵ
					eigen_vects_map[it_1->first],          // ����superpixel��������
					center_map[it_1->first]);              // ����superpixel��������

				// ����superpixel������
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

		// ���ڵ�����ƽ��, �Ե��ƽ�������ƽ��
		int Workspace::SmoothPointCloud(const float* R_arr,
			const float* T_arr,
			const float* K_inv_arr,
			const float* R_inv_arr,
			DepthMap& depth_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<float>>& plane_normal_map,
			std::unordered_map<int, std::vector<float>>& ei_vects_map)
		{
			// ����ÿ��superpixel
			for (auto it = label_map.begin(); it != label_map.end(); ++it)
			{
				// ȡsuperpixel��Ӧ��2D�㼯
				std::vector<cv::Point2f>& pts2D = it->second;

				// ��ͶӰ��3D�ռ�
				std::vector<cv::Point3f> pts3D(pts2D.size());

				// ��Ź⻬���3D��
				std::vector<cv::Point3f> pts3D_new(pts2D.size());

				// ����3D��������(��������ϵ)
				int k = 0;  // ����
				for (auto pt2d : pts2D)
				{
					const float& depth = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
					cv::Point3f pt3d = this->m_model.BackProjTo3D(K_inv_arr, R_inv_arr, T_arr,
						depth, pt2d);
					pts3D[k++] = pt3d;
				}

				// ����sigma
				float sigma = this->GetSigmaOfPts3D(pts3D);

				// 3D��⻬
				const auto& normal = plane_normal_map[it->first];  // 3��Ԫ��
				const float* tangent_1 = &ei_vects_map[it->first][3];  // 3��Ԫ��
				const float* tangent_2 = &ei_vects_map[it->first][6];  // 3��Ԫ��
				for (int i = 0; i < (int)pts3D.size(); ++i)
				{
					const cv::Point3f& pt3d_1 = pts3D[i];

					// �⻬������ÿһ����
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

					// ����⻬���3D����
					if (sum_weight > 1e-8)
					{
						cv::Point3f pt3d_1_new = sum / sum_weight;
						pts3D_new[i] = pt3d_1_new;

						// �������ֵ
						float depth_new = this->GetDepthBy3dPtRT(R_arr, T_arr, pt3d_1_new);
						depth_map.Set((int)pts2D[i].y, (int)pts2D[i].x, depth_new);
					}
				}

				//// ����smoothǰ��3D���Ƶķ���仯
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

		// �������ϵ
		int Workspace::SmoothPointCloudCam(const float* K_arr,
			const float* K_inv_arr,
			const std::vector<cv::Point2f>& pts2D,
			const float* normal,
			const float* tangent_1,
			const float* tangent_2,
			DepthMap& depth_map)
		{
			// ��ͶӰ��3D�ռ�
			std::vector<cv::Point3f> pts3D(pts2D.size());

			// ��Ź⻬���3D��
			std::vector<cv::Point3f> pts3D_new(pts2D.size());

			// ����3D����(�������ϵ)
			int k = 0;  // ����
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

			// ȷ��sigma
			float sigma = this->GetSigmaOfPts3D(pts3D);

			// 3D��⻬
			for (int i = 0; i < (int)pts3D.size(); ++i)
			{
				const cv::Point3f& pt3d_1 = pts3D[i];

				// �⻬������ÿһ����
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

				// ����⻬���3D����
				if (sum_weight > 1e-8)
				{
					cv::Point3f pt3d_1_new = sum / sum_weight;
					pts3D_new[i] = pt3d_1_new;

					// �������ֵ
					float depth_new = this->GetDepthBy3dPtK(K_arr, pt3d_1_new);
					//if (depth_new < 0.0f)
					//{
					//	printf("pause\n");
					//}
					depth_map.Set((int)pts2D[i].y, (int)pts2D[i].x, depth_new);
				}
			}

			//// ����smoothǰ��3D���Ƶķ���仯
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

		// ����ƽ�����ĵ�(center), 2����������ȷ��ƽ����ĸ�����, ����ȷ��2D������Χ
		int Workspace::GetSearchRange(const cv::Point3f& center,
			const float* P,  // �����������
			const float* ei_vals,  // ����ֵ
			const float* ei_vects,  // ��������
			const float fold,  // ������չ����(Ĭ��3.0f)
			float* x_range, float* y_range)
		{
			// �����ĸ������3D����(˳ʱ��)
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

			// �������
			const Eigen::Vector4f vert_1_H(vertex_1.x, vertex_1.y, vertex_1.z, 1.0f);
			const Eigen::Vector4f vert_2_H(vertex_2.x, vertex_2.y, vertex_2.z, 1.0f);
			const Eigen::Vector4f vert_3_H(vertex_3.x, vertex_3.y, vertex_3.z, 1.0f);
			const Eigen::Vector4f vert_4_H(vertex_4.x, vertex_4.y, vertex_4.z, 1.0f);

			// ���ĸ�����ͶӰ��2D����ϵ
			const Eigen::Vector3f xyd_1 = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(P) * vert_1_H;
			const Eigen::Vector3f xyd_2 = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(P) * vert_2_H;
			const Eigen::Vector3f xyd_3 = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(P) * vert_3_H;
			const Eigen::Vector3f xyd_4 = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(P) * vert_4_H;

			// 2D�����׼��
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
					// ��Ч2d����
					memset(x_arr.data(), 0, sizeof(float) * 4);
					memset(y_arr.data(), 0, sizeof(float) * 4);

					return -1;
				}
			}

			// ����x, y���귶Χ
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

		// �������ϵ
		int Workspace::GetSearchRangeCam(const cv::Point3f& center,  // �������ϵ
			const float* K_arr,
			const float* ei_vals,
			const float* ei_vects,
			const float fold,
			float* x_range, float* y_range)
		{
			// �����ĸ������3D����(�������ϵ)(˳ʱ��)
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

			// ���ĸ�����ͶӰ��2D����ϵ
			const Eigen::Vector3f xyd_1 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * vert_1;
			const Eigen::Vector3f xyd_2 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * vert_2;
			const Eigen::Vector3f xyd_3 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * vert_3;
			const Eigen::Vector3f xyd_4 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_arr) * vert_4;

			// 2D�����׼��
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
					// ��Ч2d����
					memset(x_arr.data(), 0, sizeof(float) * 4);
					memset(y_arr.data(), 0, sizeof(float) * 4);

					return -1;
				}
			}

			// ����x, y���귶Χ
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

			// �ܵ��ӽǸ���
			int NumView = (int)this->m_model.m_images.size();

			//// ��ȡ����SFM����Ա�
			//auto pair_map = this->GetImgPairMap();

			for (int img_id = 0; img_id < NumView; ++img_id)  // ע��: img_id != IMAGE_ID
			{
				//// ������Ա����ƽ�����߾���
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
				src = cv::imread(src_dir + file_name, cv::IMREAD_COLOR);  // ԭͼ��ȡBGR��ɫͼ
				if (src.empty())
				{
					printf("[Err]: empty src image\n");
					return;
				}

				// ----- super-pixel segmentation using SEEDS or SLIC
				// SEEDS super-pixel segmentation
				const int num_superpixels = 700;  // ����ĳ�ʼ�ָ֤�߽�
				Ptr<cv::ximgproc::SuperpixelSEEDS> superpixel = cv::ximgproc::createSuperpixelSEEDS(src.cols,
					src.rows,
					src.channels(),
					num_superpixels,  // num_superpixels
					15,  // num_levels: 5, 15
					2,
					5,
					true);
				superpixel->iterate(src);  // ����������Ĭ��Ϊ4
				superpixel->getLabels(labels);  // ��ȡlabels
				superpixel->getLabelContourMask(mask);  // ��ȡ�����صı߽�

				//// ����SLIC super-pixel segmentation
				//Ptr<cv::ximgproc::SuperpixelSLIC> superpixel = cv::ximgproc::createSuperpixelSLIC(src,
				//	101,
				//	50);
				//superpixel->iterate();  // ����������Ĭ��Ϊ10
				//superpixel->enforceLabelConnectivity();
				//superpixel->getLabelContourMask(mask);  // ��ȡ�����صı߽�
				//superpixel->getLabels(labels);  // ��ȡlabels

				// ��ȡ�����ص�����
				//int actual_number = superpixel->getNumberOfSuperpixels();  

				// construct 2 Hashmaps for each super-pixel
				std::unordered_map<int, std::vector<cv::Point2f>> label_map, has_depth_map, has_no_depth_map;

				// traverse each pxiel to put into hashmaps
				for (int y = 0; y < labels.rows; ++y)
				{
					for (int x = 0; x < labels.cols; ++x)
					{
						const int& label = labels.at<int>(y, x);

						// label -> ͼ��2D����㼯
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

				// ----- Ϊԭͼ����superpixel merge֮ǰ��mask
				cv::Mat src_1 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_1);
				string mask_src1_name = file_name + "_before_merge_mask.jpg";
				depth_map_path = depth_dir + mask_src1_name;
				cv::imwrite(depth_map_path, src_1);
				printf("%s written\n", mask_src1_name.c_str());

				// ----- superpixel�ϲ�(�ϲ���Чdepth������ֵ��)
				//printf("Before merging, %d superpixels\n", has_depth_map.size());
				this->MergeSuperpixels(src, 500, labels, label_map, has_depth_map, has_no_depth_map);
				printf("After merging, %d superpixels\n", has_depth_map.size());

				// ----- Ϊԭͼ����superpixel merge֮���mask
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

				//// Ϊ��ǰͼ��(�ӽ�)��ȡconsistency map...
				//string consistency_path = consistency_dir + consistency_f_name;
				//this->ReadConsistencyGraph(consistency_path);

				// ----- ����Merge���superpixels, ƽ�����
				std::unordered_map<int, std::vector<float>> eigen_vals_map;  // superpixel������ֵ
				std::unordered_map<int, std::vector<float>> eigen_vects_map;  // superpixel��������
				std::unordered_map<int, std::vector<float>> plane_normal_map;  // superpxiel�ķ�����
				std::unordered_map<int, std::vector<float>> plane_map;  // superpixel����ƽ�淽��
				std::unordered_map<int, cv::Point3f> center_map;  // superpixel�����ĵ�����
				this->FitPlaneForSuperpixels(depth_map,
					K_inv_arr, R_inv_arr, T_arr,
					label_map, has_depth_map,
					center_map, eigen_vals_map, eigen_vects_map,
					plane_normal_map, plane_map);
				printf("Superpixel plane fitting done\n");

				// ----- ����tagent plane
				this->CorrectPCPlane(P_arr, K_inv_arr, R_inv_arr, T_arr,
					src.cols, src.rows,
					depth_range * 0.1f,
					3.0f,
					label_map,
					plane_map, center_map, plane_normal_map,
					eigen_vals_map, eigen_vects_map);
				printf("Superpixel plane correction done\n");

				// ----- superpixel����: ���ӿ����ӵ�����superpixel,
				//printf("Before connecting,  %d superpixels\n", has_depth_map.size());
				this->ConnectSuperpixels(0.0005f, 0.28f, depth_map,
					K_inv_arr, R_inv_arr, T_arr,
					eigen_vals_map, plane_normal_map, center_map,
					labels, label_map, has_depth_map, has_no_depth_map);
				printf("After connecting, %d superpixels\n", has_depth_map.size());

				// ----- ����Connect֮���mask
				cv::Mat src_3 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_3);
				string mask_src3_name = file_name + "_after_connect_mask.jpg";
				depth_map_path = depth_dir + mask_src3_name;
				cv::imwrite(depth_map_path, src_3);
				printf("%s written\n", mask_src3_name.c_str());

				// traver each super-pixel
				// ��¼plane��non-plane���, ����debug
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
						ransac.RunRansac(Pts3D);  // ����ETH3D����~0.0204m(20.4mm)
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

					// ----- ����: ƽ�����ƽ��
					// ƽ��ʹ��ƽ��Լ��, ��ƽ�泬������JBU+����⻬
					if (this->IsPlaneSuperpixelCloud(Pts3D,
						plane_arr,
						eigen_vals_map[it->first],
						0.0002f,
						depth_range * 0.01f))  // modify Dist_TH?
					{
						//printf("Superpixel %d is a plane\n", it->first);
						plane_ids.push_back(it->first);

						// ƽ��Լ�������ֵ
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

							// ����depth
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
							// JBU��ֵ�������ֵ, ��ͬһ��superpixel�ڲ�ʹ��
							float depth = this->JBUSP(pt2D,
								has_depth_map[it->first], src, depth_map, sigma_s);

							// ����depth
							depth_map.Set(pt2D.y, pt2D.x, depth);
						}
					}

					printf("Superpixel %d processed\n", it->first);
				}
				printf("Total %d planes, %d non-planes\n",
					(int)plane_ids.size(), (int)non_plane_ids.size());

				// ���plane_ids��non_plane_ids
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

				// ----- ���ƹ⻬
				//this->SmoothPointCloud(R_arr, T_arr, K_inv_arr, R_inv_arr, 0.1f,
				//	depth_map,
				//	label_map,
				//	plane_normal_map,
				//	eigen_vects_map);

				// ----- ���Filled���������ͼ: �����ؽ����������ͼ����
				string filled_out_name = file_name + ".geometric.bin";
				depth_map_path = depth_dir + filled_out_name;
				depth_map.WriteBinary(depth_map_path);
				printf("%s written\n", filled_out_name.c_str());

				// ���Filled���ͼ���ڿ��ӻ�[0, 255]
				string filled_name = file_name + "_filled.jpg";
				depth_map_path = depth_dir + filled_name;
				cv::imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				printf("%s written\n", filled_name.c_str());

				// ----- ΪFilled���ͼ����mask, ������ͼ+super-pixel mask ���ڿ��ӻ�[0, 255]
				cv::Mat depth_filled = cv::imread(depth_map_path, cv::IMREAD_COLOR);
				this->DrawMaskOfSuperpixels(labels, depth_filled);

				// ����superpixel���
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

			// �ܵ��ӽǸ���
			int NumView = (int)this->m_model.m_images.size();

			for (int img_id = 0; img_id < NumView; ++img_id)  // ע��: img_id != IMAGE_ID
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
				src = cv::imread(src_dir + file_name, cv::IMREAD_COLOR);  // ԭͼ��ȡBGR��ɫͼ
				if (src.empty())
				{
					printf("[Err]: empty src image\n");
					return;
				}

				// ----- super-pixel segmentation using SEEDS or SLIC
				// SEEDS super-pixel segmentation
				const int num_superpixels = 700;  // ����ĳ�ʼ�ָ֤�߽�
				Ptr<cv::ximgproc::SuperpixelSEEDS> superpixel = cv::ximgproc::createSuperpixelSEEDS(src.cols,
					src.rows,
					src.channels(),
					num_superpixels,  // num_superpixels
					15,  // num_levels: 5, 15
					2,
					5,
					true);
				superpixel->iterate(src);  // ����������Ĭ��Ϊ4
				superpixel->getLabels(labels);  // ��ȡlabels
				superpixel->getLabelContourMask(mask);  // ��ȡ�����صı߽�

				// construct 2 Hashmaps for each super-pixel
				std::unordered_map<int, std::vector<cv::Point2f>> label_map, has_depth_map, has_no_depth_map;

				// traverse each pxiel to put into hashmaps
				for (int y = 0; y < labels.rows; ++y)
				{
					for (int x = 0; x < labels.cols; ++x)
					{
						const int& label = labels.at<int>(y, x);

						// label -> ͼ��2D����㼯
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

				// ----- Ϊԭͼ����superpixel merge֮ǰ��mask
				cv::Mat src_1 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_1);
				string mask_src1_name = file_name + "_before_merge_mask.jpg";
				depth_map_path = depth_dir + mask_src1_name;
				cv::imwrite(depth_map_path, src_1);
				printf("%s written\n", mask_src1_name.c_str());

				// ----- superpixel�ϲ�(�ϲ���Чdepth������ֵ��)
				printf("Before merging, %d superpixels\n", has_depth_map.size());
				this->MergeSuperpixels(src, 500, labels, label_map, has_depth_map, has_no_depth_map);
				printf("After merging, %d superpixels\n", has_depth_map.size());

				// ----- Ϊԭͼ����superpixel merge֮���mask
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

				//// Ϊ��ǰͼ��(�ӽ�)��ȡconsistency map...
				//string consistency_path = consistency_dir + consistency_f_name;
				//this->ReadConsistencyGraph(consistency_path);

				// ----- ����Merge���superpixels, ƽ�����
				std::unordered_map<int, std::vector<float>> eigen_vals_map;  // superpixel������ֵ
				std::unordered_map<int, std::vector<float>> eigen_vects_map;  // superpixel��������
				std::unordered_map<int, std::vector<float>> plane_normal_map;  // superpxiel�ķ�����
				std::unordered_map<int, std::vector<float>> plane_map;  // superpixel����ƽ�淽��
				std::unordered_map<int, cv::Point3f> center_map;  // superpixel�����ĵ�����
				this->FitPlaneForSPsCam(depth_map,
					K_inv_arr,
					label_map, has_depth_map,
					center_map, eigen_vals_map, eigen_vects_map,
					plane_normal_map, plane_map);
				printf("Superpixel plane fitting done\n");

				// ----- ����tagent plane
				this->CorrectPlaneCam(K_arr, K_inv_arr,
					src.cols, src.rows,
					depth_range * 0.1f,
					3.0f, 5,
					label_map,
					plane_map, center_map, plane_normal_map,
					eigen_vals_map, eigen_vects_map);
				printf("Superpixel plane correction done\n");

				// ----- superpixel����: ���ӿ����ӵ�����superpixel,
				//printf("Before connecting,  %d superpixels\n", has_depth_map.size());
				this->ConnectSuperpixelsCam(0.0005f, 0.28f, depth_map,
					K_inv_arr,
					plane_map,
					eigen_vals_map, plane_normal_map, center_map,
					labels, label_map, has_depth_map, has_no_depth_map);
				printf("After connecting, %d superpixels\n", has_depth_map.size());

				// ----- ����Connect֮���mask
				cv::Mat src_3 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_3);
				string mask_src3_name = file_name + "_after_connect_mask.jpg";
				depth_map_path = depth_dir + mask_src3_name;
				cv::imwrite(depth_map_path, src_3);
				printf("%s written\n", mask_src3_name.c_str());

				// traver each super-pixel
				// ��¼plane��non-plane���, ����debug
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
						ransac.RunRansac(Pts3D);  // ����ETH3D����~0.0204m(20.4mm)
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

					// ----- ����: ƽ�����ƽ��
					// ƽ��ʹ��ƽ��Լ��, ��ƽ�泬������JBU+����⻬
					if (this->IsPlaneSuperpixelCloud(Pts3D,
						plane_arr,
						eigen_vals_map[it->first],
						0.0002f,
						depth_range * 0.01f))  // modify Dist_TH?
					{
						//printf("Superpixel %d is a plane\n", it->first);
						plane_ids.push_back(it->first);

						// ƽ��Լ�������ֵ
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

							// ����depth
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
							// JBU��ֵ�������ֵ, ��ͬһ��superpixel�ڲ�ʹ��
							float depth = this->JBUSP(pt2D,
								has_depth_map[it->first], src, depth_map, sigma_s);

							// ����depth
							depth_map.Set(pt2D.y, pt2D.x, depth);
						}

						// ���ƹ⻬
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

				// ���plane_ids��non_plane_ids
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

				// ----- ���Filled���������ͼ: �����ؽ����������ͼ����
				string filled_out_name = file_name + ".geometric.bin";
				depth_map_path = depth_dir + filled_out_name;
				depth_map.WriteBinary(depth_map_path);
				printf("%s written\n", filled_out_name.c_str());

				// ���Filled���ͼ���ڿ��ӻ�[0, 255]
				string filled_name = file_name + "_filled.jpg";
				depth_map_path = depth_dir + filled_name;
				cv::imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				printf("%s written\n", filled_name.c_str());

				// ----- ΪFilled���ͼ����mask, ������ͼ+super-pixel mask ���ڿ��ӻ�[0, 255]
				cv::Mat depth_filled = cv::imread(depth_map_path, cv::IMREAD_COLOR);
				this->DrawMaskOfSuperpixels(labels, depth_filled);

				// ����superpixel���
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

		// ����MRF����
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
			// ----- �����Ԫ������
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

			// ----- ����һԪ������
			float depth_mean = 0.0f, depth_std = 0.0f;
			const int pts2d_size = int(has_depth_map[SP_Label].size()
				+ has_no_depth_map[SP_Label].size());
			std::vector<float> depths_(pts2d_size);
			// �����ֵ��2d��
			const std::vector<cv::Point2f>& has_depth_pts = has_depth_map[SP_Label];
			for (int i = 0; i < (int)has_depth_pts.size(); ++i)
			{
				const cv::Point2f& pt2d = has_depth_pts[i];
				depths_[i] = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
			}
			// û�����ֵ��2d��: ƽ��Լ����ֵ
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

			// ͳ�Ʊ�׼��
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
				// ����һ��2D��
				const cv::Point2f& Pt2D = Pts2D[pt_i];

				// �����Ԫ��������Χ: ���ĵ�ȷ����Ҳ��ȷ����������Χ
				int y_begin = Pt2D.y - radius;
				y_begin = y_begin >= 0 ? y_begin : 0;
				int y_end = Pt2D.y + radius;
				y_end = y_end <= depth_map.GetHeight() - 1 ? y_end : depth_map.GetHeight() - 1;

				int x_begin = Pt2D.x - radius;
				x_begin = x_begin >= 0 ? x_begin : 0;
				int x_end = Pt2D.x + radius;
				x_end = x_end <= depth_map.GetWidth() - 1 ? x_end : depth_map.GetWidth() - 1;

				// ��������label(plane), �ҳ�������С��plane
				float energy_min = FLT_MAX;
				int best_pl_id = -1;
				for (int pl_i = 0; pl_i < PlaneCount; ++pl_i)
				{
					const int& SP_Label = PlaneID2SPLabel.at(pl_i);

					// ----- �����Ԫ������
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

					// ----- ����һԪ������
					float depth_mean = 0.0f, depth_std = 0.0f;
					const size_t pts2d_size = Label2Pts2d.at(SP_Label).size();
					std::vector<float> depths_(pts2d_size);
					const std::vector<cv::Point2f>& pts2d = Label2Pts2d.at(SP_Label);

					// ��ȡ��label��Ӧ�����ֵ
					for (int i = 0; i < (int)pts2d.size(); ++i)
					{
						const cv::Point2f& pt2d = pts2d[i];
						depths_[i] = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
					}
					depth_mean = std::accumulate(depths_.begin(), depths_.end(), 0.0f)
						/ float(pts2d_size);

					// ͳ�Ʊ�׼��
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

				// ����label: ���ܸ��¹�, Ҳ����û����
				const int& label = PlaneID2SPLabel.at(best_pl_id);
				Pt2DSPLabelsRet[pt_i] = label;

				// ������µ�label: ÿ1000��
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

		// �������ϵ
		void Workspace::TestDepth6()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";

			// �ܵ��ӽǸ���
			const int NumView = (int)this->m_model.m_images.size();
			for (int img_id = 0; img_id < NumView; ++img_id)  // ע��: img_id != IMAGE_ID
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

				float denominator = 50.0f, ratio = 0.08;  // ����
#ifdef MRF_GPU
				for (int iter_i = 0; iter_i < NUM_MRF_ITER; ++iter_i)
#else
				for (int iter_i = 0; iter_i < NUM_ITER; ++iter_i)
#endif
				{
					// ----- ���˰ߵ�
					const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
						/ denominator);  // 80
					const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
					const float maxDiff = ratio * depth_range;  // 0.038

					cv::Mat depth_mat = depth_map.Depth2Mat();

					// speckle filtering for depth_mat
					this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

					// fill depth_map
					depth_map.fillDepthWithMat(depth_mat);  // �˴���depth_map������д�����!!!

					// write to disk for visualization
					string filter_name = file_name + "_filtered.jpg";
					string filter_path = depth_dir + filter_name;
					cv::imwrite(filter_path, depth_map.ToBitmapGray(2, 98));
					std::printf("%s written\n", filter_name.c_str());

					// ----------- super-pixel segmentation
					cv::Mat src, mask, labels;

					// ԭͼ��ȡBGR��ɫͼ
					src = cv::imread(src_dir + file_name, cv::IMREAD_COLOR);
					if (src.empty())
					{
						std::printf("[Err]: empty src image\n");
						return;
					}

					// ----- super-pixel segmentation using SEEDS or SLIC
					// SEEDS super-pixel segmentation
					const int num_superpixels = 1000;  // ����ĳ�ʼ�ָ֤�߽�
					Ptr<cv::ximgproc::SuperpixelSEEDS> superpixel = cv::ximgproc::createSuperpixelSEEDS(src.cols,
						src.rows,
						src.channels(),
						num_superpixels,  // num_superpixels
						15,  // num_levels: 5, 15
						2,
						5,
						true);
					superpixel->iterate(src);  // ����������Ĭ��Ϊ4
					superpixel->getLabels(labels);  // ��ȡlabels
					superpixel->getLabelContourMask(mask);  // ��ȡ�����صı߽�

					//Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(src,
					//	101, 25);
					//slic->iterate();  // ����������Ĭ��Ϊ10
					//slic->enforceLabelConnectivity();
					//slic->getLabelContourMask(mask);  // ��ȡ�����صı߽�
					//slic->getLabels(labels);  // ��ȡlabels

					// construct 2 Hashmaps for each super-pixel
					std::unordered_map<int, std::vector<cv::Point2f>> Label2Pts2d,
						has_depth_map, has_no_depth_map;

					// traverse each pxiel to put into hashmaps
					for (int y = 0; y < labels.rows; ++y)
					{
						for (int x = 0; x < labels.cols; ++x)
						{
							const int& label = labels.at<int>(y, x);

							// label -> ͼ��2D����㼯
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
					// ----- Ϊԭͼ����superpixel merge֮ǰ��mask
					cv::Mat src_1 = src.clone();
					this->DrawMaskOfSuperpixels(labels, src_1);
					string mask_src1_name = file_name + "_before_merge_mask.jpg";
					viz_out_path = depth_dir + mask_src1_name;
					cv::imwrite(viz_out_path, src_1);
					std::printf("%s written\n", mask_src1_name.c_str());
#endif // DRAW_MASK

					// ----- superpixel�ϲ�(�ϲ���Чdepth������ֵ��)
					std::printf("Before merging, %d superpixels\n", has_depth_map.size());
					this->MergeSuperpixels(src,
						4000,  // 2000
						labels,
						Label2Pts2d, has_depth_map, has_no_depth_map);
					std::printf("After merging, %d superpixels\n", has_depth_map.size());

#ifdef DRAW_MASK
					// ----- Ϊԭͼ����superpixel merge֮���mask
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

					// ----- ����Merge���superpixels, ƽ�����
					std::unordered_map<int, std::vector<float>> eigen_vals_map;  // superpixel������ֵ
					std::unordered_map<int, std::vector<float>> eigen_vects_map;  // superpixel��������
					std::unordered_map<int, std::vector<float>> plane_normal_map;  // superpxiel�ķ�����
					std::unordered_map<int, std::vector<float>> plane_map;  // superpixel����ƽ�淽��
					std::unordered_map<int, cv::Point3f> center_map;  // superpixel�����ĵ�����
					this->FitPlaneForSPsCam(depth_map,
						K_inv_arr,
						Label2Pts2d, has_depth_map,
						center_map, eigen_vals_map, eigen_vects_map,
						plane_normal_map, plane_map);
					std::printf("Superpixel plane fitting done\n");

					// ----- ����tagent plane
					this->CorrectPlaneCam(K_arr, K_inv_arr,
						src.cols, src.rows,
						depth_range * 0.01f,  // ����: distance threshold
						3.0f, 5,  // ����: fold, minimum neighbor number
						Label2Pts2d,
						plane_map, center_map, plane_normal_map,
						eigen_vals_map, eigen_vects_map);
					std::printf("Superpixel plane correction done\n");

					// ----- superpixel����: ���ӿ����ӵ�����superpixel,
					this->ConnectSuperpixelsCam(0.0005f, 0.15f, depth_map,
						K_inv_arr,
						plane_map,
						eigen_vals_map, plane_normal_map, center_map,
						labels, Label2Pts2d, has_depth_map, has_no_depth_map);
					std::printf("%d superpixels after connection\n", has_depth_map.size());

#ifdef DRAW_MASK
					// ----- Ϊԭͼ����Connect֮���mask
					cv::Mat src_3 = src.clone();
					this->DrawMaskOfSuperpixels(labels, src_3);
					string mask_src3_name = file_name + "_after_connect_mask.jpg";
					viz_out_path = depth_dir + mask_src3_name;
					cv::imwrite(viz_out_path, src_3);
					printf("%s written\n", mask_src3_name.c_str());

					// ----- �����depth_map֮ǰ, Ϊdepth_map����mask(�����㷨����)
					cv::Mat depth_mask = cv::imread(filter_path, cv::IMREAD_COLOR);
					this->DrawMaskOfSuperpixels(labels, depth_mask);
					string filter_mask = file_name + "_depth_filter_mask.jpg";
					viz_out_path = depth_dir + filter_mask;
					cv::imwrite(viz_out_path, depth_mask);
					std::printf("%s written\n", filter_mask.c_str());
#endif // DRAW_MASK


					//// ----- CPU��, ��ʼ��depth_map�����ֵ2D��...
					//for (auto it = has_no_depth_map.begin();
					//	it != has_no_depth_map.end(); ++it)
					//{
					//	// ����space_sigma
					//	float sigma_s = this->GetSigmaOfPts2D(Label2Pts2d[it->first]);
					//	for (auto pt2D : has_no_depth_map[it->first])
					//	{
					//		// ƽ��Լ�������ֵ
					//		float depth = this->GetDepthCoPlaneCam(K_inv_arr, 
					//			plane_map[it->first].data(), pt2D);
					//		if (depth <= 0.0f)
					//		{
					//			// ���ֵ����ȷ, ͨ��JBU��ֵ���¼������ֵ
					//			depth = this->JBUSP(pt2D,
					//				has_depth_map[it->first], src, depth_map, sigma_s);
					//			//printf("Pixel[%d, %d] @Superpixel%d, depth: %.3f filled with JBU\n",
					//			//	(int)pt2D.x, (int)pt2D.y, it->first, depth);
					//		}
					//		// ����depth
					//		depth_map.Set(pt2D.y, pt2D.x, depth);
					//	}
					//}
					//std::printf("Init depth_map done\n");

					// ----- ����������Ĺ��ü�������
					int stride = 0, sp_count = 0, pt_count = 0;

					// ----- GPU�˳�ʼ��depth_map, JBUSPGPU����
					// ---- ����GPU�������
					// 1. ---���¾����������ֵ��pt2d, ��depth < 0�ķ���һ������
					// 2. ---ͳ��depth<0��pt2d�����Ϣ:
					// pts2d_has_no_depth_jbu, sp_labels_idx_jbu
					std::vector<cv::Point2f> pts2d_no_depth_jbu;  // ���д������pt2d��
					std::vector<int> sp_labels_idx_jbu;  // ÿ�����sp label index

					// ---JBUSPGPU��׼����������4������ĳ�ʼ��: 
					// (1). sp_labels_jbu, (2). pts2d_has_depth_jbu
					// (3). sp_has_depth_pt2ds_num (4). sigmas_s_jbu
					// pts2d_has_depth_jbu,
					// ����GPU��JBUSP��sp label����
					std::vector<int> sp_labels_jbu(has_no_depth_map.size(), 0);

					// ---ͳ�������ֵpt2d���ܹ��ĵ���
					pt_count = 0;
					for (auto it = has_no_depth_map.begin();
						it != has_no_depth_map.end(); ++it)
					{
						pt_count += (int)has_depth_map.at(it->first).size();
					}

					// ����label��Ӧ��pt2d������: ����has_no_depth_map��label˳��
					std::vector<cv::Point2f> pts2d_has_depth_jbu(pt_count);

					// ÿ��label��Ӧ��pt2d�����
					std::vector<int> sp_has_depth_pt2ds_num(has_no_depth_map.size());

					// ����sigma_s����: ÿ��label����Ӧһ��sigma_sֵ
					std::vector<float> sigmas_s_jbu(has_no_depth_map.size());

					stride = 0, sp_count = 0;  // sp_count��has_no_depth_map��label_idx
					for (auto it = has_no_depth_map.begin();
						it != has_no_depth_map.end(); ++it)
					{
						// ����sp_labels_jbu
						sp_labels_jbu[sp_count] = it->first;

						// ����pts2d_has_depth_jbu
						memcpy(pts2d_has_depth_jbu.data() + stride,
							has_depth_map[it->first].data(),
							sizeof(cv::Point2f) * has_depth_map[it->first].size());

						// ����sp_has_depth_pt2ds_num
						sp_has_depth_pt2ds_num[sp_count] = int(has_depth_map[it->first].size());

						// ����space_sigma
						sigmas_s_jbu[sp_count] = this->GetSigmaOfPts2D(Label2Pts2d[it->first]);

						// ���pts2d_has_no_depth_jbu�����sp_labels_idx_jbu����
						for (auto pt2D : has_no_depth_map[it->first])
						{
							// ƽ��Լ�������ֵ
							float depth = this->GetDepthCoPlaneCam(K_inv_arr,
								plane_map.at(it->first).data(), pt2D);

							// Ҫ����JBUSP��pt2d��
							if (depth <= 0.0f)
							{
								pts2d_no_depth_jbu.push_back(pt2D);
								sp_labels_idx_jbu.push_back(sp_count);
							}
							else
							{
								// ����ƽ��Լ��, ����depth
								depth_map.Set(pt2D.y, pt2D.x, depth);  // �˴���depth_map������д�����!!!
							}
						}

						sp_count += 1;
						stride += int(has_depth_map[it->first].size());
					}
					std::printf("GPU preparations for JBUSP built done, total %d pts for JBUSP\n",
						(int)pts2d_no_depth_jbu.size());

					assert(pts2d_has_depth_jbu.size()
						== std::accumulate(sp_has_depth_pt2ds_num.begin(), sp_has_depth_pt2ds_num.end(), 0));

					// GPU������JBUSP
					std::vector<float> depths_ret(pts2d_no_depth_jbu.size(), 0.0f);

					if (pts2d_no_depth_jbu.size() > 0)
					{
						std::printf("Start GPU JBUSP...\n");
						JBUSPGPU(src,
							depth_mat,
							pts2d_no_depth_jbu,  // �������pt2d������
							sp_labels_idx_jbu,  // ÿ��������dept2d���Ӧ��label idx
							pts2d_has_depth_jbu,
							sp_has_depth_pt2ds_num,
							sigmas_s_jbu,
							depths_ret);
						std::printf("GPU JBUSP done\n");

						// ���depth_map���
						for (int i = 0; i < (int)pts2d_no_depth_jbu.size(); ++i)
						{// �˴���depth_map������д�����!!!
							depth_map.Set((int)pts2d_no_depth_jbu[i].y,
								(int)pts2d_no_depth_jbu[i].x, depths_ret[i]);
						}
					}
					std::printf("Depthmap completed\n");

#ifdef MRF_GPU
					// -----����GPU������, �������(Host��)����һ������Ҫ�Ż�...
					// ��������sp_label_depths(��label���е����ֵ����): ����Label2Pts2d��������
					// ����sp_label_pts2d_size, ÿ��label��pt2d�����: ����Label2Pts2d��������
					// ����sp_label����: ����Label2Pts2d��������
					// ����NoDepthPts2d��NoDepthPt2DSPLabelsPre����: ����Label2Pts2d��������
					// --- ͳ��û�����ֵ�ĵ���
					pt_count = 0;
					for (auto it = has_no_depth_map.begin(); it != has_no_depth_map.end(); ++it)
					{
						pt_count += (int)it->second.size();
					}
					const int NoDepthPt2dCount = pt_count;  // ��¼û�����ֵpt2d������
					std::vector<int> sp_labels(Label2Pts2d.size(), 0);
					std::vector<float> sp_label_depths(labels.rows*labels.cols, 0.0f);
					std::vector<int> pts2d_size(sp_labels.size());
					std::vector<cv::Point2f> NoDepthPts2d(NoDepthPt2dCount);  // �������pt2d��
					std::vector<int> NoDepthPt2DSPLabelsRet(NoDepthPt2dCount, 0);  // ���������������
					std::vector<int> NoDepthPts2dLabelIdx(NoDepthPt2dCount);  // ����ÿ��pt2d���label_idx
					std::vector<int> NoDepthPts2DSPLabelsPre(NoDepthPt2dCount, 0);  // ÿ��pt2d��ǰһ�ε�label
					std::vector<float> sp_label_plane_arrs(Label2Pts2d.size() * 4 + 9, 0.0f);

					// ---���NoDepthPts2d(��MRF�Ż���pt2d��)
					// ---���NoDepthPt2DSPLabelsPre����
					stride = 0;
					for (auto it = has_no_depth_map.begin();
						it != has_no_depth_map.end(); ++it)
					{
						// ����MRF�Ż���pt2d��
						memcpy(NoDepthPts2d.data() + stride,
							it->second.data(),
							sizeof(cv::Point2f) * it->second.size());

						// ���pt2d��ĳ�ʼ��label
						std::vector<int> tmp(it->second.size(), it->first);  // sp label
						memcpy(NoDepthPts2DSPLabelsPre.data() + stride,
							tmp.data(),
							sizeof(int) * it->second.size());

						stride += int(it->second.size());
					}  // ���ⲻ��Ҫ��push_back

					// -----����NeighborMap����Neighbors�����label idx��Ӧ���ھ�����
					const auto NeighborMap = this->GetNeighborMap(labels);
					std::vector<int> sp_label_neighs_idx;  // ����Label2Pts2d��������
					std::vector<int> sp_label_neigh_num(Label2Pts2d.size(), 0);  // ����Label2Pts2d��������

					stride = 0;
					sp_count = 0;
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						// Label2Pts2d��de����
						sp_labels[sp_count] = it->first;

						// ���pts2d_size
						pts2d_size[sp_count] = (int)it->second.size();

						// ���ƽ�淽������
						memcpy(sp_label_plane_arrs.data() + sp_count * 4,
							plane_map.at(it->first).data(),
							sizeof(float) * 4);

						// ���sp_label_neighbor_num����
						sp_label_neigh_num[sp_count] = (int)NeighborMap.at(it->first).size();

						// ����sp_count
						sp_count++;

						// ������ֵ����: ����Label2Pts2d������
						float* ptr_depths = sp_label_depths.data() + stride;
						int pt_i = 0;
						for (auto pt2d : it->second)
						{
							ptr_depths[pt_i++] = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
						}

						// ����stride
						stride += (int)it->second.size();
					}

					// ��K_inv_arr��ӵ�����sp_label_plane_arrs�����
					memcpy(sp_label_plane_arrs.data() + int(Label2Pts2d.size()) * 4,
						K_inv_arr,
						sizeof(float) * 9);

					// ���sp_label_neighs_idx����
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						for (int neigh_label : NeighborMap.at(it->first))
						{
							auto it = std::find(sp_labels.begin(), sp_labels.end(), neigh_label);
							int label_idx = std::distance(std::begin(sp_labels), it);
							sp_label_neighs_idx.push_back(label_idx);  // label_idx����label
						}
					}

					// ���NoDepthPts2dLabelIdx����
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

					// ----- GPU������MRF�Ż�...
					int radius = 20;
					float beta = 1.0f;
					const int WIDTH = depth_map.GetWidth();
					const int HEIGHT = depth_map.GetHeight();

					// --- MRF����
					//int num_bad_pt2d = INT32_MAX, iter_i = 0;
					//while (iter_i < 10 && (float)num_bad_pt2d / (float)NoDepthPts2d.size() > 0.05f)
					//for (int iter_i = 0; iter_i < NUM_MRF_ITER; ++iter_i)
					//{
					std::printf("Start GPU MRF...\n");

					// ��ȡ��ʼ����depth_mat
					depth_mat = depth_map.Depth2Mat();

					//MRFGPU(labels,
					//	sp_labels, sp_label_depths,
					//	pts2d_size,
					//	NoDepthPts2d,
					//	sp_label_plane_arrs,
					//	radius, WIDTH, HEIGHT, beta,
					//	NoDepthPt2DSPLabelsRet);
					//MRFGPU2(labels,						 // ��Ҫ����
					//	sp_labels,
					//	sp_label_depths,                 // ��Ҫ����
					//	pts2d_size,                      // ��Ҫ����
					//	NoDepthPts2d,
					//	NoDepthPts2dLabelIdx,			 // ��Ҫ����
					//	sp_label_plane_arrs,             // �費��Ҫ�������, ����?
					//	sp_label_neighs_idx,			 // ��Ҫ����
					//	sp_label_neigh_num,				 // ��Ҫ����
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
						labels,						     // ��Ҫ����
						sp_labels,
						sp_label_depths,                 // ��Ҫ����
						pts2d_size,                      // ��Ҫ����
						NoDepthPts2d,
						NoDepthPts2dLabelIdx,			 // ��Ҫ����
						sp_label_plane_arrs,             // �費��Ҫ�������, ����?
						sp_label_neighs_idx,			 // ��Ҫ����
						sp_label_neigh_num,				 // ��Ҫ����
						radius, WIDTH, HEIGHT, beta,
						NoDepthPt2DSPLabelsRet);
					std::printf("GPU MRF done\n");

#ifdef LOGGING
					// ��ӡlabel������Ϣ
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

					// ----- MRF����
					// ����labels����(opencv Mat)
					for (int pt_i = 0; pt_i < (int)NoDepthPts2d.size(); ++pt_i)
					{
						const int& label = NoDepthPt2DSPLabelsRet[pt_i];
						if (Label2Pts2d.find(label) != Label2Pts2d.end())
						{
							const cv::Point2f& pt2d = NoDepthPts2d[pt_i];
							labels.at<int>((int)pt2d.y, (int)pt2d.x) = label;  // ����pt2d���label
						}
						else
						{
							std::printf("Wrong label: %d\n", label);
						}
					}
					std::printf("labels updated\n");

					// ����Label2Pts2d: ����labels��������Label2Pts2d, Ч�ʸ���
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						it->second.clear();  // Label2Pts2d������, ֵ��Ҫ����
					}
					for (int y = 0; y < labels.rows; ++y)
					{
						for (int x = 0; x < labels.cols; ++x)
						{
							const int& label = labels.at<int>(y, x);

							// label -> ͼ��2D����㼯
							Label2Pts2d[label].push_back(cv::Point2f(float(x), float(y)));
						}
					}
					std::printf("Labels2Pts2d updated\n");

					// ����sp_label_depths��pts2d_size
					sp_count = 0, stride = 0;
					sp_label_depths.resize(labels.cols*labels.rows, 0.0f);
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						// ����pts2d_size
						pts2d_size[sp_count++] = (int)it->second.size();

						// �������ֵ����: ����Label2Pts2d������
						float* ptr_depths = sp_label_depths.data() + stride;
						int pt_i = 0;
						for (auto pt2d : it->second)
						{
							ptr_depths[pt_i++] = depth_map.GetDepth((int)pt2d.y, (int)pt2d.x);
						}

						// ����stride
						stride += (int)it->second.size();
					}
					std::printf("sp_label_depths and pts2d_size updated\n");

					// ��NoDepthPt2DSPLabelsRet����NoDepthPt2DSPLabelsPre
					memcpy(NoDepthPts2DSPLabelsPre.data(), NoDepthPt2DSPLabelsRet.data(),
						sizeof(int) * (size_t)NoDepthPt2dCount);
					std::printf("NoDepthPt2DSPLabelsPre updated\n");

					// ����NoDepthPts2DSPLabelsPre����NoDepthPts2dLabelIdx(ÿ�����Ӧ��label idx)����
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

					// ---����Label2Pts2d, ����sp_label_neighs_idx
					// ����labels, ���¼���neighbor_map
					auto neighbor_map = this->GetNeighborMap(labels);
					std::printf("NeighborMap updated\n");

					// ����������¼���sp_label_neighs_idx
					sp_label_neighs_idx.clear();
					sp_label_neigh_num.clear();
					sp_label_neigh_num.resize(Label2Pts2d.size(), 0);

					sp_count = 0;  // ���¼���
					for (auto it = Label2Pts2d.begin(); it != Label2Pts2d.end(); ++it)
					{
						// ����sp_label_neighbor_num����
						sp_label_neigh_num[sp_count++] = (int)neighbor_map.at(it->first).size();

						// ����sp_label_neighs_idx����
						for (int neigh_label : neighbor_map.at(it->first))
						{
							auto it = std::find(sp_labels.begin(), sp_labels.end(), neigh_label);
							int label_idx = std::distance(std::begin(sp_labels), it);
							sp_label_neighs_idx.push_back(label_idx);  // label_idx����label
						}
					}
					std::printf("sp_label_neighs_idx and sp_label_neigh_num updated\n");

					// ---����depth_mat
					// ---- GPU������JBUSP
					// �ͷž��ڴ�
					pts2d_no_depth_jbu.clear();
					pts2d_no_depth_jbu.shrink_to_fit();
					sp_labels_idx_jbu.clear();
					sp_labels_idx_jbu.shrink_to_fit();

					// �ռ���Ҫ����JBUSP��pt2d��
					for (int pt_i = 0; pt_i < (int)NoDepthPts2d.size(); ++pt_i)
					{
						const cv::Point2f& pt2d = NoDepthPts2d[pt_i];
						const int& label = labels.at<int>((int)pt2d.y, (int)pt2d.x);

						// ���Ȼ���ƽ��Լ��, �������ֵ
						float depth = 0.0f;
						const float* plane_arr = plane_map.at(label).data();
						depth = this->GetDepthCoPlaneCam(K_inv_arr, plane_arr, pt2d);

						if (depth <= 0.0f)
						{
							// ����Ҫ����JBUSP��pt2d��Ž�����
							pts2d_no_depth_jbu.push_back(pt2d);

							// �����pt2d���label idx
							auto it = std::find(sp_labels_jbu.begin(),
								sp_labels_jbu.end(),
								label);
							int label_idx = std::distance(std::begin(sp_labels_jbu), it);
							sp_labels_idx_jbu.push_back(label_idx);
						}
						else
						{
							// ����ƽ��Լ���������ֵ
							depth_map.Set((int)pt2d.y, (int)pt2d.x, depth);
						}
					}

					if (0 < pts2d_no_depth_jbu.size())
					{
						// ����GPU
						depths_ret.clear();
						depths_ret.resize(pts2d_no_depth_jbu.size(), 0.0f);
						std::printf("Start GPU JBUSP...\n");
						JBUSPGPU(src, depth_mat,
							pts2d_no_depth_jbu,  // �������pt2d������
							sp_labels_idx_jbu,  // ÿ��������dept2d���Ӧ��label idx
							pts2d_has_depth_jbu,
							sp_has_depth_pt2ds_num,
							sigmas_s_jbu,
							depths_ret);
						std::printf("GPU JBUSP done\n");

						// ���depth_map���
						for (int i = 0; i < (int)pts2d_no_depth_jbu.size(); ++i)
						{// �˴���depth_map������д�����!!!
							const cv::Point2f& pt2d = pts2d_no_depth_jbu[i];
							depth_map.Set((int)pt2d.y, (int)pt2d.x, depths_ret[i]);
						}

						std::printf("Depthmap completed\n");
						std::printf("Iteration %d done\n", iter_i + 1);
					}

					// ���°߿��˳�����
					//denominator -= 10.0f;
					//ratio -= 0.005;
					denominator *= 0.95f;
					ratio *= 0.95f;
#endif // MRF_GPU
				}

				// ----- ���Filled���������ͼ: �����ؽ����������ͼ����
				string filled_out_name = file_name + ".geometric.bin";
				viz_out_path = depth_dir + filled_out_name;
				depth_map.WriteBinary(viz_out_path);
				std::printf("%s written\n", filled_out_name.c_str());

				// ���Filled���ͼ���ڿ��ӻ�[0, 255]
				string filled_name = file_name + "_filled.jpg";
				viz_out_path = depth_dir + filled_name;
				cv::imwrite(viz_out_path, depth_map.ToBitmapGray(2, 98));;;
				std::printf("%s written\n\n", filled_name.c_str());

				//// ----- ΪFilled���ͼ����mask, ������ͼ+super-pixel mask ���ڿ��ӻ�[0, 255]
				//cv::Mat depth_filled = cv::imread(viz_out_path, cv::IMREAD_COLOR);
				//this->DrawMaskOfSuperpixels(labels, depth_filled);
				//// ����superpixel���
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
			const float* K_inv_arr,  // ����ڲξ������
			std::vector<int>& blks_pt_cnt,  // ��¼����block��pt2d����
			std::vector<cv::Point2f>& blks_pts2d,  // ��¼����block��pt2d������
			std::vector<int>& blks_pt_cnt_has,  // ��¼������block�������ֵ�����
			std::vector<int>& blks_pt_cnt_non,  // ��¼������block�������ֵ�����
			std::vector<std::vector<float>>& plane_equa_arr,  // ��¼��Ϊlabel��blk_id��Ӧ��ƽ�淽��
			std::vector<int>& label_blk_ids,  // ��¼���㹻�����ֵ���blk_id: �ɵ���label
			std::vector<int>& proc_blk_ids,  // ��¼��(MRF)�����blk_id
			std::vector<float>& proc_blks_depths_has,  // ��¼������block�������ֵ(��ɵ�����)
			std::vector<int>& proc_blks_pts2d_has_num,  // ��¼������block�������ֵ�����
			std::vector<int>& proc_blks_pts2d_non_num,  // ��¼������block����ȵ����
			std::vector<cv::Point2f>& proc_blks_pt2d_non,  // ��¼������block�������ֵ������
			std::vector<int>& all_blks_labels,  // ��¼ÿ��block��Ӧ��label(blk_id): ��ʼlabel����
			int& num_y, int& num_x)  // ��¼blk�ܵ�����, ����
		{
			// ���黮��depth mat
			const int& HEIGHT = depth_mat.rows;
			const int& WIDTH = depth_mat.cols;
			const int remain_y = HEIGHT % blk_size;
			const int remain_x = WIDTH % blk_size;

			// block idx����label
			int blk_id = 0;

			if (remain_x && remain_y)
			{

			}
			else if (remain_x)  // y����, xû����
			{

			}
			else if (remain_y)  // x����, yû����
			{
				// Ϊblks_pts2d�����ڴ�
				blks_pts2d.resize(HEIGHT*WIDTH);

				num_y = (HEIGHT / blk_size) + 1;
				num_x = WIDTH / blk_size;

				// Ϊblks_labels�����ڴ��С
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
						// ����ÿһ��
						if (i == num_y - 1)  // �߽��ϵ�block
						{

						}
						else
						{
							// ͳ����Ч���ֵ�������ֵ�ĵ�
							const int y_start = i * blk_size;
							const int y_end = (i + 1) * blk_size;
							const int x_start = j * blk_size;
							const int x_end = (j + 1) * blk_size;

							std::vector<cv::Point2f> blk_pts2d_has;  // ÿ��block�����ֵ�ĵ�
							int blk_pt_cnt = 0, blk_pt_cnt_has = 0, blk_pt_cnt_non = 0;

							cv::Point2f* ptr_blks_pts2d = blks_pts2d.data() + blk_stride;
							for (int y = y_start; y < y_end; ++y)
							{
								for (int x = x_start; x < x_end; ++x)
								{
									// ���ptr_blks_pts2d����
									ptr_blks_pts2d[blk_pt_cnt++] = cv::Point2f(float(x), float(y));

									const float& depth = depth_mat.at<float>(y, x);
									if (depth > 0.0f)
									{
										// ����block����ȵ����
										blk_pt_cnt_has++;

										// ��¼����block�������ֵ�ĵ�
										blk_pts2d_has.push_back(cv::Point2f((float)x, (float)y));
									}
									else
									{
										// ����block����ȵ����
										blk_pt_cnt_non++;
									}
								}
							}

							// ����blk_stride
							blk_stride += blk_pt_cnt;

							// ��¼ÿ��block��pt2d�����
							blks_pt_cnt[blk_id] = blk_pt_cnt;

							// ��¼ÿ��block��Ӧ�������ֵ����, �����ֵ����
							blks_pt_cnt_has[blk_id] = blk_pt_cnt_has;
							blks_pt_cnt_non[blk_id] = blk_pt_cnt_non;

							// ���block��������ȵ�, ��Ϊ������block
							// ���������block�������ֵ����
							if (blk_pt_cnt_non > 0)
							{
								// ֻҪblock�г���1�������ֵ�㼴��ҪMRF�Ż�label
								proc_blk_ids.push_back(blk_id);

								// ���process_blks_pts2d_num����
								proc_blks_pts2d_has_num.push_back(blk_pt_cnt_has);

								// ���process_blks_pts2d_non_num����
								proc_blks_pts2d_non_num.push_back(blk_pt_cnt_non);

								for (int y = y_start; y < y_end; ++y)
								{
									for (int x = x_start; x < x_end; ++x)
									{
										// ���blks_pts2d����

										const float& depth = depth_mat.at<float>(y, x);
										if (depth > 0.0f)
										{
											// ���blks_depths_has����
											proc_blks_depths_has.push_back(depth);
										}
										else
										{
											// ���blks_pt2d_non����
											proc_blks_pt2d_non.push_back(cv::Point2f((float)x, (float)y));
										}
									}
								}
							}

							// ���㹻���������ֵ�ĵ�, ��¼label
							if (float(blk_pt_cnt_has) / float(blk_size*blk_size) >= 0.9f)
							{
								// ͳ��label(blk_id)
								label_blk_ids.push_back(blk_id);

								// ���blks_labels����: ��ʱblk_id����
								all_blks_labels[blk_id] = blk_id;

								// ----- ����3D�������ϵ�µ�����
								std::vector<cv::Point3f> Pts3D;
								Pts3D.resize(blk_pts2d_has.size());
								for (int i = 0; i < (int)blk_pts2d_has.size(); i++)
								{
									cv::Point3f Pt3D = this->m_model.BackProjTo3DCam(K_inv_arr,
										depth_mat.at<float>((int)blk_pts2d_has[i].y, (int)blk_pts2d_has[i].x),
										blk_pts2d_has[i]);
									Pts3D[i] = Pt3D;
								}

								// 3D�ռ�ƽ�����: using OLS, SVD or PCA
								RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
								ransac.RunRansac(Pts3D);
								std::vector<float> plane_equa(4);
								memcpy(plane_equa.data(), ransac.m_plane, sizeof(float) * 4);

								// ���plane_equa_arr����: ��label_blk_ids��˳��һ��
								plane_equa_arr.push_back(plane_equa);

							}  // ��blockԤ�������

							// ����blk_id, Ԥ������һ��block
							blk_id++;
						}
					}
				}

				// �������all_blks_labels����, Ϊ�������blk_id�����ֵlabel
				// ��all_blks_labels�����ѡȡ
				for (int blk_id : proc_blk_ids)
				{
					all_blks_labels[blk_id] = label_blk_ids[rand() % int(label_blk_ids.size())];
				}

			}
			else  // x, y��ȫ����
			{

			}

			return 0;
		}

		// �ֿ�MRF
		void Workspace::TestDepth7()
		{
			const string depth_dir = this->options_.workspace_path \
				+ "/dense/stereo/depth_maps/dslr_images_undistorted/";
			const string src_dir = this->options_.workspace_path \
				+ "/dense/images/dslr_images_undistorted/";

			// �ܵ��ӽǸ���
			const int NumView = (int)this->m_model.m_images.size();
			for (int img_id = 0; img_id < NumView; ++img_id)  // ע��: img_id != IMAGE_ID
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
				depth_map.fillDepthWithMat(depth_mat);  // �˴���depth_map������д�����!!!

				// ����, ����filtered��depth_mat
				cv::Mat depth_mat_filter = depth_mat.clone();

				// write to disk for visualization
				string filter_name = file_name + "_filtered_orig.jpg";
				string filter_path = depth_dir + filter_name;
				cv::imwrite(filter_path, depth_map.ToBitmapGray(2, 98));
				std::printf("%s written\n", filter_name.c_str());

				// ----------- super-pixel segmentation
				cv::Mat src, mask, labels;

				// ԭͼ��ȡBGR��ɫͼ
				src = cv::imread(src_dir + file_name, cv::IMREAD_COLOR);
				if (src.empty())
				{
					std::printf("[Err]: empty src image\n");
					return;
				}

				// ----- super-pixel segmentation using SEEDS or SLIC
				// SEEDS super-pixel segmentation
				const int num_superpixels = 700;  // ����ĳ�ʼ�ָ֤�߽�
				Ptr<cv::ximgproc::SuperpixelSEEDS> superpixel = cv::ximgproc::createSuperpixelSEEDS(src.cols,
					src.rows,
					src.channels(),
					num_superpixels,  // num_superpixels
					15,  // num_levels: 5, 15
					2,
					5,
					true);
				superpixel->iterate(src);  // ����������Ĭ��Ϊ4
				superpixel->getLabels(labels);  // ��ȡlabels
				superpixel->getLabelContourMask(mask);  // ��ȡ�����صı߽�

				// construct 2 Hashmaps for each super-pixel
				std::unordered_map<int, std::vector<cv::Point2f>> Label2Pts2d,
					has_depth_map, has_no_depth_map;

				// traverse each pxiel to put into hashmaps
				for (int y = 0; y < labels.rows; ++y)
				{
					for (int x = 0; x < labels.cols; ++x)
					{
						const int& label = labels.at<int>(y, x);

						// label -> ͼ��2D����㼯
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
				// ----- Ϊԭͼ����superpixel merge֮ǰ��mask
				cv::Mat src_1 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_1);
				string mask_src1_name = file_name + "_before_merge_mask.jpg";
				viz_out_path = depth_dir + mask_src1_name;
				cv::imwrite(viz_out_path, src_1);
				std::printf("%s written\n", mask_src1_name.c_str());
#endif // DRAW_MASK

				// ----- superpixel�ϲ�(�ϲ���Чdepth������ֵ��)
				std::printf("Before merging, %d superpixels\n", has_depth_map.size());
				this->MergeSuperpixels(src,
					2000,  // 2000
					labels,
					Label2Pts2d, has_depth_map, has_no_depth_map);
				std::printf("After merging, %d superpixels\n", has_depth_map.size());

#ifdef DRAW_MASK
				// ----- Ϊԭͼ����superpixel merge֮���mask
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

				// ----- ����Merge���superpixels, ƽ�����
				std::unordered_map<int, std::vector<float>> eigen_vals_map;  // superpixel������ֵ
				std::unordered_map<int, std::vector<float>> eigen_vects_map;  // superpixel��������
				std::unordered_map<int, std::vector<float>> plane_normal_map;  // superpxiel�ķ�����
				std::unordered_map<int, std::vector<float>> plane_map;  // superpixel����ƽ�淽��
				std::unordered_map<int, cv::Point3f> center_map;  // superpixel�����ĵ�����
				this->FitPlaneForSPsCam(depth_map,
					K_inv_arr,
					Label2Pts2d, has_depth_map,
					center_map, eigen_vals_map, eigen_vects_map,
					plane_normal_map, plane_map);
				std::printf("Superpixel plane fitting done\n");

				//// ----- ����tagent plane
				//this->CorrectPCPlaneCam(K_arr, K_inv_arr,
				//	src.cols, src.rows,
				//	depth_range * 0.05f,
				//	3.0f,
				//	Label2Pts2d,
				//	plane_map, center_map, plane_normal_map,
				//	eigen_vals_map, eigen_vects_map);
				//std::printf("Superpixel plane correction done\n");

				// ----- superpixel����: ���ӿ����ӵ�����superpixel,
				this->ConnectSuperpixelsCam(0.0002f, 0.2f, depth_map,
					K_inv_arr,
					plane_map,
					eigen_vals_map, plane_normal_map, center_map,
					labels, Label2Pts2d, has_depth_map, has_no_depth_map);
				printf("Connect done, after connect, %d superpixels\n",
					has_depth_map.size());

#ifdef DRAW_MASK
				// ----- Ϊԭͼ����Connect֮���mask
				cv::Mat src_3 = src.clone();
				this->DrawMaskOfSuperpixels(labels, src_3);
				string mask_src3_name = file_name + "_after_connect_mask.jpg";
				viz_out_path = depth_dir + mask_src3_name;
				cv::imwrite(viz_out_path, src_3);
				printf("%s written\n", mask_src3_name.c_str());

				// ----- �����depth_map֮ǰ, Ϊdepth_map����mask(�����㷨����)
				cv::Mat depth_mask = cv::imread(filter_path, cv::IMREAD_COLOR);
				this->DrawMaskOfSuperpixels(labels, depth_mask);
				string filter_mask = file_name + "_depth_filter_mask.jpg";
				viz_out_path = depth_dir + filter_mask;
				cv::imwrite(viz_out_path, depth_mask);
				std::printf("%s written\n", filter_mask.c_str());
#endif // DRAW_MASK

				// ----- ����������Ĺ��ü�������
				int stride = 0, sp_count = 0, pt_count = 0;

				// ----- GPU�˳�ʼ��depth_map, JBUSPGPU����
				// ---- ����GPU�������
				// 1. ---���¾����������ֵ��pt2d, ��depth < 0�ķ���һ������
				// 2. ---ͳ��depth<0��pt2d�����Ϣ:
				// pts2d_has_no_depth_jbu, sp_labels_idx_jbu
				std::vector<cv::Point2f> pts2d_has_no_depth_jbu;  // ���д������pt2d��
				std::vector<int> sp_labels_idx_jbu;  // ÿ�����sp label index

				// ---JBUSPGPU��׼����������4������ĳ�ʼ��: 
				// (1). sp_labels_jbu, (2). pts2d_has_depth_jbu
				// (3). sp_has_depth_pt2ds_num (4). sigmas_s_jbu
				// pts2d_has_depth_jbu,
				// ����GPU��JBUSP��sp label����
				std::vector<int> sp_labels_jbu(has_no_depth_map.size(), 0);

				// ---ͳ�������ֵpt2d���ܹ��ĵ���
				pt_count = 0;
				for (auto it = has_no_depth_map.begin();
					it != has_no_depth_map.end(); ++it)
				{
					pt_count += (int)has_depth_map.at(it->first).size();
				}

				// ����label��Ӧ��pt2d������: ����has_no_depth_map��label˳��
				std::vector<cv::Point2f> pts2d_has_depth_jbu(pt_count);

				// ÿ��label��Ӧ��pt2d�����
				std::vector<int> sp_has_depth_pt2ds_num(has_no_depth_map.size());

				// ����sigma_s����: ÿ��label����Ӧһ��sigma_sֵ
				std::vector<float> sigmas_s_jbu(has_no_depth_map.size());

				stride = 0, sp_count = 0;  // sp_count��has_no_depth_map��label_idx
				for (auto it = has_no_depth_map.begin();
					it != has_no_depth_map.end(); ++it)
				{
					// ����sp_labels_jbu
					sp_labels_jbu[sp_count] = it->first;

					// ����pts2d_has_depth_jbu
					memcpy(pts2d_has_depth_jbu.data() + stride,
						has_depth_map[it->first].data(),
						sizeof(cv::Point2f) * has_depth_map[it->first].size());

					// ����sp_has_depth_pt2ds_num
					sp_has_depth_pt2ds_num[sp_count] = int(has_depth_map[it->first].size());

					// ����space_sigma
					sigmas_s_jbu[sp_count] = this->GetSigmaOfPts2D(Label2Pts2d[it->first]);

					// ���pts2d_has_no_depth_jbu�����sp_labels_idx_jbu����
					for (auto pt2D : has_no_depth_map[it->first])
					{
						// ƽ��Լ�������ֵ
						float depth = this->GetDepthCoPlaneCam(K_inv_arr,
							plane_map.at(it->first).data(), pt2D);

						// Ҫ����JBUSP��pt2d��
						if (depth <= 0.0f)
						{
							pts2d_has_no_depth_jbu.push_back(pt2D);
							sp_labels_idx_jbu.push_back(sp_count);
						}
						else
						{
							// ����ƽ��Լ��, ����depth
							depth_map.Set(pt2D.y, pt2D.x, depth);  // �˴���depth_map������д�����!!!
					}
				}

					sp_count += 1;
					stride += int(has_depth_map[it->first].size());
			}
				std::printf("GPU preparations for JBUSP built done, total %d pts for JBUSP\n",
					(int)pts2d_has_no_depth_jbu.size());

				assert(pts2d_has_depth_jbu.size()
					== std::accumulate(sp_has_depth_pt2ds_num.begin(), sp_has_depth_pt2ds_num.end(), 0));

				// GPU������JBUSP
				std::vector<float> depths_ret(pts2d_has_no_depth_jbu.size(), 0.0f);
				std::printf("Start GPU JBUSP...\n");
				JBUSPGPU(src,
					depth_mat,
					pts2d_has_no_depth_jbu,  // �������pt2d������
					sp_labels_idx_jbu,  // ÿ��������dept2d���Ӧ��label idx
					pts2d_has_depth_jbu,
					sp_has_depth_pt2ds_num,
					sigmas_s_jbu,
					depths_ret);
				std::printf("GPU JBUSP done\n");

				// ���depth_map���
				for (int i = 0; i < (int)pts2d_has_no_depth_jbu.size(); ++i)
				{// �˴���depth_map������д�����!!!
					depth_map.Set((int)pts2d_has_no_depth_jbu[i].y,
						(int)pts2d_has_no_depth_jbu[i].x, depths_ret[i]);
				}

				// ���³�ʼ��֮���depth_mat
				cv::Mat init_depth_mat = depth_map.Depth2Mat();

				std::printf("Depthmap initialized\n");

				//----- �ֿ�MRF�Ż�
				int blk_size = 5, radius = 50;
				const float beta = 1.0f;
				std::vector<int> blks_pt_cnt;  // ��¼����block��pt2d����
				std::vector<cv::Point2f> blks_pts2d;  // ��¼����block��pt2d������
				std::vector<int> blks_pt_cnt_has;  // ��¼������block�������ֵ�����
				std::vector<int> blks_pt_cnt_non;  // ��¼������block�������ֵ�����
				std::vector<std::vector<float>> plane_equa_arr;  // ��¼��Ϊlabel��blk_id��Ӧ��ƽ�淽��
				std::vector<int> label_blk_ids;  // ��¼���㹻�����ֵ���blk_id: �ɵ���label
				std::vector<int> proc_blk_ids;  // ��¼��(MRF)�����blk_id
				std::vector<float> proc_blks_depths_has;  // ��¼������block�������ֵ(��ɵ�����)
				std::vector<int> proc_blks_pts2d_has_num;  // ��¼������block�������ֵ�����
				std::vector<int> proc_blks_pts2d_non_num;  // ��¼������block����ȵ����
				std::vector<cv::Point2f> proc_blks_pt2d_non;  // ��¼������block�������ֵ������
				std::vector<int> all_blks_labels;  // ��¼ÿ��block��Ӧ��label(blk_id): ��ʼlabel����
				int num_y, num_x;  // y, x����block����
				std::vector<int> label_ids_ret;  // ���ؽ������

				maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					/ 10.0f);
				maxDiff = 0.04f * depth_range;

				const int NumIter = 3;
				for (int iter_i = 0; iter_i < NumIter; ++iter_i)
				{
					std::printf("Block size: %d\n", blk_size);

					// �ָ�depth_mat
					this->SplitDepthMat(depth_mat_filter,  // ʹ��filtered֮���depth_mat
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

					// ��ʼ��label_ids_ret����
					label_ids_ret.resize(proc_blk_ids.size(), 0);

					// ����GPU block MRF
					BlockMRF(init_depth_mat,  // ΪһԪ������׼��
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

					// ����all_blks_labels
					for (int proc_i = 0; proc_i < (int)proc_blk_ids.size(); ++proc_i)
					{
						const int& blk_id_old = proc_blk_ids[proc_i];
						const int& blk_id_new = label_blk_ids[label_ids_ret[proc_i]];
						all_blks_labels[blk_id_old] = blk_id_new;
					}

					// ����depth_map, ����ÿһ��MRF�Ż���block
					for (int proc_i = 0; proc_i < (int)proc_blk_ids.size(); ++proc_i)
					{
						// ȡ�ɵ�block idx
						const int& blk_id_old = proc_blk_ids[proc_i];

						// ȡ��label��Ӧ��plane equation
						const float* pl_arr = plane_equa_arr[label_ids_ret[proc_i]].data();

						// �жϸ�block��ȫ�ջ��ǲ��ֿ�
						if (proc_blks_pts2d_non_num[proc_i] == blks_pt_cnt[blk_id_old])  // ȫ��block
						{
							// ---��blockÿ��pt2d�㶼ͨ����label��ƽ��Լ����ֵ
							// --������block��ÿһ����
							// ����offset
							int offset = 0;
							for (int idx = 0; idx < blk_id_old; ++idx)
							{
								offset += blks_pt_cnt[idx];
							}

							// ����block pt2d�㼯��ʼָ��
							const cv::Point2f* ptr_blks_pts2d = blks_pts2d.data() + offset;

							// ����blockÿһ��pt2d��
							for (int k = 0; k < blks_pt_cnt[blk_id_old]; ++k)
							{
								const cv::Point2f& pt2d = ptr_blks_pts2d[k];

								// ƽ��Լ����ֵ
								const float depth = this->GetDepthCoPlaneCam(K_inv_arr, pl_arr, pt2d);
								depth_map.Set((int)pt2d.y, (int)pt2d.x, depth);
							}
						}
						else if (proc_blks_pts2d_non_num[proc_i] < blks_pt_cnt[blk_id_old])  // ��ȫ��block
						{
							// �ǿղ��ֲ�������, �ҳ����ֵΪ�յ�pt2d��,
							// ���Ϊ�ղ���ͨ����labelƽ��Լ����ֵ
							// ��������ȵ������offset
							int offset = 0;
							for (int j = 0; j < proc_i; ++j)
							{
								offset += proc_blks_pts2d_non_num[j];
							}

							// ��������ȵ����鿪ʼָ��
							const cv::Point2f* ptr_proc_blks_pt2d_non = proc_blks_pt2d_non.data() + offset;

							for (int k = 0; k < proc_blks_pts2d_non_num[proc_i]; ++k)
							{
								const cv::Point2f& pt2d = ptr_proc_blks_pt2d_non[k];

								// ƽ��Լ����ֵ
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

					// ����blk_size, radius
					blk_size = blk_size * 2;

					// ����ڴ�
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

					// ----- for debug... �������ļ�
					// ����depth_map
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

					// ----- Ϊ��һ�ֵ�����׼��
					// --- ����depth_mat
					depth_mat = depth_map.Depth2Mat();

					// --- ����depth_mat_filter
					depth_mat_filter = depth_mat.clone();

					// --- �˳��߿�, speckle filtering for depth_mat
					this->FilterSpeckles<float>(depth_mat_filter, 0.0f, maxSpeckleSize, maxDiff);

					// ����filtered
					depth_map.fillDepthWithMat(depth_mat_filter);
					sprintf(buff, "_iter%d_filterSpecke_32F.jpg", iter_i + 1);
					string iter_filter_out_name = file_name + string(buff);
					viz_out_path = depth_dir + iter_filter_out_name;
					cv::imwrite(viz_out_path, depth_map.ToBitmapGray(2, 98));
					std::printf("%s written\n", iter_filter_out_name.c_str());

					std::printf("Iter %d done\n", iter_i + 1);
		}

				// ----- ���Filled���������ͼ: �����ؽ����������ͼ����
				string filled_out_name = file_name + ".geometric.bin";
				viz_out_path = depth_dir + filled_out_name;
				depth_map.WriteBinary(viz_out_path);  // д��depth_mapΪbin�ļ�
				std::printf("%s written\n", filled_out_name.c_str());

				// ���Filled���ͼ���ڿ��ӻ�[0, 255]
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

			//	// ����ÿ���ӽ�
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

			//		// ������opencv Mat
			//		cv::Mat& depth_mat = depth_map.Depth2Mat();

			//		const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
			//			/ 100.0f);

			//		// using depth range
			//		const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
			//		const float maxDiff = 0.1f * depth_range;

			//		// ----- speckle filtering for depth_mat
			//		this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

			//		// ��ȡROI
			//		cv::Mat ROI(depth_mat, cv::Rect2i(20, 1, 1960, 1330));
			//		printf("ROI: %d��%d\n", ROI.cols, ROI.rows);

			//		// ��ȡ�������
			//		//const float* P_arr = this->m_model.m_images[img_id].GetP();
			//		//const float* K_arr = this->m_model.m_images[img_id].GetK();
			//		//const float* R_arr = this->m_model.m_images[img_id].GetR();
			//		const float* T_arr = this->m_model.m_images[img_id].GetT();
			//		const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();
			//		const float* R_inv_arr = this->m_model.m_images[img_id].GetInvR();

			//		// ��ȡROI����
			//		cv::Mat_<cv::Vec3f> cloud(ROI.rows, ROI.cols);
			//		for (int r = 0; r < ROI.rows; r++)
			//		{
			//			// ���ͼһ�е�ָ��
			//			const float* depth_ptr = ROI.ptr<float>(r);

			//			// ����һ�е�ָ��
			//			cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
			//			for (int c = 0; c < ROI.cols; c++)
			//			{
			//				// ��ȡ3D����
			//				const float& depth = depth_ptr[c];

			//				// �ж����ֵ�Ƿ���Ч


			//				cv::Point3f pt3d = this->m_model.BackProjTo3D(K_inv_arr, R_inv_arr, T_arr, 
			//					depth, cv::Point2f(c, r));
			//				//printf("%.3f, %.3f, %.3f\n", pt3d.x, pt3d.y, pt3d.z);

			//				// m -> mm
			//				pt_ptr[c][0] = pt3d.x * 1000.0f;
			//				pt_ptr[c][1] = pt3d.y * 1000.0f;
			//				pt_ptr[c][2] = pt3d.z * 1000.0f;
			//			}
			//		}

			//		// ƽ����
			//		cv::Mat seg(ROI.rows, ROI.cols, CV_8UC3);
			//		OrganizedImage3D Ixyz(cloud);
			//		pf.run(&Ixyz, 0, &seg);

			//		// ���ӻ�ԭʼ���ͼ��ƽ������
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

			// ����ÿ���ӽ�
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

				// ������opencv Mat
				cv::Mat& depth_mat = depth_map.Depth2Mat();

				//// ----- speckle filtering for depth_mat
				//const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					//	/ 100.0f);
					//const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
					//const float maxDiff = 0.1f * depth_range;
					//this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

				float depth_max = depth_map.GetDepthMax();
				float depth_min = depth_map.GetDepthMin();

				// ��ȡROI
				cv::Mat ROI(depth_mat, cv::Rect2i(20, 1, 1960, 1330));
				printf("ROI: %d*%d\n", ROI.cols, ROI.rows);

				// ��ȡ�������
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

				// ��ȡROI����
				//cv::Mat depth_out(ROI.rows, ROI.cols, CV_16UC1);

				cv::Mat_<cv::Vec3f> cloud(ROI.rows, ROI.cols);
				for (int r = 0; r < ROI.rows; r++)
				{
					// ���ͼһ�е�ָ��
					const float* depth_ptr = ROI.ptr<float>(r);

					// ����һ�е�ָ��
					cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
					for (int c = 0; c < ROI.cols; c++)
					{
						// ��ȡ3D����
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

				//// ���u16���ͼ
				//int pos = color_filename.find_last_of("/\\");
				//string frame_name = color_filename.substr(pos + 1);
				//string out_depth_path = out_dir + frame_name + string("_16u.png");
				//cv::imwrite(out_depth_path, depth_out);

				//// �����ɫͼ
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

				//// ������ͼjpg
				//string depth_map_path = out_dir + depth_f_name + ".jpg";
				//cv::imwrite(depth_map_path, depth_map.ToBitmapGray(2, 98));
				//printf("%s written\n", depth_map_path.c_str());

				// ���ƽ������
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

			// ����ÿ���ӽ�
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

				// ������opencv Mat
				cv::Mat& depth_mat = depth_map.Depth2Mat();

				// ��ȡ�������
				const float* K_inv_arr = this->m_model.m_images[img_id].GetInvK();

				// �����Mat
				cv::Mat depth_out(depth_mat.rows, depth_mat.cols, CV_16UC1);

				// ����ԭʼ���ͼ
				for (int r = 0; r < depth_mat.rows; r++)
				{
					// ���ͼһ�е�ָ��
					const float* depth_ptr = depth_mat.ptr<float>(r);

					// ����һ�е�ָ��
					for (int c = 0; c < depth_mat.cols; c++)
					{
						// ��ȡ3D����
						depth_out.at<unsigned short>(r, c) = unsigned short(depth_ptr[c] * 1000.0f);
					}
				}

				// ���*1000��uint16���ͼ
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

			// ����ÿ���ӽ�
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

				// ������opencv Mat
				cv::Mat& depth_mat = depth_map.Depth2Mat();

				//// ----- speckle filtering for depth_mat
				//const int maxSpeckleSize = int(depth_map.GetWidth() * depth_map.GetHeight() \
					//	/ 100.0f);
					//const float depth_range = depth_map.GetDepthMax() - depth_map.GetDepthMin();
					//const float maxDiff = 0.1f * depth_range;
					//this->FilterSpeckles<float>(depth_mat, 0.0f, maxSpeckleSize, maxDiff);

					// ��ȡROI
				cv::Mat ROI(depth_mat, cv::Rect2i(20, 1, 1960, 1330));
				printf("ROI: %d*%d\n", ROI.cols, ROI.rows);

				// ��ȡ�������
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

				// ���ƽ������
				int pos = color_filename.find_last_of("/\\");
				string frame_name = color_filename.substr(pos + 1);
				frame_name += string("_nofilter_")
					+ std::to_string(plane_detection.plane_num_) + string("planes");
				plane_detection.writeOutputFiles(out_dir, frame_name, run_mrf);
				printf("%s written @ %s\n\n", frame_name.c_str(), out_dir);
			}
		}

		// superpixel��Ӧ�ĵ��������ƽ��
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

			// ----- ����3D��������ϵ�µ�����
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

			// ----- ����ÿ��superpixel������
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

			// ���ص�����������
			center_arr.reserve(3);
			center_arr.resize(3);
			memcpy(center_arr.data(), center, sizeof(float) * 3);

			// ----- PCAƽ�����RANSAC
			if (Pts3D.size() <= 5)
			{
				// using OLS, SVD or PCA
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
				ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

				// ����ƽ�淨����
				plane_normal.reserve(3);
				plane_normal.resize(3);
				memcpy(plane_normal.data(), ransac.m_plane, sizeof(float) * 3);

				// ��������ֵ
				eigen_vals.reserve(3);
				eigen_vals.resize(3);
				memcpy(eigen_vals.data(), ransac.m_eigen_vals, sizeof(float) * 3);
			}
			else
			{
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);  // using OLS, SVD or PCA
				ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

				// ����ƽ�淨����
				plane_normal.reserve(3);
				plane_normal.resize(3);
				memcpy(plane_normal.data(), ransac.m_plane, sizeof(float) * 3);

				// ��������ֵ
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

			// ----- ����3D�ռ�(��������ϵ�����������ϵ)������
			std::vector<cv::Point3f> Pts3D;
			Pts3D.resize(Pts2D.size());

			for (int i = 0; i < (int)Pts2D.size(); i++)
			{
				cv::Point3f Pt3D = this->m_model.BackProjTo3DCam(K_inv_arr,
					depth_map.GetDepth((int)Pts2D[i].y, (int)Pts2D[i].x),
					Pts2D[i]);
				Pts3D[i] = Pt3D;
			}

			// ----- ����ÿ��superpixel������
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

			// ���ص�����������
			center_arr.resize(3, 0.0f);
			memcpy(center_arr.data(), center, sizeof(float) * 3);

			// ----- PCAƽ�����RANSAC
			//if (Pts3D.size() <= 5)
			{
				// using OLS, SVD or PCA
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
				ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

				// ����ƽ�淽��
				plane_arr.resize(4, 0.0f);
				memcpy(plane_arr.data(), ransac.m_plane, sizeof(float) * 4);

				// ����ƽ�淨����
				plane_normal.resize(3, 0.0f);
				memcpy(plane_normal.data(), ransac.m_plane, sizeof(float) * 3);

				// ��������ֵ
				eigen_vals.resize(3, 0.0f);
				memcpy(eigen_vals.data(), ransac.m_eigen_vals, sizeof(float) * 3);
			}
			//else
			//{
			//	RansacRunner ransac(0.08f, (int)Pts3D.size(), 3);  // using OLS, SVD or PCA
			//	ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

			//	// ����ƽ�淽��
			//	plane_arr.resize(4, 0.0f);
			//	memcpy(plane_arr.data(), ransac.m_plane, sizeof(float) * 4);

			//	// ����ƽ�淨����
			//	plane_normal.resize(3, 0.0f);
			//	memcpy(plane_normal.data(), ransac.m_plane, sizeof(float) * 3);

			//	// ��������ֵ
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

			// ----- ����ÿ��superpixel������
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
				ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

				// ���ص�����������
				memcpy(center_arr, center, sizeof(float) * 3);

				// ����ƽ�淨����
				memcpy(plane_normal, ransac.m_plane, sizeof(float) * 3);

				// ��������ֵ
				memcpy(eigen_vals, ransac.m_eigen_vals, sizeof(float) * 3);

				// ������������
				memcpy(eigen_vects, ransac.m_eigen_vect, sizeof(float) * 9);
			}
			else
			{
				RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
				ransac.RunRansac(Pts3D);

				// ���ص�����������
				memcpy(center_arr, center, sizeof(float) * 3);

				// ����ƽ�淨����
				memcpy(plane_normal, ransac.m_plane, sizeof(float) * 3);

				// ��������ֵ
				memcpy(eigen_vals, ransac.m_eigen_vals, sizeof(float) * 3);

				// ������������
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
			// ----- ����superpixel����������
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

			// ���ص�����������
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
				ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

				// ����ƽ�淽��
				plane_arr.resize(4, 0.0f);
				memcpy(plane_arr.data(), ransac.m_plane, sizeof(float) * 4);

				// ��������ֵ
				eigen_vals.resize(3, 0.0f);
				memcpy(eigen_vals.data(), ransac.m_eigen_vals, sizeof(float) * 3);

				// ������������
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
			// ����ÿһ��superpixel
			for (auto it = labels_map.begin();
				it != labels_map.end(); it++)
			{
				if ((int)it->second.size() < 3)
				{
					printf("[Warning]: Not enough valid depth within super-pixel %d\n", it->first);
					continue;
				}

				// ȡ�������ֵ��2D��
				const std::vector<cv::Point2f>& Pts2D = has_depth_map[it->first];

				// ----- ����3D��������ϵ�µ�����
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

				// ----- ����ÿ��superpixel������
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

				// ���superpixel��Ӧ3D���Ƶ����ĵ�����
				center_map[it->first] = cv::Point3f(center_x, center_y, center_z);

				// ----- 3D plane fittin using RANSAC
				if (Pts3D.size() <= 5)
				{
					// using OLS, SVD or PCA
					RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
					ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

					// �������ֵeigen_vals_map
					eigen_vals_map[it->first].reserve(3);
					eigen_vals_map[it->first].resize(3);
					memcpy(eigen_vals_map[it->first].data(), ransac.m_eigen_vals, sizeof(float) * 3);

					// �����������: 9��Ԫ��
					eigen_vects_map[it->first].reserve(9);
					eigen_vects_map[it->first].resize(9);
					memcpy(eigen_vects_map[it->first].data(), ransac.m_eigen_vect, sizeof(float) * 9);

					// ���ƽ�淽��: 4��Ԫ��
					plane_map[it->first].reserve(4);
					plane_map[it->first].resize(4);
					memcpy(plane_map[it->first].data(), ransac.m_plane, sizeof(float) * 4);

					// ��䷨����: ��ƽ�淽�̵�ǰ�����Ƿ�����
					plane_normal_map[it->first].reserve(3);
					plane_normal_map[it->first].resize(3);
					memcpy(plane_normal_map[it->first].data(), plane_map[it->first].data(), sizeof(float) * 3);
				}
				else
				{
					RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
					ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

					// �������ֵeigen_vals_map
					eigen_vals_map[it->first].reserve(3);
					eigen_vals_map[it->first].resize(3);
					memcpy(eigen_vals_map[it->first].data(), ransac.m_eigen_vals, sizeof(float) * 3);

					// �����������: 9��Ԫ��
					eigen_vects_map[it->first].reserve(9);
					eigen_vects_map[it->first].resize(9);
					memcpy(eigen_vects_map[it->first].data(), ransac.m_eigen_vect, sizeof(float) * 9);

					// ���ƽ�淽��: 4��Ԫ��
					plane_map[it->first].reserve(4);
					plane_map[it->first].resize(4);
					memcpy(plane_map[it->first].data(), ransac.m_plane, sizeof(float) * 4);

					// ��䷨����: ��ƽ�淽�̵�ǰ�����Ƿ�����
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
			// ����ÿһ��superpixel
			for (auto it = labels_map.begin();
				it != labels_map.end(); it++)
			{
				if ((int)it->second.size() < 3)
				{
					std::printf("[Warning]: Not enough valid depth within super-pixel %d\n", it->first);
					continue;
				}

				// ȡ�������ֵ��2D��
				const std::vector<cv::Point2f>& Pts2D = has_depth_map[it->first];

				// ----- ����3D�������ϵ�µ�����
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

				// ----- ����ÿ��superpixel������
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

				// ���superpixel��Ӧ3D���Ƶ����ĵ�����
				center_map[it->first] = cv::Point3f(center_x, center_y, center_z);

				// ----- 3D plane fittin using RANSAC
				//if (Pts3D.size() <= 5)
				{
					// using OLS, SVD or PCA
					RansacRunner ransac(0.05f, (int)Pts3D.size(), 3);
					ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

					// �������ֵeigen_vals_map
					eigen_vals_map[it->first].resize(3, 0.0f);
					memcpy(eigen_vals_map[it->first].data(), ransac.m_eigen_vals, sizeof(float) * 3);

					// �����������: 9��Ԫ��
					eigen_vects_map[it->first].resize(9, 0.0f);
					memcpy(eigen_vects_map[it->first].data(), ransac.m_eigen_vect, sizeof(float) * 9);

					// ���ƽ�淽��: 4��Ԫ��
					plane_map[it->first].resize(4, 0.0f);
					memcpy(plane_map[it->first].data(), ransac.m_plane, sizeof(float) * 4);

					// ��䷨����: ��ƽ�淽�̵�ǰ�����Ƿ�����
					plane_normal_map[it->first].reserve(3);
					plane_normal_map[it->first].resize(3);
					memcpy(plane_normal_map[it->first].data(), plane_map[it->first].data(), sizeof(float) * 3);
				}
				//else
				//{
				//	RansacRunner ransac(0.08f, (int)Pts3D.size(), 3);
				//	ransac.RunRansac(Pts3D);  // �����ETH3D: ����~0.0204m(20.4mm)

				//	// �������ֵeigen_vals_map
				//	eigen_vals_map[it->first].resize(3, 0.0f);
				//	memcpy(eigen_vals_map[it->first].data(), ransac.m_eigen_vals, sizeof(float) * 3);

				//	// �����������: 9��Ԫ��
				//	eigen_vects_map[it->first].resize(9, 0.0f);
				//	memcpy(eigen_vects_map[it->first].data(), ransac.m_eigen_vect, sizeof(float) * 9);

				//	// ���ƽ�淽��: 4��Ԫ��
				//	plane_map[it->first].resize(4, 0.0f);
				//	memcpy(plane_map[it->first].data(), ransac.m_plane, sizeof(float) * 4);

				//	// ��䷨����: ��ƽ�淽�̵�ǰ�����Ƿ�����
				//	plane_normal_map[it->first].resize(3, 0.0f);
				//	memcpy(plane_normal_map[it->first].data(), plane_map[it->first].data(), sizeof(float) * 3);
				//}

				//printf("Superpixel %d tagent plane fitted\n", it->first);
			}

			return 0;
		}

		// �����ڽӱ��plane_normal_map��center_map����(�⻬)�����ӵ�superpixel
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
			// ����superpxiel
			for (auto it = label_map.begin(); it != label_map.end(); ++it)
			{
				// ��ȡsuperpixel���ƽ���ά��
				int dim_1 = this->GetPlaneDimForSuperpixel(eigen_vals_map[it->first], THRESH_1);

				// TODO: ����dim < 2��superpixel��չ�������¼�����ƽ��...

				// �ų�"ά��"Ϊ3��
				if (3 == dim_1)
				{
					continue;
				}
				else
				{
					const cv::Point3f& center_1 = center_map[it->first];
					const float* normal_1 = plane_normal_map[it->first].data();

					// ����superpixel�ڽӱ�
					auto NeighborMap = GetNeighborMap(labels);

					// ��������"������"���ڽ�superpixel
					for (int neighbor : NeighborMap[it->first])
					{
						int dim_2 = this->GetPlaneDimForSuperpixel(eigen_vals_map[neighbor], THRESH_1);
						if (3 == dim_2)
						{
							continue;
						}
						else  // ----- ���Ͽ����ӵĵ�һ������
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

							// ȡ�ϴ�ֵ
							float numerator = abs_dot_1 >= abs_dot_2 ? abs_dot_1 : abs_dot_2;

							float connectivity = numerator / abs_normal_dot;
							if (connectivity > THRESH_2)  // ���"���Ӷ�"������ֵ
							{
								continue;
							}

							// ----- ���Ͽ����ӵĵڶ�������, ����������superpixel
							else  // ����������superpixel: neighbor -> it->first 
							{
								// �ϲ�����labels
								for (auto pt : label_map[neighbor])
								{
									labels.at<int>((int)pt.y, (int)pt.x) = it->first;
								}
								// �ϲ�label_map
								for (auto pt : label_map[neighbor])
								{
									label_map[it->first].push_back(pt);
								}
								// �ϲ�����has_depth_map
								for (auto pt : has_depth_map[neighbor])
								{
									has_depth_map[it->first].push_back(pt);
								}
								// �ϲ�has_no_depth_map
								for (auto pt : has_no_depth_map[neighbor])
								{
									has_no_depth_map[it->first].push_back(pt);
								}

								// ----- �������superpixel�����ƽ��
								std::vector<float> center_arr(3, 0.0f);

								// --- ����plane_normal_map��eigen_vals_map
								// ��ȡ�����ֵ��2D��
								const std::vector<cv::Point2f>& Pts2D = has_depth_map[it->first];

								// ���������ƽ��
								this->FitPlaneForSuperpixel(depth_map,
									K_inv_arr, R_inv_arr, T_arr,
									Pts2D,
									plane_normal_map[it->first],
									eigen_vals_map[it->first],
									center_arr);

								// ����center_map
								center_map[it->first].x = center_arr[0];
								center_map[it->first].y = center_arr[1];
								center_map[it->first].z = center_arr[2];

								// ---ɾ��neighbor
								// ɾ��plane_normal_map�е�neighbor
								if (plane_normal_map.find(neighbor) != plane_normal_map.end())
								{
									plane_normal_map.erase(neighbor);
								}

								// ɾ��eigen_vals_map�е�neighbor
								if (eigen_vals_map.find(neighbor) != eigen_vals_map.end())
								{
									eigen_vals_map.erase(neighbor);
								}

								// ɾ��center_map�е�neighbor
								if (center_map.find(neighbor) != center_map.end())
								{
									center_map.erase(neighbor);
								}

								// ɾ��label_map�е�neighbor
								if (label_map.find(neighbor) != label_map.end())
								{
									label_map.erase(neighbor);
								}

								// ɾ��has_depth_map�е�neighbor
								if (has_depth_map.find(neighbor) != has_depth_map.end())
								{
									has_depth_map.erase(neighbor);
								}

								// ɾ��has_no_depth_map�е�neighbor
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
			// ����superpxiel
			for (auto it = label_map.begin(); it != label_map.end(); ++it)
			{
				// ��ȡsuperpixel���ƽ���ά��
				int dim_1 = this->GetPlaneDimForSuperpixel(eigen_vals_map[it->first], THRESH_1);

				// TODO: ����dim < 2��superpixel��չ�������¼�����ƽ��...

				// �ų�"ά��"Ϊ3��
				if (3 == dim_1)
				{
					continue;
				}
				else
				{
					const cv::Point3f& center_1 = center_map[it->first];
					const float* normal_1 = plane_normal_map[it->first].data();

					// ����superpixel�ڽӱ�
					auto NeighborMap = GetNeighborMap(labels);

					// ��������"������"���ڽ�superpixel
					for (int neighbor : NeighborMap[it->first])
					{
						int dim_2 = this->GetPlaneDimForSuperpixel(eigen_vals_map[neighbor],
							THRESH_1);
						if (3 == dim_2)
						{
							continue;
						}
						else  // ----- ���Ͽ����ӵĵ�һ������
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

							// ȡ�ϴ�ֵ
							float numerator = abs_dot_1 >= abs_dot_2 ? abs_dot_1 : abs_dot_2;

							float connectivity = numerator / abs_normal_dot;
							if (connectivity > THRESH_2)  // ���"���Ӷ�"������ֵ
							{
								continue;
							}

							// ----- ���Ͽ����ӵĵڶ�������, ����������superpixel
							else  // ����������superpixel: neighbor -> it->first 
							{
								// �ϲ�����labels
								for (auto pt : label_map[neighbor])
								{
									labels.at<int>((int)pt.y, (int)pt.x) = it->first;
								}
								// �ϲ�label_map
								for (auto pt : label_map[neighbor])
								{
									label_map[it->first].push_back(pt);
								}
								// �ϲ�����has_depth_map
								for (auto pt : has_depth_map[neighbor])
								{
									has_depth_map[it->first].push_back(pt);
								}
								// �ϲ�has_no_depth_map
								for (auto pt : has_no_depth_map[neighbor])
								{
									has_no_depth_map[it->first].push_back(pt);
								}

								// ----- Connect֮��, �������superpixel�����ƽ��
								std::vector<float> center_arr(3, 0.0f);

								// --- ����plane_normal_map��eigen_vals_map
								// ��ȡ�����ֵ��2D��
								const std::vector<cv::Point2f>& Pts2D = has_depth_map[it->first];

								// connect֮�����������ƽ��
								this->FitPlaneForSuperpixelCam(depth_map,
									K_inv_arr,
									Pts2D,
									plane_map[it->first],
									plane_normal_map[it->first],
									eigen_vals_map[it->first],
									center_arr);

								// ����center_map
								center_map[it->first].x = center_arr[0];
								center_map[it->first].y = center_arr[1];
								center_map[it->first].z = center_arr[2];

								// ---ɾ��neighbor
								// ɾ��plane_arr��neighbor
								if (plane_map.find(neighbor) != plane_map.end())
								{
									plane_map.erase(neighbor);
								}

								// ɾ��plane_normal_map�е�neighbor
								if (plane_normal_map.find(neighbor) != plane_normal_map.end())
								{
									plane_normal_map.erase(neighbor);
								}

								// ɾ��eigen_vals_map�е�neighbor
								if (eigen_vals_map.find(neighbor) != eigen_vals_map.end())
								{
									eigen_vals_map.erase(neighbor);
								}

								// ɾ��center_map�е�neighbor
								if (center_map.find(neighbor) != center_map.end())
								{
									center_map.erase(neighbor);
								}

								// ɾ��label_map�е�neighbor
								if (label_map.find(neighbor) != label_map.end())
								{
									label_map.erase(neighbor);
								}

								// ɾ��has_depth_map�е�neighbor
								if (has_depth_map.find(neighbor) != has_depth_map.end())
								{
									has_depth_map.erase(neighbor);
								}

								// ɾ��has_no_depth_map�е�neighbor
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

		// ����filtered depth map�ϲ�superpixel
		int Workspace::MergeSuperpixels(const cv::Mat& src,
			const int MinNum,
			cv::Mat& labels,
			std::unordered_map<int, std::vector<cv::Point2f>>& label_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_depth_map,
			std::unordered_map<int, std::vector<cv::Point2f>>& has_no_depth_map)
		{
			// ����superpixel��ȡ��Ч���ֵ����������ֵ��superpixel
			// hash_mapһ�ߵ���һ��ɾ��
			//for (auto it = has_depth_map.begin(); it != has_depth_map.end(); it++)
			for (auto it = label_map.begin(); it != label_map.end(); it++)
			{
				// ���superpixel������Ч�����ֵ
				if (has_depth_map.find(it->first) != has_depth_map.end())
				{
					while ((int)has_depth_map[it->first].size() < MinNum)
					{
						// ����labels�����ڽӱ�
						std::unordered_map<int, std::set<int>> NeighborMap = GetNeighborMap(labels);

						// �����ڽ�superpixel, �������Ͼ�����С���ڽ�superpixel
						float dist_min = FLT_MAX;  // �����ʼ��Ϊ���ֵ
						int best_neigh = -1;
						for (int neigh : NeighborMap[it->first])  // �����ڽӱ�
						{
							// ȡsuperpixel������ֵ, ������Ͼ���
							float dist = BaDistOf2Superpixel(src, label_map[it->first], label_map[neigh]);
							if (dist < dist_min)
							{
								dist_min = dist;
								best_neigh = neigh;
							}
						}

						// -----�ϲ�it->first��best_neigh
						// ---��best_neigh������ת�Ƶ�it->first
						// �ϲ�labels
						for (auto pt : label_map[best_neigh])
						{
							labels.at<int>((int)pt.y, (int)pt.x) = it->first;
						}
						// �ϲ�label_map
						for (auto pt : label_map[best_neigh])
						{
							label_map[it->first].push_back(pt);
						}
						// �ϲ�has_Depth_map
						for (auto pt : has_depth_map[best_neigh])
						{
							has_depth_map[it->first].push_back(pt);  // ����it->first����Ч��ȵ�
						}
						// �ϲ�has_no_depth_map
						for (auto pt : has_no_depth_map[best_neigh])
						{
							has_no_depth_map[it->first].push_back(pt);
						}

						// ---ɾ��best_neigh
						// ɾ��label_map�е�best_neigh
						if (label_map.find(best_neigh) != label_map.end())
						{
							label_map.erase(best_neigh);
						}
						// ɾ��has_depth_map�е�best_neigh
						if (has_depth_map.find(best_neigh) != has_depth_map.end())
						{
							has_depth_map.erase(best_neigh);
						}
						// ɾ��has_no_depth_map�е�best_neigh
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
						// ����labels�����ڽӱ�
						std::unordered_map<int, std::set<int>> NeighborMap = GetNeighborMap(labels);

						// �����ڽ�superpixel, �������Ͼ�����С���ڽ�superpixel
						float dist_min = FLT_MAX;  // �����ʼ��Ϊ���ֵ
						int best_neigh = -1;
						for (int neigh : NeighborMap[it->first])  // �����ڽӱ�
						{
							// ȡsuperpixel������ֵ, ������Ͼ���
							float dist = BaDistOf2Superpixel(src, label_map[it->first], label_map[neigh]);
							if (dist < dist_min)
							{
								dist_min = dist;
								best_neigh = neigh;
							}
						}

						// -----�ϲ�it->first��best_neigh
						// ---��best_neigh������ת�Ƶ�it->first
						// �ϲ�labels
						for (auto pt : label_map[best_neigh])
						{
							labels.at<int>((int)pt.y, (int)pt.x) = it->first;
						}
						// �ϲ�label_map
						for (auto pt : label_map[best_neigh])
						{
							label_map[it->first].push_back(pt);
						}
						// �ϲ�has_Depth_map
						for (auto pt : has_depth_map[best_neigh])
						{
							has_depth_map[it->first].push_back(pt);  // ����it->first����Ч��ȵ�
						}
						// �ϲ�has_no_depth_map
						for (auto pt : has_no_depth_map[best_neigh])
						{
							has_no_depth_map[it->first].push_back(pt);
						}

						// ---ɾ��best_neigh
						// ɾ��label_map�е�best_neigh
						if (label_map.find(best_neigh) != label_map.end())
						{
							label_map.erase(best_neigh);
						}
						// ɾ��has_depth_map�е�best_neigh
						if (has_depth_map.find(best_neigh) != has_depth_map.end())
						{
							has_depth_map.erase(best_neigh);
						}
						// ɾ��has_no_depth_map�е�best_neigh
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
			// ����labels
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

					// ����������, �ж��Ƿ��Ǳ߽�
					if (label_up != label || label_down != label
						|| label_left != label || label_right != label)
					{
						cv::Vec3b& bgr = Input.at<cv::Vec3b>(y, x);
						bgr[0] = 255;  // ���߽紦����Ϊ��ɫ
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

			// Ϊÿһ������ͳ���������Ƿ���mask border
			// ���������ڽӱ�
			for (int y = 0; y < labels.rows; ++y)
			{
				for (int x = 0; x < labels.cols; ++x)
				{
					const int& center = labels.at<int>(y, x);

					// �м䲿��
					if (y > 0 && x > 0 && y < labels.rows - 1 && x < labels.cols - 1)
					{
						const int& up = labels.at<int>(y - 1, x);
						const int& down = labels.at<int>(y + 1, x);
						const int& left = labels.at<int>(y, x - 1);
						const int& right = labels.at<int>(y, x + 1);

						// ���ݱ�ǩһ�������������ڽӱ�
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
					else if (y == 0 && x == 0)  // ���Ͻ�
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
					else if (y == 0 && x == labels.cols - 1)  // ���Ͻ�
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
					else if (y == 0 && x > 0 && x < labels.cols - 1)  // ��һ���м�
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
					else if (y == labels.rows - 1 && x > 0 && x < labels.cols - 1)  // ���һ���м�
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
					else if (x == 0 && y == labels.rows - 1)  // ���½�
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
					else if (x == 0 && y > 0 && y < labels.rows - 1)  // ��һ���м�
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
					else if (x == labels.cols - 1 && y == labels.rows - 1)  // ���½�
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
					else if (x == labels.cols - 1 && y > 0 && y < labels.rows - 1)  // ���һ���м�
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
			// ͳ������ͨ��BGR��ֱ��ͼ
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

			// ----- ��ȡBGRͨ��intensity
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

			// ---- ����ֱ��ͼ�ĵȲ�����
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

			// ----- ����ֱ��ͼƵ��
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

			// ����ֱ��ͼƵ��
			for (int i = 0; i < num_bins; ++i)
			{
				histgram_b1[i] /= float(superpix1.size());
				histgram_g1[i] /= float(superpix1.size());
				histgram_r1[i] /= float(superpix1.size());

				histgram_b2[i] /= float(superpix1.size());
				histgram_g2[i] /= float(superpix1.size());
				histgram_r2[i] /= float(superpix1.size());
			}

			// ֱ��ͼ��һ��
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

			// ----- ������ͨ�����Ͼ���
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

			// ������ͨ�����Ͼ����ֵ
			return (B_b + B_g + B_r) / 3.0f;
		}

		// �ϲ�����depth, normal maps(src and enhance)�����ҽ��к�������ѡ��������˫�ߴ�����ֵ
		void Workspace::MergeDepthNormalMaps(const bool is_merged, const bool is_sel_JBPF)
		{
			////ָ���������ͼ�ͷ�����ͼ��·��
			// ԭʼͼ��
			const string DepthPath_src = options_.workspace_path + "/SrcMVS/depth_maps/dslr_images_undistorted/";
			const string NormalPath_src = options_.workspace_path + "/SrcMVS/normal_maps/dslr_images_undistorted/";

			// ϸ����ǿͼ��
			//const string DepthPath_detailEnhance = options_.workspace_path + "/detailEnhance/depth_maps/dslr_images_undistorted/";
			//const string NormalPath_detailEnhance = options_.workspace_path + "/detailEnhance/normal_maps/dslr_images_undistorted/";

			// �ṹ��ǿͼ��
			const string DepthPath_structEnhance = options_.workspace_path + "/EnhanceMVS/depth_maps/dslr_images_undistorted/";
			const string NormalPath_structEnhance = options_.workspace_path + "/EnhanceMVS/normal_maps/dslr_images_undistorted/";

			// �ϲ�������Ⱥͷ�����ͼ�Ľ��·��
			const string resultDepthPath = options_.workspace_path + "/result/depth_maps/";
			const string resultNormalPath = options_.workspace_path + "/result/normal_maps/";

			// �Ժϲ������������ѡ��������˫�ߴ�����ֵ���·��
			const string resultProDepthPath = options_.workspace_path + "/resultPro/depth_maps/";
			const string resultProNormalPath = options_.workspace_path + "/resultPro/normal_maps/";

			// ԭʼ��ɫͼ��·��
			const string srcColorImgPath = options_.workspace_path + "/SrcMVS/images/dslr_images_undistorted/";

			clock_t T_start, T_end;

			// ����ÿһ��ͼ
			for (int img_id = 0; img_id < m_model.m_images.size(); img_id++)
			{
				const string DepthAndNormalName = GetFileName(img_id, true);

				// �����û�кϲ�������ô�ϲ�
				if (!is_merged)
				{
					T_start = clock();

					// �ֱ��ȡ���ͼ�ͷ�����ͼ
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

					// @even Fusion���ͼ, ����ͼѡ���߽�С����Ϊ�Լ��Ŀ��
					const int src_width = depthMap_src.GetWidth();
					const int enhance_width = depthMap_structEnhance.GetWidth();
					const int src_height = depthMap_src.GetHeight();
					const int enhance_height = depthMap_structEnhance.GetHeight();

					const int Fusion_Width = std::min(src_width, enhance_width);
					const int Fusion_Height = std::min(src_height, enhance_height);

					//const int width = depthMap_src.GetWidth();
					//const int height = depthMap_src.GetHeight();

					// ��ʼ��Fusion�����ͼ, ����ͼΪ0
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

							// ��ʼ������ֵΪ0
							float normal_src[3],
								//normal_detailEnhance[3],
								normal_structEnhance[3],
								normal_result[3] = { 0.0f };

							normalMap_src.GetSlice(row, col, normal_src);
							//normalMap_detailEnhance.GetSlice(row, col, normal_detailEnhance);
							normalMap_structEnhance.GetSlice(row, col, normal_structEnhance);

							// �ռ����õ���Ⱥͷ�����Ϣ
							vector<float> depths;
							vector<float*> normals;

							// �ռ����õ�src���,����
							if (depth_src != NON_VALUE)
							{
								depths.push_back(depth_src);
								normals.push_back(normal_src);
							}

							// �ռ����õ�enhance���, ����
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
										// �������С����ֵ��ȡ���ֵ��С��
										if (abs(depths[0] - depths[1]) / depths[0] > 0.01)
										{
											depthMap_result.Set(row, col, depths[0] < depths[1]
												? depths[0] : depths[1]);
											normalMap_result.SetSlice(row, col, depths[0] < depths[1]
												? normals[0] : normals[1]);
										}
										else  // ���ȡ��ֵ, �������������L2 norm
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

					// ���ù����ռ��depth, normal mapsΪ�ϲ������˺��ֵ
					m_depth_maps.at(img_id) = depthMap_result;
					m_normal_maps.at(img_id) = normalMap_result;

					// ����"�Ѷ�": ���¼���һ����depth, normal maps�Ķ�ȡ״̬
					hasReadMapsGeom_ = true;

					// ���ϲ����depth, normal mapsд��workspace
					imwrite(resultDepthPath + DepthAndNormalName + ".jpg",
						depthMap_result.ToBitmapGray(2, 98));
					imwrite(resultNormalPath + DepthAndNormalName + ".jpg",
						normalMap_result.ToBitmap());

					depthMap_result.WriteBinary(resultDepthPath + DepthAndNormalName);
					normalMap_result.WriteBinary(resultNormalPath + DepthAndNormalName);

					T_end = clock();
					std::cout << "Merge image:" << img_id << " Time:" << (float)(T_end - T_start) / CLOCKS_PER_SEC << "s" << endl;
				}

				// ��ѡ��������˫�ߴ�����ֵ
				if (is_sel_JBPF)
				{
					T_start = clock();

					// ���֮ǰ�ϲ���Mapͼ�ˣ�ֱ�Ӵ��ļ��ж�ȡ������
					if (is_merged)
					{
						// ��ȡ���ͼ�ͷ�����ͼ
						DepthMap depthMap(depth_ranges_.at(img_id).first,
							depth_ranges_.at(img_id).second);
						depthMap.ReadBinary(resultDepthPath + DepthAndNormalName);

						NormalMap normalMap;
						normalMap.ReadBinary(resultNormalPath + DepthAndNormalName);

						m_depth_maps.at(img_id) = depthMap;
						m_normal_maps.at(img_id) = normalMap;
					}

					// ��������
					DepthMap depthMap_pro = m_depth_maps.at(img_id);
					NormalMap normalMap_pro = m_normal_maps.at(img_id);

					// ��ȡԭ��ɫͼ��resize
					const auto& src_img_path = srcColorImgPath + m_model.GetImageName(img_id);
					cv::Mat src_img = imread(src_img_path);
					resize(src_img,
						src_img,
						Size(m_depth_maps.at(img_id).GetWidth(),
							m_depth_maps.at(img_id).GetHeight()));

					// ѡ��˫�ߴ����˲�
					this->selJointBilateralPropagateFilter(src_img,
						this->m_depth_maps.at(img_id),
						this->m_normal_maps.at(img_id),
						this->m_model.m_images.at(img_id).GetK(),
						25, 10,  // 25, 10
						-1, 16,
						depthMap_pro, normalMap_pro);

					//// ����SelJointBilateralPropagateFilter
					//int sigma_color = 23, sigma_space = 7;
					//for (int iter_i = 0; iter_i < 3; ++iter_i)
					//{
					//	// ѡ��˫�ߴ����˲�
					//	this->selJointBilateralPropagateFilter(src_img,
					//		this->m_depth_maps.at(img_id),
					//		this->m_normal_maps.at(img_id),
					//		model_.m_images.at(img_id).GetK(),
					//		sigma_color, sigma_space,  // 25, 10
					//		-1, 16,
					//		depthMap_pro, normalMap_pro);

					//	// ��̬����
					//	sigma_color += 1;
					//	sigma_space += 1;

					//	// ���ù����ռ��depth, normal mapsΪ�ϲ������˺��ֵ
					//	this->m_depth_maps.at(img_id) = depthMap_pro;
					//	this->m_normal_maps.at(img_id) = normalMap_pro;
					//}

					const int num_iter = 1;  // ��������
					const double sigma_space = 5.0, sigma_color = 5.0, sigma_depth = 5.0;
					const float THRESH = 0.00f, eps = 1.0f, tau = 0.3f;   // ������ 
					const bool is_propagate = false;   // �Ƿ�ʹ�ô������ֵ
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

						//// д���м���..
						//if (iter_i % 10 == 0 || iter_i == NUM_ITER - 1)
						//{
						//	char buff[100];
						//	sprintf(buff, "_iter%d.jpg", iter_i);
						//	imwrite(std::move(resultProDepthPath + DepthAndNormalName + string(buff)),
						//		depthMap_pro.ToBitmapGray(2, 98));
						//}
					}

					//const int NUM_ITER = 1;  // ��������
					//const double sigma_space = 1.5, sigma_color = 0.09;
					//double sigma_depth = 0.02;
					//const float THRESH = 0.06f;   // ������ 
					//const bool is_propagate = false;   // �Ƿ�ʹ�ô������ֵ
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

					//	//// д���м���..
					//	//if (iter_i % 10 == 0 || iter_i == NUM_ITER - 1)
					//	//{
					//	//	char buff[100];
					//	//	sprintf(buff, "_iter%d.jpg", iter_i);
					//	//	imwrite(std::move(resultProDepthPath + DepthAndNormalName + string(buff)),
					//	//		depthMap_pro.ToBitmapGray(2, 98));
					//	//}
					//}

					// ����"�Ѷ�": ���¼���һ����depth, normal maps�Ķ�ȡ״̬
					hasReadMapsGeom_ = true;

					// ��depth, normal mapsת��bitmap��д�����
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

		// ��ѡ���Ե�����˫�ߴ����˲�
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
				radius = round(sigma_space * 1.5);  // original parameters, ���� sigma_space ���� radius  
			}

			//assert(radius % 2 == 1);  // ȷ�����ڳߴ�������
			const int d = 2 * radius + 1;

			// ԭ����ͼ���ͨ����
			const int channels = joint.channels();

			//float *color_weight = new float[cnj * 256];
			//float *space_weight = new float[d*d];
			//int *space_ofs_row = new int[d*d];  // ����Ĳ�ֵ
			//int *space_ofs_col = new int[d*d];

			vector<float> color_weight(channels * 256);
			vector<float> space_weight(d * d);
			vector<int> space_offsets_row(d * d), space_offsets_col(d * d);

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
			// initialize color-related bilateral filter coefficients

			// ɫ��ĸ�˹Ȩ��  
			for (int i = 0; i < 256 * channels; i++)
			{
				color_weight[i] = std::expf(i * i * gauss_color_coeff);
			}

			int MAX_K = 0;   // 0 ~ (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			//�ռ��ĸ�˹Ȩ��
			// ͳ������������������:���㷽�ε��������Բ������
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
					// ���������ֵ(���ֵ����)������
					if (depthMap.GetDepth(y, x) != 0.0f)
					{
						continue;
					}

					// bgr
					const cv::Vec3b color_0 = joint.ptr<cv::Vec3b>(y)[x];

					// ����Ȩ�غ�����λ�õ�����
					vector<pair<float, int>> weightAndIndex;
					weightAndIndex.clear();
					for (int k = 0; k < MAX_K; k++)
					{
						const int yy = y + space_offsets_row[k];
						const int xx = x + space_offsets_col[k];

						// �ж�q, ��ҪqҲ�����ֵ
						if (yy < 0 || yy >= MapHeight || xx < 0
							|| xx >= MapWidth || depthMap.GetDepth(yy, xx) == 0.0f)
						{
							continue;
						}

						//��ɫ����Ȩ�أ��������ڸ߷ֱ���ͼ���ϵ�
						cv::Vec3b color_1 = joint.ptr<cv::Vec3b>(yy)[xx];

						// ����joint��ǰ���غ��������ص� ����Ȩ�� �� ɫ��Ȩ�أ������ۺϵ�Ȩ��
						const float& the_color_weight = color_weight[abs(color_0[0] - color_1[0]) +
							abs(color_0[1] - color_1[1]) + abs(color_0[2] - color_1[2])];
						float w = space_weight[k] * the_color_weight;

						//ֻ����space������ΪȨ��!!!!!!
						//float w = space_weight[k];

						weightAndIndex.push_back(make_pair(w, k));
					}

					// ���Ȩ��ֵΪ��
					if (weightAndIndex.size() == 0)
					{
						continue;
					}
					//if (weightAndIndex.size() < int(0.1f * (float)space_offsets_row.size()))
					//{
					//	continue;
					//}

					//�Դ洢��Ȩ�ؽ��дӴ�С����
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

					// ���մӴ�С��Ȩ�أ�������ȴ���
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

						/****************���ֵ��������****************/

						// ������ȴ���ֵ
						float propagated_depth = PropagateDepth(refK,
							src_depth, src_normal,
							yy, xx, y, x);

						// ��������ֱ����ԭ���ֵ
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

					// �������ֵ
					const float out_depth = sum_value_depth * sum_w;

					//// @even DEBUG: to check for Nan dpeth
					//if (isnan(out_depth))
					//{
					//	cout << "\n[Nan out depth]: " << out_depth << endl;
					//}

					outDepthMap.Set(y, x, out_depth);

					// ���÷���ֵ
					sum_value_normal[0] *= sum_w;
					sum_value_normal[1] *= sum_w;
					sum_value_normal[2] *= sum_w;

					// ��������
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

			// original parameters, ���� sigma_space ���� radius 
			if (radius <= 0)
			{
				radius = (int)round(sigma_space * 1.5 + 0.5);
			}

			//assert(radius % 2 == 1);  // ȷ�����ڳߴ�������
			const int d = 2 * radius + 1;

			// ԭ����ͼ���ͨ����
			const int channels = joint.channels();
			const int& color_levels = 256 * channels;

			// ------------ RGBԭͼɫ��, �ռ�����˹Ȩ��
			vector<float> color_weights(color_levels);
			vector<float> space_weights(d * d);
			vector<int> space_offsets_row(d * d), space_offsets_col(d * d);

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
			// initialize color-related bilateral filter coefficients

			// ɫ��ĸ�˹Ȩ��  
			for (int i = 0; i < color_levels; ++i)
			{
				color_weights[i] = (float)std::exp(i * i * gauss_color_coeff);
			}

			int MAX_K = 0;   // 0 ~ (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			// �ռ��ĸ�˹Ȩ��
			// ͳ��������������������������������Բ����
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

			//// ����ԭʼ���ͼ��˹ƽ����� 
			//cv::Mat depth_mat, depth_blur;
			//depth_mat = depthMap.Depth2Mat();
			//cv::GaussianBlur(depth_mat, depth_blur, cv::Size(3, 3), 0);

			// ����ÿһ������
			//printf("eps: %.3f, tau: %.3f\n", eps, tau);
			for (int y = 0; y < MapHeight; y++)
			{
				for (int x = 0; x < MapWidth; x++)
				{
					// ���������ֵ(���ֵ����)������
					if (depthMap.GetDepth(y, x) != 0.0f)
					{
						continue;
					}

					//// ����뾶�����ڵ�omega_depth
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

					//// �����������С,������ֵ,û��Ҫ�������ļ���,����������
					//if (depth_min == DBL_MAX || depth_max == -1.0f)
					//{
					//	continue;
					//}

					//const double omega_depth = depth_max - depth_min;

					// p����bgr��ɫֵ
					const cv::Vec3b& color_0 = joint.ptr<cv::Vec3b>(y)[x];

					// p���ص����ֵ
					const double& depth_0 = (double)depthMap.GetDepth(y, x);

					// ͳ��pΪ���ĵ�Բ�δ���, ��Ч��Ȩ�ؼ�������λ�õ�����
					vector<pair<float, int>> WeightAndIndex;
					WeightAndIndex.clear();
					for (int k = 0; k < MAX_K; ++k)
					{
						const int yy = y + space_offsets_row[k];
						const int xx = x + space_offsets_col[k];

						// �ж�q, ��ҪqҲ�����ֵ
						if (yy < 0 || yy >= MapHeight || xx < 0
							|| xx >= MapWidth || depthMap.GetDepth(yy, xx) == 0.0f)
						{
							// ����û�����ֵ��neighbor
							continue;
						}

						// q����bgr��ɫֵ
						cv::Vec3b color_1 = joint.ptr<cv::Vec3b>(yy)[xx];

						// q���ص����ֵ
						const double depth_1 = (double)depthMap.GetDepth(yy, xx);

						// ����ԭʼ���ͼ��Ȳ�ֵ�ĸ�˹����ֵ
						double delta_depth = depth_0 - depth_1;
						const double depth_weight = std::exp(-0.5 * delta_depth * delta_depth
							/ sigma_depth);

						// ����joint��ǰ���غ��������صľ���Ȩ�غ�ɫ��Ȩ�أ������ۺϵ�Ȩ��
						const int delta_color = abs(color_0[0] - color_1[0]) +
							abs(color_0[1] - color_1[1]) + abs(color_0[2] - color_1[2]);
						const float color_weight = color_weights[delta_color];

						// ����Alpha
						//double alpha = depthMap.CalculateAlpha(eps, tau, omega_depth);

						// ���Ǹ���color_weight��depth_weight�����ƶ�ȷ��Alphaֵ....
						const double delta_color_ratio = double(delta_color) / double(color_levels);
						const double delta_depth_ratio = std::abs(delta_depth) / double(depthMap.depth_max_);
						double diff = std::abs(delta_color_ratio - delta_depth_ratio);
						double alpha = std::exp(-0.5 * diff * diff / 0.2);  // to reduce sigma_alpha: 0.2

						const float compound_weight = float(alpha * color_weight + \
							(1.0f - alpha) * depth_weight);

						float weight = space_weights[k] * compound_weight;
						WeightAndIndex.push_back(make_pair(weight, k));
					}

					// ��WeightAndIndex��Size��С���й���
					//if (WeightAndIndex.size() == 0)
					//{
					//	continue;
					//}
					if (WeightAndIndex.size() < size_t(THRESH * (float)space_offsets_row.size()))
					{
						continue;
					}

					// �����Ȩ���ֵ�ͷ�����
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

						/****************���ֵ��������****************/
						float depth_val = 0.0f;
						if (is_propagate)
						{
							// ������ȴ���ֵ
							depth_val = PropagateDepth(refK,
								src_depth, src_normal,
								yy, xx, y, x);
						}
						else
						{
							//��������ֱ����ԭ���ֵ
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

					// �������ֵ
					const float out_depth = sum_value_depth * sum_w;
					outDepthMap.Set(y, x, out_depth);

					// ���÷���ֵ
					sum_value_normal[0] *= sum_w;
					sum_value_normal[1] *= sum_w;
					sum_value_normal[2] *= sum_w;

					// ��������
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

			// original parameters, ���� sigma_space ���� radius 
			if (radius <= 0)
			{
				radius = (int)round(sigma_space * 1.5 + 0.5);
			}

			//assert(radius % 2 == 1);  // ȷ�����ڳߴ�������
			const int d = 2 * radius + 1;

			// ԭ����ͼ���ͨ����
			const int channels = joint.channels();
			const int& color_levels = 256 * channels;

			// ------------ RGBԭͼɫ��, �ռ�����˹Ȩ��
			vector<float> color_weights(color_levels);
			vector<float> space_weights(d * d);
			vector<int> space_offsets_row(d * d), space_offsets_col(d * d);

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
			// initialize color-related bilateral filter coefficients

			//// ɫ��ĸ�˹Ȩ��  
			//for (int i = 0; i < color_levels; ++i)
			//{
			//	color_weights[i] = (float)std::exp(i * i * gauss_color_coeff);
			//}

			int MAX_K = 0;   // 0 ~ (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			// �ռ��ĸ�˹Ȩ��
			// ͳ��������������������������������Բ����
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

			// ����ÿһ������
			//printf("eps: %.3f, tau: %.3f\n", eps, tau);
			for (int y = 0; y < MapHeight; y++)
			{
				for (int x = 0; x < MapWidth; x++)
				{
					// ���������ֵ(���ֵ����)������
					if (depthMap.GetDepth(y, x) != 0.0f)
					{
						continue;
					}

					// p����bgr��ɫֵ
					const cv::Vec3b& color_0 = joint.ptr<cv::Vec3b>(y)[x];

					// p���ص����ֵ
					const double depth_0 = (double)depthMap.GetDepth(y, x);

					// ͳ��pΪ���ĵ�Բ�δ���, ��Ч��Ȩ�ؼ�������λ�õ�����
					vector<pair<float, int>> WeightAndIndex;
					WeightAndIndex.clear();
					for (int k = 0; k < MAX_K; ++k)
					{
						const int yy = y + space_offsets_row[k];
						const int xx = x + space_offsets_col[k];

						// �ж�q, ��ҪqҲ�����ֵ
						if (yy < 0 || yy >= MapHeight || xx < 0
							|| xx >= MapWidth || depthMap.GetDepth(yy, xx) == 0.0f)
						{
							// ����û�����ֵ��neighbor
							continue;
						}

						// q����bgr��ɫֵ
						cv::Vec3b color_1 = joint.ptr<cv::Vec3b>(yy)[xx];

						// q���ص����ֵ
						const double depth_1 = (double)depthMap.GetDepth(yy, xx);

						// ����ԭʼ���ͼ��Ȳ�ֵ�ĸ�˹����ֵ
						double delta_depth = depth_0 - depth_1;
						delta_depth /= depthMap.depth_max_;
						if (float(color_0[0] + color_0[1] + color_0[2])
							/ float(color_levels) < 0.16f)  // threshold of sigma_depth
						{
							sigma_depth = 0.06;
						}
						const double depth_weight = std::exp(-0.5 * delta_depth * delta_depth
							/ sigma_depth);

						// ����ɫ��Ȩ��
						double delta_color = abs(color_0[0] - color_1[0]) +
							abs(color_0[1] - color_1[1]) + abs(color_0[2] - color_1[2]);
						delta_color /= double(color_levels);
						const double color_weight = std::exp(-0.5 * delta_color * delta_color
							/ sigma_color);
						//const float color_weight = color_weights[delta_color];

						// �������Ȩ��
						//float& space_weight = space_weights[k];
						double delta_space = sqrt((x - xx) * (x - xx) + (y - yy) * (y - yy));
						//delta_space /= double(radius);
						const double space_weight = std::exp(-0.5 * delta_space * delta_space
							/ sigma_space);

						// �����ۺ�Ȩ��
						const float weight = space_weight * color_weight * depth_weight;
						WeightAndIndex.push_back(make_pair(weight, k));
					}

					// ��weightAndIndex��Size��С���й���
					//if (WeightAndIndex.size() == 0)
					//{
					//	continue;
					//}
					if (WeightAndIndex.size() < size_t(THRESH * (float)space_offsets_row.size()))
					{
						continue;
					}

					// �����Ȩ���ֵ�ͷ�����
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

						/****************���ֵ��������****************/
						float depth_val = 0.0f;
						if (is_propagate)
						{
							// ������ȴ���ֵ
							depth_val = PropagateDepth(refK,
								src_depth, src_normal,
								yy, xx, y, x);
						}
						else
						{
							//��������ֱ����ԭ���ֵ
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

					// �������ֵ
					const float out_depth = sum_value_depth * sum_w;
					outDepthMap.Set(y, x, out_depth);

					// ���÷���ֵ
					sum_value_normal[0] *= sum_w;
					sum_value_normal[1] *= sum_w;
					sum_value_normal[2] *= sum_w;

					// ��������
					SuitNormal(y, x, refK, sum_value_normal);
					outNormalMap.SetSlice(y, x, sum_value_normal);
				}
			}
		}

		//�����ͼ�ͷ�����mapͼ��������˫���˲�
		void Workspace::jointBilateralFilter_depth_normal_maps(const cv::Mat& joint,
			const DepthMap& depthMap, const NormalMap& normalMap,
			const float *refK, const double sigma_color, const double sigma_space, int radius,
			DepthMap& outDepthMap, NormalMap& outNormalMap) const
		{
			const int mapWidth = depthMap.GetWidth();
			const int mapHeight = depthMap.GetHeight();

			if (radius <= 0)
				radius = round(sigma_space * 1.5);  // ���� sigma_space ���� radius  

			//assert(radius % 2 == 1);//ȷ�����ڰ뾶������
			const int d = 2 * radius + 1;

			//ԭ����ͼ���ͨ����
			const int cnj = joint.channels();
			vector<float> color_weight(cnj * 256);
			vector<float> space_weight(d*d);
			vector<int> space_ofs_row(d*d), space_ofs_col(d*d);

			double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

			// initialize color-related bilateral filter coefficients  
			// ɫ��ĸ�˹Ȩ��  
			for (int i = 0; i < 256 * cnj; i++)
				color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

			int maxk = 0;   // 0 - (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			//�ռ��ĸ�˹Ȩ��
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
					if (depthMap.GetDepth(r, l) != 0.0f)//�������ȵ��ˣ�������
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

						//��ɫ����Ȩ�أ��������ڸ߷ֱ���ͼ���ϵ�
						cv::Vec3b color1 = joint.ptr<cv::Vec3b>(rr)[ll];

						//// ����joint��ǰ���غ��������ص� ����Ȩ�� �� ɫ��Ȩ�أ������ۺϵ�Ȩ��  
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

		//ֻ���þ���Ȩ�ؽ����˲���ֵ
		void Workspace::distanceWeightFilter(const DepthMap &depthMap, const NormalMap &normalMap,
			const float *refK, const double sigma_color, const double sigma_space, int radius,
			DepthMap &outDepthMap, NormalMap &outNormalMap) const
		{
			const int mapWidth = depthMap.GetWidth();
			const int mapHeight = depthMap.GetHeight();

			if (radius <= 0)
				radius = round(sigma_space * 1.5);  // ���� sigma_space ���� radius  

			//assert(radius % 2 == 1);//ȷ�����ڰ뾶������
			const int d = 2 * radius + 1;

			//ԭ����ͼ���ͨ����
			vector<float> space_weight(d*d);
			vector<int> space_ofs_row(d*d), space_ofs_col(d*d);
			double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

			int maxk = 0;   // 0 - (2*radius + 1)^2  

			// initialize space-related bilateral filter coefficients  
			//�ռ��ĸ�˹Ȩ��
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
					if (depthMap.GetDepth(r, l) != 0.0f)  // �������ȵ��ˣ�������
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

						// ����Ȩ��   
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


					// @even: ����lapulace��Ե, �������ͼ������˫���˲�
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

					// @even: ��selJB֮������ͼ��������˫���˲�
					//cv::Mat dst, mat = depthMap_pro.Depth2Mat();
					//src_img.convertTo(src_img, CV_32FC1);
					//int k = 10;
					//cv::ximgproc::jointBilateralFilter(src_img, mat, dst,
					//-1, 2 * k - 1, 2 * k - 1);
					//depthMap_pro.fillDepthWithMat(dst);

					// @even: ��selJB֮������ͼ���������˲�
					//cv::Mat dst, mat = depthMap_pro.Depth2Mat();
					//double eps = 1e-6;
					//eps *= 255.0 * 255.0;
					//dst = guidedFilter(src_img, mat, 10, eps);
					//depthMap_pro.fillDepthWithMat(dst);

						// ����˫���˲�
						//jointBilateralFilter_depth_normal_maps(srcImage, depthMaps_.at(image_id), normalMaps_.at(image_id),
						//	model_.images.at(image_id).GetK(), 25, 10, -1, depthMap_pro, normalMap_pro);

						// ֻ���ÿռ����Ȩ�ز�ֵ
						//distanceWeightFilter(depthMaps_.at(image_id), normalMaps_.at(image_id),
						//	model_.images.at(image_id).GetK(), 25, 10, -1, depthMap_pro, normalMap_pro);