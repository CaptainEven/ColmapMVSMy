#include "MyPatchMatch.h"


#include <Eigen/core>

namespace colmap {
	namespace mvs {

		// The return values is 1 - NCC, so the range is [0, 2], the smaller the
		// value, the better the color consistency.
		struct PhotoConsistencyCostComputer {
			//窗口大小
			int kWindowSize = 10;
			// Parameters for bilateral weighting.
			float sigma_spatial = 3.0f;
			float sigma_color = 0.3f;

			//reference and src image
			const Image *srcImage = nullptr;
			const Image *refImage = nullptr;

			//参考和原图像灰度图像
			const cv::Mat *srccvImage = nullptr;
			const cv::Mat *refcvImage = nullptr;

			// Precomputed sum of raw and squared image intensities.
			float local_ref_sum = 0.0f;
			float local_ref_squared_sum = 0.0f;

			// Center position of patch in reference image.
			int row = -1;
			int col = -1;

			// Depth and normal for which to warp patch.
			float depth = 0.0f;
			const float* normal = nullptr;

			// Dimensions of reference image.
			int ref_image_width = 0;
			int ref_image_height = 0;


			//float ComputeBilateralWeight1(const float row1, const float col1, const float row2, const float col2,
			//	const float color1, const float color2, const float sigma_spatial, const float sigma_color) const
			//{
			//	const float row_diff = row1 - row2;
			//	const float col_diff = col1 - col2;
			//	const float spatial_dist = sqrt(row_diff * row_diff + col_diff * col_diff);
			//	const float color_dist = abs(color1 - color2);
			//	//const double dist = -spatial_dist / (2.0f * sigma_spatial * sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color);
			//	//const double result = exp(dist);
			//	//return result;
			//	return exp(-spatial_dist / (2.0f * sigma_spatial * sigma_spatial) -
			//		color_dist / (2.0f * sigma_color * sigma_color));
			//}

			inline float Compute() const {
				const float kMaxCost = 2.0f;
				const int kWindowRadius = kWindowSize / 2;

				const int row_start = row - kWindowRadius;
				const int col_start = col - kWindowRadius;
				const int row_end = row + kWindowRadius;
				const int col_end = col + kWindowRadius;

				if (row_start < 0 || col_start < 0 || row_end >= ref_image_height ||
					col_end >= ref_image_width) {
					return kMaxCost;
				}

				float tform[9];
				ComposeHomography(refImage, srcImage, row, col, depth, normal, tform);

				float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
				float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
				float z = tform[6] * col_start + tform[7] * row_start + tform[8];
				float base_col_src = col_src;
				float base_row_src = row_src;
				float base_z = z;

				const float center_ref = refcvImage->at<uchar>(row, col) / 255.0f;


				//cv::Mat	srcImageDebug = cv::imread(srcImage->GetPath());
				//cv::Mat	refImageDebug = cv::imread(refImage->GetPath());


				const float sum_ref = local_ref_sum;
				const float sum_ref_ref = local_ref_squared_sum;
				float sum_src = 0.0f;
				float sum_src_src = 0.0f;
				float sum_ref_src = 0.0f;
				float bilateral_weight_sum = 0.0f;

				for (int row = 0; row < kWindowSize; ++row) {
					// Accumulate values per row to reduce numerical errors.
					float sum_src_row = 0.0f;
					float sum_src_src_row = 0.0f;
					float sum_ref_src_row = 0.0f;
					float bilateral_weight_sum_row = 0.0f;

					for (int col = 0; col < kWindowSize; ++col) {
						const float inv_z = 1.0f / z;
						const float norm_col_src = inv_z * col_src + 0.5f;
						const float norm_row_src = inv_z * row_src + 0.5f;
						const float ref = refcvImage->at<uchar>(row_start + row, col_start + col) / 255.0f;
						const float src = (norm_col_src >= 0 && norm_col_src < srcImage->GetWidth() && norm_row_src >= 0 && norm_row_src < srcImage->GetHeight()) ?
							srccvImage->at<uchar>(norm_row_src, norm_col_src) / 255.0f : 0.0f;

						//if (row == kWindowRadius && col == kWindowRadius)
						//{
						//	refImageDebug.at<cv::Vec3b>(row_start + row, col_start + col) = cv::Vec3b(0, 0, 255);
						//	srcImageDebug.at<cv::Vec3b>(norm_row_src, norm_col_src) = cv::Vec3b(0, 0, 255);
						//}
						//else
						//{
						//	refImageDebug.at<cv::Vec3b>(row_start + row, col_start + col) = cv::Vec3b(255, 0, 0);
						//	if (norm_col_src >= 0 && norm_col_src < srcImage->GetWidth() && norm_row_src >= 0 && norm_row_src < srcImage->GetHeight())
						//		srcImageDebug.at<cv::Vec3b>(norm_row_src, norm_col_src) = cv::Vec3b(255, 0, 0);
						//}


						const float bilateral_weight =
							ComputeBilateralWeight(kWindowRadius, kWindowRadius, row, col,
								center_ref, ref, sigma_spatial, sigma_color);

						sum_src_row += bilateral_weight * src;
						sum_src_src_row += bilateral_weight * src * src;
						sum_ref_src_row += bilateral_weight * ref * src;
						bilateral_weight_sum_row += bilateral_weight;

						col_src += tform[0];
						row_src += tform[3];
						z += tform[6];
					}

					sum_src += sum_src_row;
					sum_src_src += sum_src_src_row;
					sum_ref_src += sum_ref_src_row;
					bilateral_weight_sum += bilateral_weight_sum_row;

					base_col_src += tform[1];
					base_row_src += tform[4];
					base_z += tform[7];

					col_src = base_col_src;
					row_src = base_row_src;
					z = base_z;
				}


				//cv::imwrite("./refImage.jpg", refImageDebug);
				//cv::imwrite("./srcImage.jpg", srcImageDebug);


				const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
				sum_src *= inv_bilateral_weight_sum;
				sum_src_src *= inv_bilateral_weight_sum;
				sum_ref_src *= inv_bilateral_weight_sum;

				const float var_ref = sum_ref_ref - sum_ref * sum_ref;//ref方差
				const float var_src = sum_src_src - sum_src * sum_src;//src方差

				// Based on Jensen's Inequality for convex functions, the variance
				// should always be larger than 0. Do not make this threshold smaller.
				const float kMinVar = 1e-5f;
				if (var_ref < kMinVar || var_src < kMinVar) {
					return kMaxCost;
				}
				else {
					const float covar_src_ref = sum_ref_src - sum_ref * sum_src;//ref和src的协方差
					const float var_ref_src = sqrt(var_ref * var_src);//ref和src方差乘积的平方
					return max(0.0f, min(kMaxCost, 1.0f - covar_src_ref / var_ref_src));
				}
			}
		};

		//计算问题
		void MyPatchMatch::getProblems()
		{
			problems_.clear();

			const auto model = workspace_->GetModel();

			std::cout << "Reading configuration..." << std::endl << std::endl;

			std::vector<std::map<int, int>> shared_num_points;
			std::vector<std::map<int, float>> triangulation_angles;

			const float min_triangulation_angle_rad =
				DegToRad(pmOptions_.min_triangulation_angle);

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
					const float kTriangulationAnglePercentile = 75;
					triangulation_angles =
						model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
				}

				//const size_t max_num_src_images =
				//    boost::lexical_cast<int>(src_image_names[1]);
				const size_t max_num_src_images = 20;  // 默认情况下，最多20个原图像

				const auto& overlapping_images =
					shared_num_points.at(problem.ref_img_id);
				const auto& overlapping_triangulation_angles =
					triangulation_angles.at(problem.ref_img_id);

				if (max_num_src_images >= overlapping_images.size()) {
					problem.src_img_ids.reserve(overlapping_images.size());
					for (const auto& image : overlapping_images) {
						if (overlapping_triangulation_angles.at(image.first) >=
							min_triangulation_angle_rad) {
							problem.src_img_ids.push_back(image.first);
						}
					}
				}
				else {
					std::vector<std::pair<int, int>> src_images;
					src_images.reserve(overlapping_images.size());
					for (const auto& image : overlapping_images) {
						if (overlapping_triangulation_angles.at(image.first) >=
							min_triangulation_angle_rad) {
							src_images.emplace_back(image.first, image.second);
						}
					}

					const size_t eff_max_num_src_images =
						std::min(src_images.size(), max_num_src_images);

					std::partial_sort(src_images.begin(),
						src_images.begin() + eff_max_num_src_images,
						src_images.end(),
						[](const std::pair<int, int> image1,
							const std::pair<int, int> image2) {
						return image1.second > image2.second;
					});

					problem.src_img_ids.reserve(eff_max_num_src_images);
					for (size_t i = 0; i < eff_max_num_src_images; ++i) {
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


		//计算所有图像各自的patch值
		void MyPatchMatch::computeAllSelfPatch()
		{
			std::cout << "Computing images local sum/squared sum..." << std::endl << std::endl;

			const auto model = workspace_->GetModel();
			const int numImages = model.m_images.size();
			image_sums_.resize(numImages);
			image_squared_sums_.resize(numImages);

			int index = -1;
			for (const auto &image : model.m_images)
			{
				index++;

				int imageWid = image.GetWidth();
				int imageHei = image.GetHeight();

				//把cv::Mat形式的图像转化为vector形式
				const cv::Mat &image_mat = workspace_->GetBitmap(index);
				std::vector<uint8_t> ref_image_array(imageHei*imageWid);
				ref_image_array.assign(image_mat.datastart, image_mat.dataend);

				//计算patch的局部和，局部平方和
				ref_image_.reset(new GpuMatRefImage(imageWid, imageHei));
				ref_image_->Filter(ref_image_array.data(), pmOptions_.window_radius,
					pmOptions_.sigma_spatial, pmOptions_.sigma_color);

				//把计算出来的patch局部和，局部平方和拷贝出来
				image_sums_[index] = ref_image_->sum_image->CopyToMat();
				image_squared_sums_[index] = ref_image_->squared_sum_image->CopyToMat();
			}
		}


		//初始化稀疏点云，计算每个三维点的法向量
		void MyPatchMatch::initSparsePoints()
		{

			std::cout << "Initial Sparse Points..." << std::endl << std::endl;
			const auto &model = workspace_->GetModel();

			sparse_normals_.reserve(model.m_points.size());//稀疏点云法向量
			std::vector<int> itNum;//标注稀疏点云是否可用
			int numErrors = 0;//可用洗漱点云数量

			PhotoConsistencyCostComputer pcc;
			pcc.kWindowSize = 2 * pmOptions_.window_radius;
			pcc.sigma_color = (float)pmOptions_.sigma_color;
			pcc.sigma_spatial = (float)pmOptions_.sigma_spatial;

			//处理每个洗漱点云，计算法向量
			for (const auto &point : model.m_points)
			{
				const int trackSize = point.track.size();//轨道数量	
				const Eigen::Vector4f pt(point.x, point.y, point.z, 1.0f);//三维点坐标

				////开始计算track中视图关于三维点的中间视图
				int id1, id2;//最大夹角的两个图像索引
				float angle_bisect[3];//最大角平分线向量
				int refImageId;//参考视图索引
				//寻找最大夹角，也即最小的cos值
				float min_cos_trang_angle = 1;
				for (int i = 0; i < trackSize - 1; i++)
				{
					const int imageId_i = point.track[i];//i视图索引
					const float *center_i = model.m_images.at(imageId_i).GetCenter();//i摄像机中心
					float view_i[3] = { center_i[0] - point.x, center_i[1] - point.y, center_i[2] - point.z };
					NormVec3(view_i);//归一化向量
					for (int j = i + 1; j < trackSize; j++)
					{
						const int imageId_j = point.track[j];//j视图索引
						const float *center_j = model.m_images.at(imageId_j).GetCenter();//j摄像机中心
						float view_j[3] = { center_j[0] - point.x, center_j[1] - point.y, center_j[2] - point.z };
						NormVec3(view_j);
						const float cosAngle = DotProduct3(view_i, view_j);
						//const float angle = RadToDeg( acos(cosAngle) );//调试用，看角度是多少,弧度转角度
						if (cosAngle < min_cos_trang_angle)
						{
							min_cos_trang_angle = cosAngle;
							angle_bisect[0] = view_i[0] + view_j[0];//角平分线向量
							angle_bisect[1] = view_i[1] + view_j[1];
							angle_bisect[2] = view_i[2] + view_j[2];
							id1 = imageId_i;
							id2 = imageId_j;
						}
					}
				}

				//寻找离角平分线最近的一幅视图作为参考视图
				float max_cos_angle = -1;
				std::vector<cv::Point3f> views;//记录相机中心到三维点向量
				for (int i = 0; i < trackSize; i++)
				{
					const int imageId_i = point.track[i];//i视图索引
					const float *center_i = model.m_images.at(imageId_i).GetCenter();//i摄像机中心
					const float view_i[3] = { center_i[0] - point.x, center_i[1] - point.y, center_i[2] - point.z };
					const float cosAngle = DotProduct3(view_i, angle_bisect)
						/ sqrt(DotProduct3(view_i, view_i)*DotProduct3(angle_bisect, angle_bisect));
					//const float angle = RadToDeg(acos(cosAngle));
					if (cosAngle > max_cos_angle)
					{
						max_cos_angle = cosAngle;
						refImageId = imageId_i;
					}

					views.push_back(cv::Point3f(view_i[0], view_i[1], view_i[2]));
				}


				////读取参考图像信息和三维点深度
				const auto &refImage = model.m_images[refImageId];//图像参数
				const cv::Mat &refcvImage = workspace_->GetBitmap(refImageId);//参考图像灰度图像

				const Eigen::Vector3f xyz = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(refImage.GetP())*pt;
				//const cv::Point3f cvImgPoint(xyz(0), xyz(1), xyz(2));//齐次坐标
				const int row = xyz(1) / xyz(2) + 0.5f;
				const int col = xyz(0) / xyz(2) + 0.5f;

				//调试用信息，用以检测投影点是否和bundler中数据一样
				//const cv::Point2f temp(xyz(0) / xyz(2) - refImage.GetWidth() / 2.0f, refImage.GetHeight() / 2.0f - xyz(1) / xyz(2));

				pcc.row = row;
				pcc.col = col;
				pcc.depth = xyz(2);
				pcc.local_ref_sum = image_sums_[refImageId].Get(row, col, 0);
				pcc.local_ref_squared_sum = image_squared_sums_[refImageId].Get(row, col, 0);
				pcc.refImage = &refImage;
				pcc.refcvImage = &refcvImage;
				pcc.ref_image_height = refImage.GetHeight();
				pcc.ref_image_width = refImage.GetWidth();

				////角平分线向量作为先前，然后摄像机到三维点摄像作为扰乱和随机
				float prepNormal[3], perturbNormal[3], RandNormal[3], bestNormal[3];//先前，扰乱,随机
				const float perturbtion = 0.05f;  // 扰乱9度
				const float view1[3] = { views.at(0).x, views.at(0).y, views.at(0).z };//views中第一个
				const float view2[3] = { views.at(1).x, views.at(1).y, views.at(1).z };//views中第二个
				PerturbNormal(row, col, refImage.GetK(), 0.0f, angle_bisect, prepNormal);
				PerturbNormal(row, col, refImage.GetK(), 0.0f, view1, perturbNormal);
				PerturbNormal(row, col, refImage.GetK(), 0.0f, view2, RandNormal);


				//每次测试3个法向
				const int numNorm = 3;
				float *normals[numNorm] = { prepNormal, perturbNormal,RandNormal };//先前,扰乱，随机		

				float bestCost = FLT_MAX;//上一次计算出来最好（小）的cost				
				int it = 0;//迭代次数
				//while (bestCost > options_.sparse_max_patch_cost*(trackSize-1))
				while (1)
				{
					it++;
					float costs[numNorm] = { 0.0 };
					for (int sample = 0; sample < numNorm; sample++)
					{
						//如果不是第一次计算，就跳过上一次最优(先前)的结果
						if (it > 1 && sample == 0)
						{
							costs[sample] = bestCost;
							continue;
						}

						pcc.normal = normals[sample];
						for (size_t i = 0; i < trackSize; i++)
						{
							//如果是参考图像，就跳过
							const int srcImageId = point.track[i];
							if (srcImageId == refImageId)
								continue;

							const auto &srcImage = model.m_images[srcImageId];
							const cv::Mat &srccvImage = workspace_->GetBitmap(srcImageId);
							pcc.srcImage = &srcImage;
							pcc.srccvImage = &srccvImage;

							costs[sample] += pcc.Compute();
						}
					}

					//提取最好法向和cost值
					const int minCostId = FindMinCost(costs, numNorm);
					bestCost = costs[minCostId];//最小的cost
					for (int l = 0; l < 3; l++)
					{
						bestNormal[l] = normals[minCostId][l];
						prepNormal[l] = bestNormal[l];//下一次迭代的先前法向，为本次的最优法向
					}


					//达到迭代次数，并且最终cost值小于阈值，成功跳出
					if (it >= options_.sparse_max_patch_iterator
						&& bestCost <= options_.sparse_max_patch_cost*(trackSize - 1))
					{
						break;
					}
					//达到迭代次数,但是最优cost值大于阈值，寻找法向量失败，那么不利用这个稀疏点
					else if (it >= options_.sparse_max_patch_iterator
						&& bestCost > options_.sparse_max_patch_cost*(trackSize - 1))
					{
						it = 0;
						numErrors++;
						bestNormal[0] = 0.0f;
						bestNormal[1] = 0.0f;
						bestNormal[2] = 0.0f;
						break;
					}

					//扰乱最优法向
					PerturbNormal(row, col, refImage.GetK(), perturbtion*M_PI, prepNormal, perturbNormal);
					//如果迭代次数少于轨道数量，那么就用views向量作为随机向量
					if (it <= trackSize - 2)
					{
						//views中前两个已经用了，这里从第三个开始使用，索引号为it+1=2
						const float view[3] = { views.at(it + 1).x, views.at(it + 1).y, views.at(it + 1).z };
						PerturbNormal(row, col, refImage.GetK(), 0.0f, view, RandNormal);
					}
					else//否则就随机产生随机向量
					{
						GenerateRandomNormal(row, col, refImage.GetK(), RandNormal);
					}
				}

				//存储最优的法向量和最好的深度
				sparse_normals_.push_back(cv::Point3f(bestNormal[0], bestNormal[1], bestNormal[2]));
				itNum.push_back(it);
			}

			//把稀疏点云法向量传递给workspace
			workspace_->sparse_normals_ = sparse_normals_;
		}

		//辐射传播稀疏点云数据
		void MyPatchMatch::radiantPropagation()
		{

			std::cout << "Radiant Porpagation Depth and Normal..." << std::endl;
			const Model &model = workspace_->GetModel();
			const int numImages = model.m_images.size();

			depthMaps_.resize(numImages);
			normalMaps_.resize(numImages);
			////处理每一幅图像
			//for (int i = 0; i < problems_.size(); i++)
			for (int i = 0; i < 1; i++)
			{
				//参考图像参数
				const auto &problem = problems_.at(i);
				const int refId = problem.ref_img_id;
				const Image &refImage = model.m_images.at(refId);

				//初始化概率和cost Map
				selProbMaps_.resize(problem.src_img_ids.size());
				prevSelProMaps_.resize(problem.src_img_ids.size());
				costMaps_.resize(problem.src_img_ids.size());
				for (int ii = 0; ii < problem.src_img_ids.size(); ii++)
				{
					Mat<float> selPM(refImage.GetWidth(), refImage.GetHeight(), 1);
					selProbMaps_.at(ii) = selPM;
					Mat<float> prevSelPM(refImage.GetWidth(), refImage.GetHeight(), 1);
					prevSelPM.Fill(0.5f);
					prevSelProMaps_.at(ii) = prevSelPM;
					Mat<float> costM(refImage.GetWidth(), refImage.GetHeight(), 1);
					costM.Fill(3.0f);
					costMaps_.at(ii) = costM;
				}

				//初始化深度和法向map
				DepthMap depthmap1(refImage.GetWidth(), refImage.GetHeight(),
					workspace_->GetDepthRange(refId, false), workspace_->GetDepthRange(refId, true));
				depthMaps_.at(refId) = depthmap1;

				NormalMap normalmap1(refImage.GetWidth(), refImage.GetHeight());
				normalMaps_.at(refId) = normalmap1;

				DepthMap &depthmap = depthMaps_.at(refId);
				NormalMap &normalmap = normalMaps_.at(refId);

				//mask图标注每个像素被转播了几次
				this->mask_ = Mat<int>(refImage.GetWidth(), refImage.GetHeight(), 1);

				std::cout << "Bigen With Sparse Points..." << std::endl;
				////先以稀疏点云为中心辐射扩散
				for (int j = 0; j < model.m_points.size(); j++)
				{
					//没有法向量
					if (sparse_normals_[j].x == 0 && sparse_normals_[j].y == 0 && sparse_normals_[j].z == 0)
						continue;

					const auto &point = model.m_points[j];
					const int tracksNum = point.track.size();

					//track中没有参考图像
					if (FindGivenValue(point.track, tracksNum, refId) == -1)
						continue;

					const Eigen::Vector4f pt(point.x, point.y, point.z, 1.0f);//三维点坐标
					const Eigen::Vector3f xyz = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(refImage.GetP())*pt;
					const int row = xyz(1) / xyz(2) + 0.5f;
					const int col = xyz(0) / xyz(2) + 0.5f;
					const float normal[3] = { sparse_normals_[j].x, sparse_normals_[j].y, sparse_normals_[j].z };
					depthmap.Set(row, col, xyz(2));
					normalmap.SetSlice(row, col, normal);

					//把轨道中的图像写入到选择概率为1
					for (int k = 0; k < tracksNum; k++)
					{
						int srcId = point.track[k];
						if (srcId == refId)
							continue;
						for (int l = 0; l < problem.src_img_ids.size(); l++)
						{
							if (problem.src_img_ids[l] == srcId)
							{
								prevSelProMaps_[l].Set(row, col, 1.0f);
							}
						}
					}

					seedExtend(row, col, problem);
				}

				std::cout << "Using generated to porpagation..." << std::endl;

				//如果队列不空
				while (!Rows_.empty())
				{
					seedExtend(Rows_.front(), Cols_.front(), problem);
					//mask_.Set(Rows_.front(), Cols_.front(), mask_.Get(Rows_.front(), Cols_.front()) + 1);
					Rows_.pop();//清除第一个元素
					Cols_.pop();
				}
				cv::imwrite("11_0.jpg", depthmap.ToBitmap(2, 98));


				//mask_.Fill(0);
				//options_.radiant_max_patch_cost = 0.8;
				//std::cout << "Up->Down,Left->Right..." << std::endl;
				////从左上到右下，迭代一次
				//for (int rr = 0; rr < refImage.GetHeight(); rr++)
				//{
				//	for (int ll = 0; ll < refImage.GetWidth(); ll++)
				//	{
				//		seedExtend(rr, ll, problem);
				//	}
				//}
				//
				//std::cout << "Using queue..." << std::endl;
				////使用队列中
				//while (!Rows_.empty())
				//{
				//	seedExtend(Rows_.front(), Cols_.front(), problem);
				//	Rows_.pop();//清除第一个元素
				//	Cols_.pop();
				//}
				//cv::imwrite("11_1.jpg", depthmap.ToBitmap(2, 98));
				//
				//
				//mask_.Fill(0);
				//options_.radiant_max_patch_cost = 0.6;
				//std::cout << "Left->Right,Up->Down..." << std::endl;
				////从左上到右下，迭代一次			
				//for (int ll = 0; ll < refImage.GetWidth(); ll++)
				//{
				//	for (int rr = 0; rr < refImage.GetHeight(); rr++)
				//	{
				//		seedExtend(rr, ll, problem);
				//	}
				//}
				//std::cout << "Using queue..." << std::endl;
				////使用队列中
				//while (!Rows_.empty())
				//{
				//	seedExtend(Rows_.front(), Cols_.front(), problem);
				//	Rows_.pop();//清除第一个元素
				//	Cols_.pop();
				//}
				//cv::imwrite("11_2.jpg", depthmap.ToBitmap(2, 98));
				//
				//mask_.Fill(0);
				//std::cout << "Down->Up,Right->Left..." << std::endl;
				////从左上到右下，迭代一次			
				//
				//for (int rr = refImage.GetHeight() - 1; rr >= 0; rr--)
				//{
				//	for (int ll = refImage.GetWidth() - 1; ll >= 0; ll--)
				//	{
				//		seedExtend(rr, ll, problem);
				//	}
				//}
				//std::cout << "Using queue..." << std::endl;
				////使用队列中
				//while (!Rows_.empty())
				//{
				//	seedExtend(Rows_.front(), Cols_.front(), problem);
				//	Rows_.pop();//清除第一个元素
				//	Cols_.pop();
				//}
				//cv::imwrite("11_3.jpg", depthmap.ToBitmap(2, 98));
				//
				//
				//mask_.Fill(0);
				//std::cout << "Left->Right,Down->Up..." << std::endl;
				////从左上到右下，迭代一次						
				//for (int ll = refImage.GetWidth() - 1; ll >= 0; ll--)
				//{
				//	for (int rr = refImage.GetHeight() - 1; rr >= 0; rr--)
				//	{
				//		seedExtend(rr, ll, problem);
				//	}
				//}
				//
				//std::cout << "Using queue..." << std::endl;
				////使用队列中
				//while (!Rows_.empty())
				//{
				//	seedExtend(Rows_.front(), Cols_.front(), problem);
				//	Rows_.pop();//清除第一个元素
				//	Cols_.pop();
				//}
				//cv::imwrite("11_4.jpg", depthmap.ToBitmap(2, 98));
				//
				//////				char path1[20];
				//////				char path2[20];
				//////				sprintf_s(path1, "1_%d.jpg", it + 1);
				//////				sprintf_s(path2, "2_%d.jpg", it + 1);
				//////				cv::imwrite(path1, depthmap.ToBitmap(2, 98));
				//////				cv::imwrite(path2, normalmap.ToBitmap());				
				//
				//cv::imwrite("1.jpg", depthmap.ToBitmap(2, 98));
				//cv::imwrite("2.jpg", normalmap.ToBitmap());

				workspace_->WriteDepthMap(refId, depthmap);
				workspace_->WriteNormalMap(refId, normalmap);
			}
		}

		//种子点的扩散
		void MyPatchMatch::seedExtend(const int row, const int col, const PatchMatch::Problem &problem)
		{
			const auto &model = workspace_->GetModel();
			const auto &refImage = model.m_images.at(problem.ref_img_id);
			const cv::Mat &refcvImage = workspace_->GetBitmap(problem.ref_img_id);

			const int dx4[4] = { -1, 0, 1, 0 };
			const int dy4[4] = { 0, -1, 0, 1 };

			const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
			const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

			DepthMap &depthmap = depthMaps_.at(problem.ref_img_id);
			NormalMap &normalmap = normalMaps_.at(problem.ref_img_id);

			const int width = depthmap.GetWidth();
			const int height = depthmap.GetHeight();

			LikelihoodComputer llc(pmOptions_.ncc_sigma,
				pmOptions_.min_triangulation_angle,
				pmOptions_.incident_angle_sigma);

			PhotoConsistencyCostComputer pcc;
			pcc.kWindowSize = 2 * pmOptions_.window_radius;
			pcc.sigma_color = pmOptions_.sigma_color;
			pcc.sigma_spatial = pmOptions_.sigma_spatial;
			pcc.refImage = &refImage;
			pcc.refcvImage = &refcvImage;
			pcc.ref_image_width = width;
			pcc.ref_image_height = height;

			//如果扩散中心没有深度，退出
			if (depthmap.GetDepth(row, col) == 0)
				return;

			//记录四个最小的cost
			std::vector<float> fourNeighbor(4, 3.0f);
			for (int i = 0; i < 4; i++)
			{
				const int col1 = col + dx4[i];
				const int row1 = row + dy4[i];

				//超出范围，跳出
				if (col1 < 0 || col1 >= width || row1 < 0 || row1 >= height)
					continue;

				//如果每个像素已经传播指定次数，跳出
				if (this->mask_.Get(row1, col1) >= this->options_.num_pixel_propagation)
					continue;
				//先前选择概率在迭代过程中所占比重
				const float prevSelWeight = 0.5 + this->mask_.Get(row1, col1) / (2 * this->options_.num_pixel_propagation);

				//如果有较小的cost值，就不用传播了，跳出
				int numSmallCost = 0;
				for (int costId = 0; costId < costMaps_.size(); costId++)
				{
					if (costMaps_[costId].Get(row1, col1) <= this->options_.radiant_max_patch_cost)
					{
						numSmallCost++;
					}
				}
				//已经是最优的深度和法向，跳出
				if (numSmallCost >= options_.samll_cost_num)
				{
					this->mask_.Set(row1, col1, this->mask_.Get(row1, col1) + 1);
					this->Rows_.push(row1);
					this->Cols_.push(col1);
					continue;
				}

				pcc.row = row1;
				pcc.col = col1;
				pcc.local_ref_sum = image_sums_[problem.ref_img_id].Get(row1, col1, 0);
				pcc.local_ref_squared_sum = image_squared_sums_[problem.ref_img_id].Get(row1, col1, 0);

				//如果当前点没有深度和法向，那么就用前一个深度和法向
				if (depthmap.GetDepth(row1, col1) == 0)
				{
					depthmap.Set(row1, col1, depthmap.GetDepth(row, col));
					float norm[3]; normalmap.GetSlice(row, col, norm);
					normalmap.SetSlice(row1, col1, norm);
				}
				const float currDepth = depthmap.GetDepth(row1, col1);
				float currNormal[3]; normalmap.GetSlice(row1, col1, currNormal);

				int iter = 0;
			label1://label标签，多次循环从这里开始
				//计算选择概率
				const Eigen::Vector3f xyz = Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(refImage.GetInvP())*
					Eigen::Vector4f(col1 * currDepth, row1 * currDepth, currDepth, 1.0f);
				const float point[3] = { xyz(0), xyz(1), xyz(2) };//当前深度值的三维点
				std::vector<float> sample_probs(problem.src_img_ids.size(), -1.0f);
				for (int j = 0; j < problem.src_img_ids.size(); j++)
				{
					const auto &srcImage = model.m_images.at(problem.src_img_ids[j]);
					const cv::Mat &srccvImage = workspace_->GetBitmap(problem.src_img_ids[j]);
					pcc.depth = currDepth;
					pcc.normal = currNormal;
					pcc.srcImage = &srcImage;
					pcc.srccvImage = &srccvImage;

					//如果没有cost值，则计算，否则利用上次计算的值
					if (costMaps_[j].Get(row1, col1) == 3.0f)
					{
						costMaps_[j].Set(row1, col1, pcc.Compute());
						//prevSelProMaps_[j].Set(row1, col1, prevSelProMaps_[j].Get(row, col));
					}
					//计算选择概率
					const float cost = costMaps_[j].Get(row1, col1);
					float alpha = 0.0f;
					int numNeigh = 0;
					for (int k = 0; k < 4; k++)
					{
						const int col2 = col1 + dx4[k];
						const int row2 = row1 + dy4[k];
						if (col2 < 0 || col2 >= width || row2 < 0 || row2 >= height)
							continue;
						if (depthmap.GetDepth(row2, col2) == 0)
							continue;
						const float forward = llc.ComputeForwardMessage(cost, prevSelProMaps_[j].Get(row2, col2));
						alpha += ((forward*forward) / (forward*forward + (1 - forward)*(1 - forward)));
						numNeigh++;
					}
					alpha /= numNeigh;
					alpha = this->prevSelProMaps_[j].Get(row1, col1)*prevSelWeight + (1 - prevSelWeight)*alpha;

					float cos_triangulation_angle;
					float cos_incident_angle;
					ComputeViewingAngles(point, currNormal, refImage, srcImage, &cos_triangulation_angle, &cos_incident_angle);
					const float tri_prob = llc.ComputeTriProb(cos_triangulation_angle);
					const float inc_prob = llc.ComputeIncProb(cos_incident_angle);

					float H[9];
					ComposeHomography(&refImage, &srcImage, row1, col1, currDepth, currNormal, H);
					const float res_prob = llc.ComputeResolutionProb(H, row1, col1, pmOptions_.window_radius * 2);

					sample_probs[j] = alpha * tri_prob*inc_prob*res_prob;
				}
				TransformPDFToCDF(sample_probs, problem.src_img_ids.size());

				struct ParamState
				{
					float depth = 0.0f;
					float normal[3];
				};

				// Parameters of previous pixel
				ParamState prev_param_state;
				// Parameters of current pixel
				ParamState curr_param_state;
				// Perturb sampled parameters.
				ParamState perturb_param_state;
				//Random smapled parameters
				ParamState rand_param_state;

				//先前 进行深度传播
				prev_param_state.depth = depthmap.GetDepth(row, col);
				normalmap.GetSlice(row, col, prev_param_state.normal);
				prev_param_state.depth = this->PropagateDepth(refImage.GetK(), prev_param_state.depth, prev_param_state.normal, row, col, row1, col1);
				//当前
				curr_param_state.depth = currDepth;
				curr_param_state.normal[0] = currNormal[0];
				curr_param_state.normal[1] = currNormal[1];
				curr_param_state.normal[2] = currNormal[2];

				//扰乱
				this->PerturbDepth(curr_param_state.depth, 0.02f, &(perturb_param_state.depth));
				this->PerturbNormal(row1, col1, refImage.GetK(), 0.02f*M_PI, curr_param_state.normal, perturb_param_state.normal);

				//随机
				rand_param_state.depth = this->GenerateRandomDepth(depthmap.GetDepthMin(), depthmap.GetDepthMax());
				this->GenerateRandomNormal(row1, col1, refImage.GetK(), rand_param_state.normal);

				//const int kNumCosts = 7;
				//float costs[kNumCosts] = { 0.0f };
				//const float depths[kNumCosts] = {
				//	curr_param_state.depth, prev_param_state.depth, rand_param_state.depth,
				//	curr_param_state.depth, rand_param_state.depth, curr_param_state.depth,
				//	perturb_param_state.depth };
				//const float* normals[kNumCosts] = {
				//	curr_param_state.normal, prev_param_state.normal, rand_param_state.normal,
				//	rand_param_state.normal, curr_param_state.normal, perturb_param_state.normal,
				//	curr_param_state.normal };

				const int kNumCosts = 5;
				float costs[kNumCosts] = { 0.0f };
				const float depths[kNumCosts] = {
					curr_param_state.depth, prev_param_state.depth, rand_param_state.depth,
					curr_param_state.depth, perturb_param_state.depth };
				const float* normals[kNumCosts] = {
					curr_param_state.normal, prev_param_state.normal, perturb_param_state.normal,
					rand_param_state.normal, curr_param_state.normal };

				for (int sample = 0; sample < this->options_.num_smaples; ++sample)
				{
					default_random_engine e(time(0));
					uniform_real_distribution<float> u(0, 1);//float型0-1均匀分布
					const float rand_prob = u(e) - FLT_EPSILON;
					int srcSelId = -1;//原图像在model中索引
					int image_id = -1;//原图像在problem中索引
					for (image_id = 0; image_id < problem.src_img_ids.size(); ++image_id)
					{
						const float prob = sample_probs[image_id];
						if (prob > rand_prob)
						{
							srcSelId = problem.src_img_ids[image_id];
							break;
						}
					}

					if (srcSelId == -1)
					{
						continue;
					}

					costs[0] += this->costMaps_[image_id].Get(row1, col1);

					for (int i = 1; i < kNumCosts; ++i)
					{
						pcc.depth = depths[i];
						pcc.normal = normals[i];
						pcc.srcImage = &(model.m_images[srcSelId]);
						pcc.srccvImage = &(this->workspace_->GetBitmap(srcSelId));
						costs[i] += pcc.Compute();
					}
				}

				// Find the parameters of the minimum cost.
				const int min_cost_idx = this->FindMinCost(costs, kNumCosts);
				const float best_depth = depths[min_cost_idx];
				const float* best_normal = normals[min_cost_idx];

				// Save best new parameters.
				depthmap.Set(row1, col1, best_depth);
				normalmap.SetSlice(row1, col1, best_normal);

				// Use the new cost to recompute the updated forward message and
				// the selection probability.
				pcc.depth = best_depth;
				pcc.normal = best_normal;
				std::vector<float> allCosts(problem.src_img_ids.size());
				for (int k = 0; k < problem.src_img_ids.size(); ++k)
				{
					// Determine the cost for best depth.
					float cost;
					if (min_cost_idx == 0)
					{
						cost = this->costMaps_[k].Get(row1, col1);
					}
					else
					{
						pcc.srcImage = &(model.m_images[problem.src_img_ids[k]]);
						pcc.srccvImage = &(this->workspace_->GetBitmap(problem.src_img_ids[k]));
						cost = pcc.Compute();
						this->costMaps_[k].Set(row1, col1, cost);
					}

					float alpha = 0.0f;
					int numNeigh = 0;
					for (int l = 0; l < 4; l++)
					{
						const int col2 = col1 + dx4[l];
						const int row2 = row1 + dy4[l];
						if (col2 < 0 || col2 >= width || row2 < 0 || row2 >= height)
							continue;
						if (depthmap.GetDepth(row2, col2) == 0)
							continue;
						const float forward = llc.ComputeForwardMessage(cost, this->prevSelProMaps_[k].Get(row2, col2));
						alpha += ((forward*forward) / (forward*forward + (1 - forward)*(1 - forward)));
						numNeigh++;
					}
					alpha /= numNeigh;
					alpha = this->prevSelProMaps_[k].Get(row1, col1)*prevSelWeight + (1 - prevSelWeight)*alpha;

					this->prevSelProMaps_[k].Set(row1, col1, alpha);

					allCosts[k] = cost;
				}

				////升续排列
				//sort(allCosts.begin(), allCosts.end(), std::less<float>());
				//fourNeighbor[i] = (allCosts[0] + allCosts[1]) / 2;

				//if (fourNeighbor[i] <= this->options_.sparse_max_patch_cost)
				//{
				//	this->mask_.Set(row1, col1, this->mask_.Get(row1, col1) + 1);
				//	this->Rows_.push(row1);
				//	this->Cols_.push(col1);
				//	continue;
				//}
				//else if (iter++ <= this->options_.radiant_max_patch_iterator)
				//{
				//	goto label1;
				//}
				//this->mask_.Set(row1, col1, this->mask_.Get(row1, col1) + 1);


				//numSmallCost = 0;
				//for (int costId = 0; costId < this->costMaps_.size(); costId++)
				//{
				//	if (this->costMaps_[costId].Get(row1, col1) < this->options_.radiant_max_patch_cost)
				//	{
				//		numSmallCost++;
				//	}
				//}
				////已经是最优的深度和法向，跳出
				//if (numSmallCost >= this->options_.samll_cost_num)
				//{
				//	this->Rows_.push(row1);
				//	this->Cols_.push(col1);
				//}			
				this->Rows_.push(row1);
				this->Cols_.push(col1);
				this->mask_.Set(row1, col1, this->mask_.Get(row1, col1) + 1);
			}


			//std::vector<int> ids;
			//this->SortWithIds(fourNeighbor, ids);
			//int numExtend = 0;
			//for (int j = 0; j < 4; j++)
			//{
			//	int rr = row + dy4[ids[j]];
			//	int ll = col + dx4[ids[j]];
			//	if (fourNeighbor[j] != 3 && this->mask_.Get(rr, ll)!=1 )
			//	{
			//		this->Rows_.push(rr);
			//		this->Cols_.push(ll); 
			//		this->mask_.Set(rr, ll, this->mask_.Get(rr, ll) + 1);
			//		
			//		if (numExtend ++ == 2)
			//			break;
			//	}
			//}

			default_random_engine e(time(0));
			uniform_real_distribution<float> u(0, 1);
			while (u(e) > 0.5)
			{
				cv::imwrite("1.jpg", depthmap.ToBitmap(2, 98));
				cv::imwrite("2.jpg", normalmap.ToBitmap());
			}
		}

		//开始执行函数
		void MyPatchMatch::Run()
		{
			getProblems();
			computeAllSelfPatch();
			initSparsePoints();
			radiantPropagation();
		}

		std::unique_ptr<Workspace> MyPatchMatch::ReturnValue()
		{
			getProblems();
			computeAllSelfPatch();
			initSparsePoints();
			return std::move(workspace_);
		}

		//产生随机深度
		inline float MyPatchMatch::GenerateRandomDepth(const float min_depht, const float max_depth)const
		{
			default_random_engine e(time(0));
			uniform_real_distribution<float> u(0, 1);//float型0-1均匀分布
			return min_depht + (max_depth - min_depht)*u(e);

		}

		//产生随机法向量
		inline void MyPatchMatch::GenerateRandomNormal(const int row, const int col, const float *refK, float normal[3])const
		{
			// Unbiased sampling of normal, according to George Marsaglia, "Choosing a
			// Point from the Surface of a Sphere", 1972.
			default_random_engine e(time(0));
			uniform_real_distribution<float> u(0, 1);//float型0-1均匀分布

			float v1 = 0.0f;
			float v2 = 0.0f;
			float s = 2.0f;
			while (s >= 1.0f)
			{
				v1 = 2.0f * u(e) - 1.0f;
				v2 = 2.0f * u(e) - 1.0f;
				s = v1 * v1 + v2 * v2;
			}

			const float s_norm = sqrt(1.0f - s);
			normal[0] = 2.0f * v1 * s_norm;
			normal[1] = 2.0f * v2 * s_norm;
			normal[2] = 1.0f - 2.0f * s;

			// Extract 1/fx, -cx/fx, fy, -cy/fy.
			const float ref_inv_K[4] = { 1.0f / refK[0], -refK[2] / refK[0], 1.0f / refK[4], -refK[5] / refK[4] };

			// Make sure normal is looking away from camera.
			const float view_ray[3] = { ref_inv_K[0] * col + ref_inv_K[1],
				ref_inv_K[2] * row + ref_inv_K[3], 1.0f };
			if (DotProduct3(normal, view_ray) > 0) {
				normal[0] = -normal[0];
				normal[1] = -normal[1];
				normal[2] = -normal[2];
			}

		}

		//扰乱法向量
		inline void MyPatchMatch::PerturbNormal(const int row, const int col, const float *refK, const float perturbation, const float normal[3], float perturbed_normal[3]) const
		{
			default_random_engine e(time(0));
			uniform_real_distribution<float> u(0, 1);//float型0-1均匀分布
			// Perturbation rotation angles.
			const float a1 = (u(e) - 0.5f) * perturbation;
			const float a2 = (u(e) - 0.5f) * perturbation;
			const float a3 = (u(e) - 0.5f) * perturbation;

			const float sin_a1 = sin(a1);
			const float sin_a2 = sin(a2);
			const float sin_a3 = sin(a3);
			const float cos_a1 = cos(a1);
			const float cos_a2 = cos(a2);
			const float cos_a3 = cos(a3);

			// R = Rx * Ry * Rz
			float R[9];
			R[0] = cos_a2 * cos_a3;
			R[1] = -cos_a2 * sin_a3;
			R[2] = sin_a2;
			R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
			R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
			R[5] = -cos_a2 * sin_a1;
			R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
			R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
			R[8] = cos_a1 * cos_a2;

			// Perturb the normal vector.
			Mat33DotVec3(R, normal, perturbed_normal);

			// Extract 1/fx, -cx/fx, fy, -cy/fy.
			const float ref_inv_K[4] = { 1.0f / refK[0], -refK[2] / refK[0], 1.0f / refK[4], -refK[5] / refK[4] };

			// Make sure the perturbed normal is still looking in the same direction as
			// the viewing direction.
			const float view_ray[3] = { ref_inv_K[0] * col + ref_inv_K[1],
				ref_inv_K[2] * row + ref_inv_K[3], 1.0f };
			if (DotProduct3(perturbed_normal, view_ray) >= 0.0f) {
				if (a1 == 0 && a2 == 0 & a3 == 0)//如果扰乱角度为0，就是简单的判断一下方向
				{
					perturbed_normal[0] *= -1;
					perturbed_normal[1] *= -1;
					perturbed_normal[2] *= -1;
				}
				else//如果扰乱角度不为0，那么就把原来法向当做扰乱后的法向
				{
					perturbed_normal[0] = normal[0];
					perturbed_normal[1] = normal[1];
					perturbed_normal[2] = normal[2];
				}
			}

			// Make sure normal has unit norm.
			const float inv_norm = 1.0f / sqrt(DotProduct3(perturbed_normal, perturbed_normal));
			perturbed_normal[0] *= inv_norm;
			perturbed_normal[1] *= inv_norm;
			perturbed_normal[2] *= inv_norm;
		}

		//扰乱深度值
		inline void MyPatchMatch::PerturbDepth(const float srcDepth, const float perturbation, float *pertDepth)const
		{
			default_random_engine e(time(0));
			uniform_real_distribution<float> u(-1, 1);//float型0-1均匀分布
			//扰乱百分之perturbation的深度
			*pertDepth = srcDepth * (1 + u(e)*perturbation);
		}

		//寻找最小值
		inline int MyPatchMatch::FindMinCost(const float *costs, const int kNumCosts) const
		{
			float min_cost = costs[0];
			int min_cost_idx = 0;
			for (int idx = 1; idx < kNumCosts; ++idx)
			{
				if (costs[idx] < min_cost) 
				{
					min_cost = costs[idx];
					min_cost_idx = idx;
				}
			}
			return min_cost_idx;
		}

		//从先前深度和法向传播深度值
		// Transfer depth on plane from viewing ray at row1 to row2. The returned
		// depth is the intersection of the viewing ray through row2 with the plane
		// at row1 defined by the given depth and normal.
		inline float MyPatchMatch::PropagateDepth(const float *refK, const float depth1, const float normal1[3],
			const float row1, const float col1, const float row2, const float col2) const
		{
			// Extract 1/fx, -cx/fx, fy, -cy/fy.
			const float ref_inv_K[4] = { 1.0f / refK[0], -refK[2] / refK[0], 1.0f / refK[4], -refK[5] / refK[4] };

			// Point along first viewing ray.
			const float p1[3] = { depth1*(ref_inv_K[0] * col1 + ref_inv_K[1]),
				depth1 * (ref_inv_K[2] * row1 + ref_inv_K[3]),
				depth1 };

			// Point on second viewing ray.
			const float p2[3] = { ref_inv_K[0] * col2 + ref_inv_K[1],
				ref_inv_K[2] * row2 + ref_inv_K[3],
				1.0f };
			// const float y4 = 1.0f;

			const float denom = DotProduct3(p1, normal1) / DotProduct3(p2, normal1);
			//如果深度值过大过小，那么就返回原深度值
			if (denom < depth1*0.8 || denom > depth1*1.2)
			{
				return depth1;
			}
			return denom;
		}

		//计算入射角三角O型夹角概率
		inline void MyPatchMatch::ComputeViewingAngles(const float point[3], const float normal[3], const Image &refImage, const Image &srcImage,
			float* cos_triangulation_angle, float* cos_incident_angle) const
		{
			*cos_triangulation_angle = 0.0f;
			*cos_incident_angle = 0.0f;

			// Projection center of source image.
			const float *C = srcImage.GetCenter();

			//Projection center of ref image
			const float *refC = refImage.GetCenter();

			// Ray from point to camera.
			const float SX[3] = { C[0] - point[0], C[1] - point[1], C[2] - point[2] };
			const float RX[3] = { refC[0] - point[0], refC[1] - point[1], refC[2] - point[2] };

			// Length of ray from reference image to point.
			const float RX_norm = sqrt(DotProduct3(RX, RX));

			// Length of ray from source image to point.
			const float SX_norm = sqrt(DotProduct3(SX, SX));

			*cos_incident_angle = DotProduct3(SX, normal) / (SX_norm);
			*cos_triangulation_angle = DotProduct3(SX, point) / (RX_norm * SX_norm);
		}

		//转化为PDF概率分布
		inline void MyPatchMatch::TransformPDFToCDF(std::vector<float> &probs, const int num_probs) const {
			float prob_sum = 0.0f;
			for (int i = 0; i < num_probs; ++i)
			{
				prob_sum += probs[i];
			}
			const float inv_prob_sum = 1.0f / prob_sum;

			float cum_prob = 0.0f;
			for (int i = 0; i < num_probs; ++i)
			{
				const float prob = probs[i] * inv_prob_sum;
				cum_prob += prob;
				probs[i] = cum_prob;
			}
		}

		//寻找给定值,返回索引值
		inline int MyPatchMatch::FindGivenValue(const std::vector<int> &value, const int num, const int ref)const
		{
			for (int i = 0; i < num; i++)
			{
				if (value[i] == ref)
					return i;
			}
			return -1;
		}

		//排序，并且返回原先序号
		inline void MyPatchMatch::SortWithIds(std::vector<float> &value, std::vector<int> &ids) const
		{
			float tempV;
			int tempId;
			ids.resize(value.size());
			std::iota(ids.begin(), ids.end(), 0);
			for (int i = 0; i < value.size() - 1; i++)
			{
				for (int j = i + 1; j < value.size(); j++)
				{
					if (value[j] < value[i])
					{
						tempV = value[i];
						value[i] = value[j];
						value[j] = tempV;

						tempId = ids[i];
						ids[i] = ids[j];
						ids[j] = tempId;
					}
				}
			}
		}

	}
}