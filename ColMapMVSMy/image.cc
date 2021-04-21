#include "image.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace colmap {
	namespace mvs {

		Image::Image() {}

		Image::Image(const std::string& path, const std::string& fileName,
			const std::string& workspacePath, const std::string& newPath,
			size_t width, const size_t height, const float* K, const float* R, const float* T, const float* k)
			: path_(path), fileName_(fileName), workspacePath_(workspacePath), newPath_(newPath), width_(width), height_(height) 
		{
			memcpy(K_, K, 9 * sizeof(float));
			memcpy(R_, R, 9 * sizeof(float));
			memcpy(T_, T, 3 * sizeof(float));
			memcpy(k_, k, 2 * sizeof(float));

			ComposeProjectionMatrix(K_, R_, T_, P_);
			ComposeInverseProjectionMatrix(K_, R_, T_, inv_P_);
			ComputeProjectionCenter(R_, T_, center_);
		}

		void Image::setFileName(const std::string &fileName) { fileName_ = fileName; }

		void Image::SetPath(const std::string &path) { path_ = path; }

		void Image::SetNok() { k_[0] = 0.0f; k_[1] = 0.0f; }

		void Image::SetBitmap(const cv::Mat& bitmap)
		{
			assert(width_ == bitmap.cols);
			assert(height_ == bitmap.rows);

			bitmap_ = bitmap;  // 不共享内存
		}

		void Image::SetWidth(size_t width) { width_ = width; }
		void Image::SetHeight(size_t height) { height_ = height; }

		void Image::ResizeBitMap()
		{
			assert(!bitmap_.empty());
			cv::resize(bitmap_, bitmap_, cv::Size(width_, height_));
		}

		void Image::WriteBitMap()const { cv::imwrite(path_, bitmap_); }

		void Image::Downsize(const size_t max_width, const size_t max_height)
		{
			if (width_ <= max_width && height_ <= max_height)
			{
				return;
			}

			float factor_x = static_cast<float>((float)width_ / (float)max_width);
			float factor_y = static_cast<float>((float)height_ / (float)max_height);

			//选择缩减尺度最大的尺度因子
			factor_x = std::max(factor_x, factor_y);
			factor_y = factor_x;

			Rescale(factor_x, factor_y);
		}

		void Image::Rescale(const float factor) { Rescale(factor, factor); }

		void Image::Rescale(const float factor_x, const float factor_y)
		{
			const size_t new_width = std::round((std::max(1.0f, width_ / factor_x)));
			const size_t new_height = std::round((std::max(1.0f, height_ / factor_y)));

			//assert(!bitmap_.empty());  // 因为model中已经读入图像了

			// 首次读入图像
			bitmap_ = cv::imread(path_);  // 先读入原始图像
			cv::resize(bitmap_, bitmap_, cv::Size(new_width, new_height));

			////修改将采样后的图像名字和路径
			const std::string imgName = fileName_;  // 图像名字不变
			const std::string imgPath = workspacePath_ + newPath_ + "/" + imgName;

			depthmapPath_ = workspacePath_ + newPath_ + "/depth_maps/dslr_images_undistorted/";
			normalmapPath_ = workspacePath_ + newPath_ + "/normal_maps/dslr_images_undistorted/";
			consistencyPath_ = workspacePath_ + newPath_ + "/consistecy_graphs/dslr_images_undistorted/";

			cv::imwrite(imgPath, bitmap_);  // 把将采样后的图像写到本地
			fileName_ = imgName;
			path_ = imgPath;

			const float scale_x = new_width / static_cast<float>(width_);
			const float scale_y = new_height / static_cast<float>(height_);

			if (K_[0] == K_[4])
			{
				K_[0] *= ((scale_x + scale_y) / 2.0f);
				K_[4] *= ((scale_x + scale_y) / 2.0f);
				K_[2] *= scale_x;
				K_[5] *= scale_y;
			}
			else
			{
				K_[0] *= scale_x;
				K_[2] *= scale_x;
				K_[4] *= scale_y;
				K_[5] *= scale_y;
			}

			ComposeProjectionMatrix(K_, R_, T_, P_);
			ComposeInverseProjectionMatrix(K_, R_, T_, inv_P_);

			width_ = new_width;
			height_ = new_height;
		}


		void Image::UndistortionRescale(const float factor_x, const float factor_y)
		{
			const size_t new_width = static_cast<size_t>(std::max(1.0f, factor_x*width_));
			const size_t new_height = static_cast<size_t>(std::max(1.0f, factor_y*height_));

			if (!bitmap_.empty())
			{
				cv::resize(bitmap_, bitmap_, cv::Size(new_width, new_height));
			}

			const float scale_x = new_width / static_cast<float>(width_);
			const float scale_y = new_height / static_cast<float>(height_);


			K_[2] *= scale_x;
			K_[5] *= scale_y;

			ComposeProjectionMatrix(K_, R_, T_, P_);
			ComposeInverseProjectionMatrix(K_, R_, T_, inv_P_);

			width_ = new_width;
			height_ = new_height;
		}

		void Image::upSamplingRescale(const size_t width, const size_t height)
		{
			const float scale_x = width / static_cast<float>(width_);
			const float scale_y = height / static_cast<float>(height_);

			K_[0] *= ((scale_x + scale_y) / 2.0f);
			K_[4] *= ((scale_x + scale_y) / 2.0f);
			K_[2] *= scale_x;
			K_[5] *= scale_y;

			ComposeProjectionMatrix(K_, R_, T_, P_);
			ComposeInverseProjectionMatrix(K_, R_, T_, inv_P_);

			width_ = width;
			height_ = height;
		}


		void ComputeRelativePose(const float R1[9], const float T1[3],
			const float R2[9], const float T2[3], float R[9],
			float T[3])
		{
			const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R1_m(R1);
			const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R2_m(R2);
			const Eigen::Map<const Eigen::Matrix<float, 3, 1>> T1_m(T1);
			const Eigen::Map<const Eigen::Matrix<float, 3, 1>> T2_m(T2);
			Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R_m(R);
			Eigen::Map<Eigen::Vector3f> T_m(T);

			R_m = R2_m * R1_m.transpose();
			T_m = T2_m - R_m * T1_m;
		}

		void ComposeProjectionMatrix(const float K[9], const float R[9],
			const float T[3], float P[12]) 
		{
			Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> P_m(P);
			P_m.leftCols<3>() =
				Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R);
			P_m.rightCols<1>() = Eigen::Map<const Eigen::Vector3f>(T);
			P_m = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K) * P_m;  // 3×3, 3×4 -> 3×4
		}

		void ComposeInverseProjectionMatrix(const float K[9], const float R[9],
			const float T[3], float inv_P[12]) 
		{
			Eigen::Matrix<float, 4, 4, Eigen::RowMajor> P;
			ComposeProjectionMatrix(K, R, T, P.data());
			P.row(3) = Eigen::Vector4f(0, 0, 0, 1);
			//const Eigen::Matrix4f inv_P_temp = P.inverse();
			//Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> inv_P_m(inv_P);
			//inv_P_m = inv_P_temp.topRows<3>();

			cv::Mat matPtemp(4, 4, CV_32FC1);
			cv::Mat matPinvtemp(4, 4, CV_32FC1);
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					matPtemp.at<float>(i, j) = P(i, j);
				}
			}
			matPinvtemp = matPtemp.inv();
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					inv_P[i * 4 + j] = matPinvtemp.at<float>(i, j);
				}
			}

		}

		void ComputeProjectionCenter(const float R[9], const float T[3], float C[3])
		{
			const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R_m(R);
			const Eigen::Map<const Eigen::Matrix<float, 3, 1>> T_m(T);
			Eigen::Map<Eigen::Vector3f> C_m(C);
			C_m = -R_m.transpose() * T_m;
		}

		void RotatePose(const float RR[9], float R[9], float T[3])
		{
			Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R_m(R);
			Eigen::Map<Eigen::Matrix<float, 3, 1>> T_m(T);
			const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> RR_m(RR);
			R_m = RR_m * R_m;
			T_m = RR_m * T_m;
		}

	}  // namespace mvs
}  // namespace colmap
