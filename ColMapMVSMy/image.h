#ifndef COLMAP_SRC_MVS_IMAGE_H_
#define COLMAP_SRC_MVS_IMAGE_H_

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>


namespace colmap {
	namespace mvs {

		class Image
		{
		public:
			Image();
			Image(const std::string& path, const std::string& fileName,
				const std::string& workspacePath, const std::string& newPath,
				size_t width, const size_t height, const float* K, const float* R, const float* T, const float* k);

			inline size_t GetWidth() const;
			inline size_t GetHeight() const;

			void SetWidth(size_t width);
			void SetHeight(size_t height);
			void ResizeBitMap();
			void WriteBitMap() const;

			void SetBitmap(const cv::Mat& bitmap);
			inline const cv::Mat& GetBitmap() const;

			inline const std::string& GetPath() const;
			inline const float* GetR() const;
			inline const float* GetInvR();

			inline const float* GetT() const;
			inline const float* GetK() const;
			inline const float* GetP() const;
			inline const float* GetInvP() const;
			inline const float* GetViewingDirection() const;
			inline const float* GetCenter() const;
			inline const float* Getk() const;
			inline const float* GetInvK();
			inline const std::string& GetfileName() const;
			inline const std::string& GetDepthMapPath()const;
			inline const std::string& GetNormalMapPath()const;
			inline const std::string& GetConsistencyPath()const;

			void setFileName(const std::string& fileName);
			void SetPath(const std::string& path);
			void SetNok();

			// 去扭曲后对图像尺寸进行修改，不改变焦距！！！！！
			void UndistortionRescale(const float facotr_x, const float factor_y);

			// 上采样，对图像信息进行修改，长宽，K
			void upSamplingRescale(const size_t width, const size_t height);

			void Rescale(const float factor);
			void Rescale(const float factor_x, const float factor_y);
			void Downsize(const size_t max_width, const size_t max_height);

		private:
			std::string path_;
			std::string fileName_;
			std::string workspacePath_;
			std::string newPath_;  // 所有结果所在目录
			std::string depthmapPath_;
			std::string normalmapPath_;
			std::string consistencyPath_;
			size_t width_;
			size_t height_;
			cv::Mat bitmap_;
			float K_[9];
			float inv_K_[9];
			float R_[9];
			float inv_R_[9];
			float T_[3];
			float P_[12];
			float inv_P_[12];
			float k_[2];
			float center_[3];
		};

		void ComputeRelativePose(const float R1[9], const float T1[3],
			const float R2[9], const float T2[3], float R[9],
			float T[3]);

		void ComposeProjectionMatrix(const float K[9], const float R[9],
			const float T[3], float P[12]);

		void ComposeInverseProjectionMatrix(const float K[9], const float R[9],
			const float T[3], float inv_P[12]);

		void ComputeProjectionCenter(const float R[9], const float T[3], float C[3]);

		void RotatePose(const float RR[9], float R[9], float T[3]);

		////////////////////////////////////////////////////////////////////////////////
		// Implementation
		////////////////////////////////////////////////////////////////////////////////

		size_t Image::GetWidth() const { return width_; }

		size_t Image::GetHeight() const { return height_; }

		const std::string& Image::GetPath() const
		{
			return path_;
		}

		const cv::Mat& Image::GetBitmap() const { return bitmap_; }

		const float* Image::GetR() const { return R_; }

		const float* Image::GetInvR()
		{
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_mat = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R_);
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_mat_inv = R_mat.inverse();

			memcpy(inv_R_, R_mat_inv.data(), sizeof(float) * 9);

			//for (int i = 0; i < 9; i++)
			//{
			//	printf("%.3f\n", inv_R_[i]);
			//}

			return inv_R_;
		}

		const float* Image::GetT() const { return T_; }

		const float* Image::GetK() const { return K_; }

		const float* Image::GetInvK()
		{
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_mat = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K_);
			Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_mat_inv = K_mat.inverse();

			memcpy(inv_K_, K_mat_inv.data(), sizeof(float) * 9);

			//for (int i = 0; i < 9; i++)
			//{
			//	printf("%.3f\n", inv_K_[i]);
			//}

			return inv_K_;
		}

		const float* Image::GetP() const { return P_; }

		const float* Image::GetInvP() const { return inv_P_; }

		const float* Image::GetViewingDirection() const { return &R_[6]; }

		const float* Image::GetCenter() const { return center_; }

		const float* Image::Getk() const { return k_; }

		const std::string& Image::GetfileName() const { return fileName_; }

		const std::string& Image::GetDepthMapPath() const { return depthmapPath_; }
		const std::string& Image::GetNormalMapPath()const { return normalmapPath_; }
		const std::string& Image::GetConsistencyPath()const { return consistencyPath_; }


	}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_IMAGE_H_
