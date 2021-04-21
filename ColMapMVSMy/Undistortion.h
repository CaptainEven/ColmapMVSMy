#ifndef	UNDISTORTION_H
#define UNDISTORTION_H

#include <iostream>
#include <vector>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include "image.h"

namespace colmap {
	namespace mvs {

		class Undistorter
		{
		public:

			struct Options
			{
				double blankPixels = 0.0;//��ȥŤ��ͼ���У��հ����ظ�����0��1��
				float minScale = 0.2;//����հ��������Ƶģ������С�ı�߶�
				float maxScale = 2.0;
				int maxImageSize = -1;//��ȥŤ��ͼ��ģ�������߶�
			};

			Undistorter(const Undistorter::Options &option,
				const std::string &outputPath, std::vector<Image> &images) :
				options_(option),
				outputPath_(outputPath),
				images_(images) {}

			virtual ~Undistorter() {}

			void run();

		private:
			void undistortCamera(const size_t imageId);
			void undistortImage(const size_t imageId);
			void writeUndistort(const size_t imageId);

			cv::Point2f imageToWorld(const float *K, const float *k, const cv::Point2f &xy) const;
			cv::Point2f worldToImage(const float *K, const float *k, const cv::Point2f &XY) const;

			template<typename T>
			void Distortion(const T *params, const T u, const T v, T *du, T *dv) const;

			Undistorter::Options options_;
			const std::string outputPath_;
			std::vector<Image>& images_;  // �����Ҫ�޸ĵı𴦵����ݣ�������Ҫ�������ã�������
										  // �������ã�ֻ�Ǹı�������ֵ�������ܸı��������õط���ֵ
			float scalex_;
			float scaley_;
			size_t undistortedImageWidth_;
			size_t undistortedImageHeight_;
			cv::Mat undistortedImage_;

		};

		cv::Vec3b interpolateBilinear(const cv::Mat &srcIamge, const cv::Point2f &point);

	}  // namespace mvs
}  // namespace colmap

#endif//UNDISTORTION_H
