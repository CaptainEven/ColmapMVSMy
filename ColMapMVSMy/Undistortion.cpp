#include "Undistortion.h"

//#include "Eigen/Core"
//#include "Eigen/Dense"

namespace colmap {
	namespace mvs {

		void Undistorter::run()
		{
			//std::cout << "Image Undistortion: ..." << std::endl;
			for (int i = 0; i < images_.size(); i++)
			{
				undistortCamera(i);
				undistortImage(i);
				writeUndistort(i);
			}
		}

		//���ų߶����ӵļ���ʽ���� ��ȥŤ��ͼ����û�пհ����أ�blankPixles=0��
		//������ȥŤ��ͼ���а�����������Ť��ͼ���е����أ�blankPixels=1��
		//���ֽ��಻�䣬��������λ��ͼ������
		void Undistorter::undistortCamera(const size_t imageId)
		{
			assert(options_.blankPixels >= 0 && options_.blankPixels <= 1);

			const Image& image = images_.at(imageId);

			float Nok[2] = { 0.0f, 0.0f };

			float leftMin = std::numeric_limits<float>::max();
			float leftMax = std::numeric_limits<float>::lowest();
			float rightMin = std::numeric_limits<float>::max();
			float rightMax = std::numeric_limits<float>::lowest();

			//����ͼ������������У�ȷ��x���������Сֵ
			for (int y = 0; y < image.GetHeight(); y++)
			{
				//��߽�
				const cv::Point2f worldPoint1 = imageToWorld(image.GetK(),
															 image.Getk(),
															 cv::Point2f(0, y));
				const cv::Point2f imagePoint1 = worldToImage(image.GetK(),
															 Nok,
															 worldPoint1);
				leftMin = std::min(leftMin, imagePoint1.x);
				leftMax = std::max(leftMax, imagePoint1.x);

				//�ұ߽�
				const cv::Point2f worldPoint2 = imageToWorld(image.GetK(), image.Getk(), cv::Point2f(image.GetWidth() - 1, y));
				const cv::Point2f imagePoint2 = worldToImage(image.GetK(), Nok, worldPoint2);
				rightMin = std::min(rightMin, imagePoint2.x);
				rightMax = std::max(rightMax, imagePoint2.x);
			}

			float topMin = std::numeric_limits<float>::max();
			float topMax = std::numeric_limits<float>::lowest();
			float bottomMin = std::numeric_limits<float>::max();
			float bottomMax = std::numeric_limits<float>::lowest();

			//����ͼ������Ϻ������У�ȷ��y���������Сֵ
			for (int x = 0; x < image.GetWidth(); x++)
			{
				//�ϱ߽�
				const cv::Point2f worldPoint1 = imageToWorld(image.GetK(), image.Getk(), cv::Point2f(x, 0));
				const cv::Point2f imagePoint1 = worldToImage(image.GetK(), Nok, worldPoint1);
				topMin = std::min(topMin, imagePoint1.y);
				topMax = std::max(topMax, imagePoint1.y);

				//�±߽�
				const cv::Point2f worldPoint2 = imageToWorld(image.GetK(), image.Getk(), cv::Point2f(x, image.GetHeight() - 1));
				const cv::Point2f imagePoint2 = worldToImage(image.GetK(), Nok, worldPoint2);
				bottomMin = std::min(bottomMin, imagePoint2.y);
				bottomMax = std::max(bottomMax, imagePoint2.y);
			}

			const float cx = image.GetK()[2];
			const float cy = image.GetK()[5];

			//�����ų߶�����
			//ʹ��ȥŤ��ͼ���������Ť��ͼ�������,�ŵ����
			const float minScalex = std::min(cx / (cx - leftMin), (image.GetWidth() - cx) / (rightMax - cx));
			const float minScaley = std::min(cy / (cy - topMin), (image.GetHeight() - cy) / (bottomMax - cy));

			//ʹ��ȥŤ��ͼ����û�пհ����أ����ظ�������ԭŤ��ͼ�񣩣��ŵ���С
			const float maxScalex = std::max(cx / (cx - leftMax), (image.GetWidth() - cx) / (rightMin - cx));
			const float maxScaley = std::max(cy / (cy - topMax), (image.GetHeight() - cy) / (bottomMin - cy));

			//����blankPixel��ֵ���ų߶�����
			scalex_ = 1.0 / (minScalex*options_.blankPixels + maxScalex * (1.0 - options_.blankPixels));
			scaley_ = 1.0 / (minScaley*options_.blankPixels + maxScaley * (1.0 - options_.blankPixels));

			//ǯ�����ų߶�����
			scalex_ = std::max(options_.minScale, std::min(scalex_, options_.maxScale));
			scaley_ = std::max(options_.minScale, std::min(scaley_, options_.maxScale));

			////////////
			//����ȥŤ��ͼ��Ĵ�С
			undistortedImageWidth_ = static_cast<size_t>(std::max(1.0f, scalex_*image.GetWidth()));
			undistortedImageHeight_ = static_cast<size_t>(std::max(1.0f, scaley_*image.GetHeight()));
		}

		//ȥŤ��ͼ���Ա�֤��ͼ����������һ����������ģ��
		void Undistorter::undistortImage(const size_t img_id)
		{
			const Image& image = images_.at(img_id);
			const cv::Mat& srcImage = cv::imread(image.GetPath());
			const float K = image.GetK()[0];

			const float cx = undistortedImageWidth_ / 2.0f;
			const float cy = undistortedImageHeight_ / 2.0f;

			const float undistortedImageK[9] = { K, 0.0f, cx, 0.0f, K, cy, 0.0f, 0.0f, 1.0f };
			float Nok[2] = { 0.0f, 0.0f };

			undistortedImage_.create(undistortedImageHeight_, undistortedImageWidth_, CV_8UC3);
			for (int y = 0; y < undistortedImage_.rows; y++)
			{
				for (int x = 0; x < undistortedImage_.cols; x++)
				{
					const cv::Point2f worldPoint = imageToWorld(undistortedImageK, Nok, cv::Point2f(x, y));
					const cv::Point2f imagePoint = worldToImage(image.GetK(), image.Getk(), worldPoint);

					cv::Vec3b color = interpolateBilinear(srcImage, imagePoint);
					undistortedImage_.at<cv::Vec3b>(y, x) = color;
				}
			}
		}

		//��ԭ��Ť��ͼ�������ΪȥŤ���Ĳ���
		void Undistorter::writeUndistort(const size_t imageId)
		{
			//���Ȱ�ImageModel�޸�
			Image &image = images_.at(imageId);
			const std::string imageName = image.GetfileName();//���޸�ԭͼ������
			const std::string outputImageNamePath = outputPath_ + imageName;

			image.SetPath(outputImageNamePath);
			image.SetNok();
			//ȥŤ��ʱ���Ĳ������������û�ж��Ӱ�죬��Ϊͼ��߶ȱ仯Ҳ�Ƿǳ�С��
			image.UndistortionRescale(scalex_, scaley_);

			//cv::Mat grayImage;
			//cv::cvtColor(undistortedImage_, grayImage, CV_RGB2GRAY);
			//image.SetBitmap(grayImage);

			//��ȥŤ����ͼ��д������
			cv::imwrite(outputImageNamePath, undistortedImage_);
		}


		cv::Point2f Undistorter::imageToWorld(const float *K, const float *k, const cv::Point2f &xy) const
		{
			cv::Point2f XY;
			const float f = K[0];
			const float c1 = K[2];
			const float c2 = K[5];

			//K^-1*x,ת��Ϊ��������
			XY.x = (xy.x - c1) / f;
			XY.y = (xy.y - c2) / f;

			//������ȥŤ��
			if (k[0] != 0 || k[1] != 0)
			{

				float cx = XY.x;//ÿ�ε����������ֵ
				float cy = XY.y;

				float lastx = cx;//��һ�ε���������ֵ
				float lasty = cy;

				do
				{
					lastx = cx;
					lasty = cy;

					cx = XY.x / (1 + k[0] * (cx*cx + cy * cy) + k[1] * (cx*cx + cy * cy)*(cx*cx + cy * cy));
					cy = XY.y / (1 + k[0] * (cx*cx + cy * cy) + k[1] * (cx*cx + cy * cy)*(cx*cx + cy * cy));

				} while ((lastx - cx)*(lastx - cx) + (lasty - cy)*(lasty - cy) > 0.000001);

				XY.x = cx;
				XY.y = cy;
			}


			//// Parameters for Newton iteration using numerical differentiation with
			//// central differences, 100 iterations should be enough even for complex
			//// camera models with higher order terms.
			//cv::Point2d XYd(XY.x, XY.y);
			//double *u = &XYd.x;
			//double *v = &XYd.y;
			//
			//const size_t kNumIterations = 100;
			//const double kMaxStepNorm = 1e-10;
			//const double kRelStepSize = 1e-6;
			//Eigen::Matrix2d J;
			//const Eigen::Vector2d x0(*u, *v);
			//Eigen::Vector2d x(*u, *v);
			//Eigen::Vector2d dx;
			//Eigen::Vector2d dx_0b;
			//Eigen::Vector2d dx_0f;
			//Eigen::Vector2d dx_1b;
			//Eigen::Vector2d dx_1f;
			//for (size_t i = 0; i < kNumIterations; ++i) {
			//	const double step0 = std::max(std::numeric_limits<double>::epsilon(),
			//		std::abs(kRelStepSize * x(0)));
			//	const double step1 = std::max(std::numeric_limits<double>::epsilon(),
			//		std::abs(kRelStepSize * x(1)));
			//	Distortion(k, x(0), x(1), &dx(0), &dx(1));
			//	Distortion(k, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
			//	Distortion(k, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
			//	Distortion(k, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
			//	Distortion(k, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
			//	J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
			//	J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
			//	J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
			//	J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
			//	const Eigen::Vector2d step_x = J.inverse() * (x + dx - x0);
			//	x -= step_x;
			//	if (step_x.squaredNorm() < kMaxStepNorm) {
			//		break;
			//	}
			//}
			//*u = x(0);
			//*v = x(1);
			//XY.x = (float)x(0);
			//XY.y = (float)x(1);


			return XY;
		}

		cv::Point2f Undistorter::worldToImage(const float *K, const float *k, const cv::Point2f &XY) const
		{
			cv::Point2f xy;
			const float f = K[0];
			const float c1 = K[2];
			const float c2 = K[5];

			//ȥŤ��
			//xy.x = XY.x*(1 + k[0] * (XY.x*XY.x + XY.y*XY.y) + k[1] * (XY.x*XY.x + XY.y*XY.y)*(XY.x*XY.x + XY.y*XY.y));
			//xy.y = XY.y*(1 + k[0] * (XY.x*XY.x + XY.y*XY.y) + k[1] * (XY.x*XY.x + XY.y*XY.y)*(XY.x*XY.x + XY.y*XY.y));

			float du, dv;
			Distortion(k, XY.x, XY.y, &du, &dv);
			xy.x = XY.x + du;
			xy.y = XY.y + dv;

			//K*x,ת����ͼ������
			xy.x = f * xy.x + c1;
			xy.y = f * xy.y + c2;

			return xy;
		}

		template<typename T>
		void Undistorter::Distortion(const T *params, const T u, const T v, T *du, T *dv) const
		{
			const T k1 = params[0];
			const T k2 = params[1];

			const T u2 = u * u;
			const T v2 = v * v;
			const T r2 = u2 + v2;
			const T radial = k1 * r2 + k2 * r2 * r2;

			*du = u * radial;
			*dv = v * radial;
		}

		cv::Vec3b interpolateBilinear(const cv::Mat &srcImage, const cv::Point2f &point)
		{
			cv::Vec3b color(0, 0, 0);
			const float x0 = std::floor(point.x);//��
			const float x1 = x0 + 1;
			const float y0 = std::floor(point.y);//��
			const float y1 = y0 + 1;
			if (x0 >= 0 && x1 < srcImage.cols && y0 >= 0 && y1 < srcImage.rows)
			{
				const cv::Vec3b v00 = srcImage.at<cv::Vec3b>(y0, x0);
				const cv::Vec3b v01 = srcImage.at<cv::Vec3b>(y0, x1);
				const cv::Vec3b v10 = srcImage.at<cv::Vec3b>(y1, x0);
				const cv::Vec3b v11 = srcImage.at<cv::Vec3b>(y1, x1);

				const float dx = point.x - x0;
				const float dy = point.y - y0;
				const float dxx = 1 - dx;
				const float dyy = 1 - dy;

				//f(i+u,j+v) = (1-u)(1-v)f(i,j) + (1-u)vf(i,j+1) + u(1-v)f(i+1,j) + uvf(i+1,j+1)
				for (int i = 0; i < 3; i++)
				{
					color[i] = static_cast<uchar>(dxx*dyy*v00[i] + dx * dyy*v01[i] + dxx * dy*v10[i] + dx * dy*v11[i]);
				}
			}
			return color;
		}

	}
}