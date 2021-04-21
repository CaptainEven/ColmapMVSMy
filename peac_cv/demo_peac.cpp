#include "./peac/AHCPlaneFitter.hpp"
#include <string>


struct OrganizedImage3D
{
	const cv::Mat_<cv::Vec3f>& cloud;

	//note: ahc::PlaneFitter assumes mm as unit!!!
	OrganizedImage3D(const cv::Mat_<cv::Vec3f>& c) : cloud(c) {}

	inline int width() const { return cloud.cols; }

	inline int height() const { return cloud.rows; }

	inline bool get(const int row, const int col,
		double& x, double& y, double& z) const
	{
		const cv::Vec3f& p = cloud.at<cv::Vec3f>(row, col);
		x = p[0];
		y = p[1];
		z = p[2];
		return z > 0 && isnan(z) == 0;  // return false if current depth is NaN
	}
};

typedef ahc::PlaneFitter<OrganizedImage3D> PlaneFitter;

int TestDemo()
{
	const std::string img_path = std::string("./resource/1305031103.027881.png");
	cv::Mat depth = cv::imread(img_path, cv::IMREAD_ANYDEPTH);

	// ------ ԭʼ����demoֻ�����ڲ�, û�п������
	const float f = 525;
	const float cx = 319.5;
	const float cy = 239.5;
	const float max_use_range = 10;

	// ���3D����
	cv::Mat_<cv::Vec3f> cloud(depth.rows, depth.cols);

	for (int r = 0; r < depth.rows; r++)
	{
		// ���ͼһ�е�ָ��
		const unsigned short* depth_ptr = depth.ptr<unsigned short>(r);

		cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);

		for (int c = 0; c < depth.cols; c++)
		{
			float z = (float)depth_ptr[c] / 5000.0;
			if (z > max_use_range)
			{
				z = 0;
			}

			pt_ptr[c][0] = (c - cx) / f * z * 1000.0;  // m->mm
			pt_ptr[c][1] = (r - cy) / f * z * 1000.0;  // m->mm
			pt_ptr[c][2] = z * 1000.0;  // m->mm
		}
	}

	// ------

	PlaneFitter pf;
	pf.minSupport = 3000;
	pf.windowWidth = 20;
	pf.windowHeight = 20;
	pf.doRefine = true;

	cv::Mat seg(depth.rows, depth.cols, CV_8UC3);
	OrganizedImage3D Ixyz(cloud);
	pf.run(&Ixyz, 0, &seg);

	// ----- 

	cv::Mat depth_color;
	depth.convertTo(depth_color, CV_8UC1, 50.0 / 5000);
	applyColorMap(depth_color, depth_color, cv::COLORMAP_JET);
	cv::imshow("seg", seg);
	cv::imshow("depth", depth_color);
	cv::waitKey();

	return 0;
}

//int main()
//{
//	TestDemo();
//}
