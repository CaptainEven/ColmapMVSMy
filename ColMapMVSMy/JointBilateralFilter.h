#ifndef JOINTBILATERALFILTER_H
#define JOINTBILATERALFILTER_H

#include <iostream>
#include <vector>

#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

using namespace std;
using namespace cv;


class JointBilateralFilter
{
public:
	JointBilateralFilter(const Mat &joint,const Mat &src,int d,double sigma_color,double sigma_space,bool isFilter):
		joint_(joint), src_(src), d_(d), sigma_color_(sigma_color), sigma_space_(sigma_space), isFilter_(isFilter)
	{
	}
	virtual ~JointBilateralFilter()
	{
		delete[]space_weight;
		delete[]color_weight;
		delete[]space_ofs_jnt;
		delete[]space_ofs_src;
	}

	bool isInputRight();

	void computerWeight();

	Mat runJBF();

private:
	Mat joint_;//联合图像
	Mat src_;//原图像
	Mat dst_;//最终输出图像
	int d_;//窗口的宽度
	int radius_;//窗口的半径
	double sigma_color_;//高斯颜色差参数
	double sigma_space_;//高斯空间差参数
	bool isFilter_;//是否是进行双边滤波，true为是，false为上采样

	Mat jim_;
	Mat sim_;

	float *color_weight;//颜色差权重
	float *space_weight;//空间差权重
	int *space_ofs_jnt;//联合图像 空间offset索引
	int *space_ofs_src;//原图像   空间offset索引
	int maxk;//窗口中被选中的像素的个数

};


#endif//JOINTBILATERALFILTER_H