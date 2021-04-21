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
	Mat joint_;//����ͼ��
	Mat src_;//ԭͼ��
	Mat dst_;//�������ͼ��
	int d_;//���ڵĿ��
	int radius_;//���ڵİ뾶
	double sigma_color_;//��˹��ɫ�����
	double sigma_space_;//��˹�ռ�����
	bool isFilter_;//�Ƿ��ǽ���˫���˲���trueΪ�ǣ�falseΪ�ϲ���

	Mat jim_;
	Mat sim_;

	float *color_weight;//��ɫ��Ȩ��
	float *space_weight;//�ռ��Ȩ��
	int *space_ofs_jnt;//����ͼ�� �ռ�offset����
	int *space_ofs_src;//ԭͼ��   �ռ�offset����
	int maxk;//�����б�ѡ�е����صĸ���

};


#endif//JOINTBILATERALFILTER_H