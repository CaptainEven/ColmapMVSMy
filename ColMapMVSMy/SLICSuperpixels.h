#ifndef SLICSUPERPIXELS_H
#define SLICSUPERPIXELS_H

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc.hpp>

//#include "zthread/Runnable.h"//һ����������Ľӿ�
//#include "zthread/Thread.h"//�߳������������Ĺ��캯����Runnable����ָ��Ϊ����
//#include "zthread/ThreadedExecutor.h"//ִ������Ϊÿ�����񴴽�һ���߳�
//#include "zthread/ConcurrentExecutor.h"//ִ������ÿ�����񶼻�����һ������ʼ֮ǰִ�����
//#include "zthread/Mutex.h"//�������,��������ֹ���ʳ�ͻ
//#include "zthread/Guard.h"//�������ƣ�guradģ�崴������ʱ��acquire�������һ��localable���󣬶��󱻳����ǣ���release�ͷŸ���
//#include "zthread/Condition.h"//ʹ�û����������������Ļ���
//#include "zthread/CountedPtr.h"//���ü���ģ�壬�������ü���Ϊ0ʱdeleteһ������
//using namespace ZThread;

using namespace std;
using namespace cv;

struct xylab{
	float x;
	float y;
	float l;
	float a;
	float b;
};


//�Ƿ���������طָ�ͼ��
const bool bSlicOut = true;

class SLIC
{
public:
	//Ĭ�ϳ����ظ�����
	SLIC(const string &path, const string &outputPath) :path_(path), outPutPath_(outputPath){ k = 1000; m = 10; bGivenStep = false; }
	//���������ظ�����
	SLIC(int &a, int &b, const string &path, const string &outputPath) :k(a), m(b), path_(path), outPutPath_(outputPath){ bGivenStep = false; }
	//����������
	SLIC(float &step, int &b, const string &path, const string &outputPath) :slicStep(step), m(b), path_(path), outPutPath_(outputPath){ bGivenStep = true; }
	virtual ~SLIC(){}

	//��ʼ���۴�����(���ӵ�����)�������������vCC��
	void initClusterCenter(const int &imageIndex);

	//���г����طָ�����������label��
	void performSegmentation(const int &imageIndex);

	//��������xylab��ɫ�ľ���
	float xylabDistance(xylab &, xylab &);

	//�ð�ɫ�߻�������������
	void drawContour(const int &imageIndex);

	//�ð�ɫ�߻�������������
	void drawContour1(const int &imageIndex);

	//��ǿ��ͨ�ԣ���С�ĳ������������ںϣ����������label������vLabelPixelCount��
	void enforceConnectivity(const int &imageIndex);

	//////////////////////////////////////////////
	//�ռ�����iΪ����ڼ���ͼ��
	Mat run(const int &imageIndex);//����label

private:

	int k;//�����صĸ���
	int m;//�����ɫ����
	float S;//�۴����ļ��

	float slicStep;//ÿ�������صĲ���
	bool bGivenStep;//�Ƿ��������

	const string outPutPath_;//�����Ϣ��Ŀ¼
	const string path_;//ͼ��·��

	Mat label;//ÿ�����صı�ǩ(�����ĸ��۴�����)
	vector<int> vLabelPixelCount;//ÿһ���۴���������������������
	vector<xylab> vCC;//�۴�����

	Mat image;//ִ�г����طָ��ͼ��
	Mat imageLab;//Lab��ɫ�ռ�ͼ��
	Mat imageLapcian;//������˹�˲�ͼ��
	Mat showImage;//������ʾ�����صķָ�ͼ��
};

////�����طָ�����
//class SLICThread :public Runnable
//{
//public:
//	SLICThread(SLIC *p, int i) :parent(p), id(i){};
//	virtual ~SLICThread(){}
//
//	//�߳�ִ�к���
//	void run();
//private:
//	SLIC *parent;//Ŀ�곬���طָ���
//	int id;//ͼ������
//};

#endif//SLICSUPERPIXELS_H