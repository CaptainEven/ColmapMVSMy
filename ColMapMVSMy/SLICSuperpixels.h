#ifndef SLICSUPERPIXELS_H
#define SLICSUPERPIXELS_H

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc.hpp>

//#include "zthread/Runnable.h"//一个定义任务的接口
//#include "zthread/Thread.h"//线程驱动器，它的构造函数以Runnable对象指针为参数
//#include "zthread/ThreadedExecutor.h"//执行器，为每个任务创建一个线程
//#include "zthread/ConcurrentExecutor.h"//执行器，每个任务都会在下一个任务开始之前执行完毕
//#include "zthread/Mutex.h"//互斥机制,互斥锁防止访问冲突
//#include "zthread/Guard.h"//保护机制，gurad模板创建对象时用acquire函数获得一个localable对象，对象被撤销是，用release释放该锁
//#include "zthread/Condition.h"//使用互斥锁允许任务挂起的基类
//#include "zthread/CountedPtr.h"//引用计数模板，并在引用计数为0时delete一个对象
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


//是否输出超像素分割图像
const bool bSlicOut = true;

class SLIC
{
public:
	//默认超像素个数的
	SLIC(const string &path, const string &outputPath) :path_(path), outPutPath_(outputPath){ k = 1000; m = 10; bGivenStep = false; }
	//给定超像素个数的
	SLIC(int &a, int &b, const string &path, const string &outputPath) :k(a), m(b), path_(path), outPutPath_(outputPath){ bGivenStep = false; }
	//给定步长的
	SLIC(float &step, int &b, const string &path, const string &outputPath) :slicStep(step), m(b), path_(path), outPutPath_(outputPath){ bGivenStep = true; }
	virtual ~SLIC(){}

	//初始化聚簇中心(种子点中心)，（结果保存在vCC）
	void initClusterCenter(const int &imageIndex);

	//进行超像素分割，（结果保存在label）
	void performSegmentation(const int &imageIndex);

	//计算两个xylab颜色的距离
	float xylabDistance(xylab &, xylab &);

	//用白色线画出超像素轮廓
	void drawContour(const int &imageIndex);

	//用白色线画出超像素轮廓
	void drawContour1(const int &imageIndex);

	//增强连通性，将小的超像素与邻域融合，（结果更新label，生成vLabelPixelCount）
	void enforceConnectivity(const int &imageIndex);

	//////////////////////////////////////////////
	//终极函数i为输入第几张图像
	Mat run(const int &imageIndex);//返回label

private:

	int k;//超像素的个数
	int m;//最大颜色距离
	float S;//聚簇中心间距

	float slicStep;//每个超像素的步长
	bool bGivenStep;//是否给定步长

	const string outPutPath_;//输出信息的目录
	const string path_;//图像路径

	Mat label;//每个像素的标签(属于哪个聚簇中心)
	vector<int> vLabelPixelCount;//每一个聚簇中心所包含的像素数量
	vector<xylab> vCC;//聚簇中心

	Mat image;//执行超像素分割的图像
	Mat imageLab;//Lab颜色空间图像
	Mat imageLapcian;//拉普拉斯滤波图像
	Mat showImage;//用于显示超像素的分割图像
};

////超像素分割任务
//class SLICThread :public Runnable
//{
//public:
//	SLICThread(SLIC *p, int i) :parent(p), id(i){};
//	virtual ~SLICThread(){}
//
//	//线程执行函数
//	void run();
//private:
//	SLIC *parent;//目标超像素分割类
//	int id;//图像索引
//};

#endif//SLICSUPERPIXELS_H