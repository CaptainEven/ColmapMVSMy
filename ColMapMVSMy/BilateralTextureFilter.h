#ifndef BILATERALTEXTUREFILTER_H
#define BILATERALTEXTUREFILTER_H


#include<opencv2/opencv.hpp>
#include<vector>

#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace std;


////************************************************************************////
////个人实现的双边滤波算法
//计算高斯滤波核——空间
void creatGaussianKernel_Space(Mat& dst, int d, double sigmaSpace)
{
	dst.create(Size(d, d), CV_32FC1);
	int mid = d / 2;
	double sigmaSpaceSquare = sigmaSpace*sigmaSpace;
	for (int i = 0; i < d; i++)
	{
		float* p = dst.ptr<float>(i);
		for (int j = 0; j < d; j++)
		{
			p[j] = (i - mid)*(i - mid) + (j - mid)* (j - mid);
			p[j] /= sigmaSpaceSquare;
		}
	}
	dst = dst*(-0.5);
	exp(dst, dst);
}
//计算高斯滤波核——颜色
void creatGaussianKernel_Color(Mat& src, Mat& dst, Mat& Lut, int d)
{
	int nr_channels = src.channels();
	dst.create(Size(d, d), CV_32FC1);
	int mid = d / 2;
	if (nr_channels == 1)
	{
		Mat tmp;
		src.copyTo(tmp);
		tmp = abs(tmp - tmp.at<uchar>(mid, mid));
		LUT(tmp, Lut, dst);//查找表
	}
	else
	{
		Mat tmp = abs(src - src.at<Vec3b>(mid, mid));
		LUT(tmp, Lut, tmp);
		vector<Mat> v;
		split(tmp, v);
		dst = v[0].mul(v[1]).mul(v[2]);//这里是相乘，不是相加。指数相乘，底不变，指数相加
	}
}
//计算查找表，加速颜色高斯滤波核计算
void createColorLUT(Mat &Lut, double sigmaColor)
{
	//建立查找表
	double sigmaColorSquare = sigmaColor*sigmaColor;
	float lut[256];
	for (int i = 0; i < 256; i++)
	{
		float tmp;
		tmp = (i*i) / sigmaColorSquare;
		tmp *= (-0.5);
		lut[i] = exp(tmp);
	}
	//Mat Lut(1, 256, CV_32FC1, lut);
	Lut.create(Size(1, 256), CV_32FC1);
	float* p = Lut.ptr<float>(0);
	for (int i = 0; i < 256; i++)
	{
		p[i] = lut[i];
	}
}
//双边滤波,src为单通道灰度图像
void BilateralFilter(const Mat& src, Mat& dst, int d, double sigmaSpace, double sigmaColor)
{
	if (d <= 0)
	{
		d = 3.0*sigmaSpace;
		if ((d & 1) == 0)//使得直接为奇数
			d += 1;
	}
	//解决边界问题
	int mid = d / 2;
	Mat src_border;
	copyMakeBorder(src, src_border, mid, mid, mid, mid, BORDER_REPLICATE);//边界重复扩充


	//原图像归一化
	Mat src_f;
	src_border.convertTo(src_f, CV_32FC1, 1.0 / 255, 0);
	dst.create(src_border.size(), CV_32FC1);//输出和输入都是单通道，归一化到【0，1】

	Mat kernel_space;//spatial kernel
	Mat kernel_color;//range kernel
	//computing spatial kernel
	creatGaussianKernel_Space(kernel_space, d, sigmaSpace);
	//创建查找表
	Mat Lut;
	createColorLUT(Lut, sigmaColor);

	int height = src_border.rows;
	int width = src_border.cols;
	Mat imageRoi_src;
	Mat imageRoi_f;
	Mat kernel;//spatial and range kernel sum
	for (int i = mid; i < height - mid; i++)
	{
		for (int j = mid; j < width - mid; j++)
		{
			//仅作计算range kernel用
			imageRoi_src = src_border(Range(i - mid, i + mid + 1), Range(j - mid, j + mid + 1));
			//用着计算最终滤波后的像素值
			imageRoi_f = src_f(Range(i - mid, i + mid + 1), Range(j - mid, j + mid + 1));
			//computing range kernel
			creatGaussianKernel_Color(imageRoi_src, kernel_color, Lut, d);
			//computing kernel
			multiply(kernel_space, kernel_color, kernel);
			//*****************************************************
			//单通道处理
			Mat tmp = kernel.mul(imageRoi_f);//逐像素的 权重与像素值相乘
			tmp /= sum(kernel)[0];//逐像素的 除以权重和
			dst.ptr<float>(i)[j] = sum(tmp)[0];//累加所有像素值	
			//*****************************************************

			////*****************************************************
			////分别对需要滤波的图像三个通道，利用同一kernel进行滤波处理
			//float* p = dst.ptr<float>(i);
			//vector<Mat> v;
			//split(imageRoi_f, v);
			//Mat tmp;
			//for (int k = 0; k < 3; k++)
			//{
			//	tmp = kernel.mul(v[k]);//逐像素的 权重与像素值相乘
			//	tmp /= sum(kernel)[0];//逐像素的 除以权重和
			//	p[3 * j + k] = sum(tmp)[0];//累加所有像素值
			//}
			////*****************************************************
		}
	}
	dst.convertTo(dst, CV_8UC1, 255, 0);//最终图像由CV_32F转化为CV_8U

	//去掉dst的边界
	dst = dst(Range(mid, height - mid), Range(mid, width - mid));
}
//联合双边滤波,src为单通道归一化浮点数[0,1]
void JointBilateralFilter_me(const Mat& src, const Mat &joint, Mat& dst, int d, double sigmaSpace, double sigmaColor)
{
	if (d <= 0)
	{
		d = 3.0*sigmaSpace;
		if ((d & 1) == 0)//使得直接为奇数
			d += 1;
	}
	//解决边界问题
	int mid = d / 2;
	Mat src_border;
	copyMakeBorder(src, src_border, mid, mid, mid, mid, BORDER_REPLICATE);//边界重复扩充
	Mat joint_border;
	copyMakeBorder(joint, joint_border, mid, mid, mid, mid, BORDER_REPLICATE);
	Mat joint_uchar;
	joint_border.convertTo(joint_uchar, CV_8UC1, 255);
	dst.create(src_border.size(), CV_32FC1);//输出和输入都是单通道，归一化到【0，1】

	Mat kernel_space;//spatial kernel
	Mat kernel_color;//range kernel
	//computing spatial kernel
	creatGaussianKernel_Space(kernel_space, d, sigmaSpace);
	//创建查找表
	Mat Lut;
	createColorLUT(Lut, sigmaColor);

	int height = src_border.rows;
	int width = src_border.cols;
	Mat imageRoi_src;
	Mat imageRoi_f;
	Mat kernel;//spatial and range kernel sum
	for (int i = mid; i < height - mid; i++)
	{
		for (int j = mid; j < width - mid; j++)
		{
			//联合图像用作计算range kernel
			imageRoi_src = joint_uchar(Range(i - mid, i + mid + 1), Range(j - mid, j + mid + 1));
			//用着计算最终滤波后的像素值
			imageRoi_f = src_border(Range(i - mid, i + mid + 1), Range(j - mid, j + mid + 1));
			//computing range kernel
			creatGaussianKernel_Color(imageRoi_src, kernel_color, Lut, d);
			//computing kernel
			multiply(kernel_space, kernel_color, kernel);
			//*****************************************************
			//单通道处理
			Mat tmp = kernel.mul(imageRoi_f);//逐像素的 权重与像素值相乘
			tmp /= sum(kernel)[0];//逐像素的 除以权重和
			dst.ptr<float>(i)[j] = sum(tmp)[0];//累加所有像素值	
			//*****************************************************

			////*****************************************************
			////分别对需要滤波的图像三个通道，利用同一kernel进行滤波处理
			//float* p = dst.ptr<float>(i);
			//vector<Mat> v;
			//split(imageRoi_f, v);
			//Mat tmp;
			//for (int k = 0; k < 3; k++)
			//{
			//	tmp = kernel.mul(v[k]);//逐像素的 权重与像素值相乘
			//	tmp /= sum(kernel)[0];//逐像素的 除以权重和
			//	p[3 * j + k] = sum(tmp)[0];//累加所有像素值
			//}
			////*****************************************************
		}
	}
	//去掉dst的边界
	dst = dst(Range(mid, height - mid), Range(mid, width - mid));
}
////************************************************************************////


////************************************************************************////
////快速双边滤波
//"A Fast Approachmation of the Bilateral Filter Using a Signal Processsing Approach"
//类似于双边网格
//src是单通道图像,且归一化为【0，1】
void fastBilateralFilter(const cv::Mat &src, cv::Mat &dst, const float sigma_space, const float sigma_range)
{
	assert(src.type() == CV_32FC1);

	const int width = src.cols;
	const int height = src.rows;
	const int padding_xy = 2;//扩展三维网格尺度，防止边界越界
	const int padding_z = 2;

	double src_min, src_max;
	minMaxLoc(src, &src_min, &src_max);//求取图像中最大和最小值
	const double src_delta = src_max - src_min;

	//降采样后图像尺度
	const int small_width = static_cast<int>((width - 1) / sigma_space) + 1 + 2 * padding_xy;
	const int small_height = static_cast<int>((height - 1) / sigma_space) + 1 + 2 * padding_xy;
	const int small_depth = static_cast<int>(src_delta / sigma_range) + 1 + 2 * padding_z;

	//降采样
#ifdef _DEBUG 
	cout << "downsampling..." << endl;
#endif

	int mat_size[3] = { small_depth, small_height, small_width };//(高，行，列)
	cv::Mat grid(3, mat_size, CV_32FC2, Scalar::all(0.0));
	for (int x = 0; x < width; x++)//列
	{
		const int small_x = static_cast<int>(x / sigma_space + 0.5) + padding_xy;
		for (int y = 0; y < height; y++)//行
		{
			const float z = src.ptr<float>(y)[x] - src_min;

			const int small_y = static_cast<int>(y / sigma_space + 0.5) + padding_xy;
			const int small_z = static_cast<int>(z / sigma_range + 0.5) + padding_z;

			//grid.at<Vec2f>(small_z, small_y, small_x)[0] += src.ptr(x)[y];
			//grid.at<Vec2f>(small_z, small_y, small_x)[1] += 1.0f;

			float *ptr = (float*)(grid.data + grid.step[0] * small_z + grid.step[1] * small_y + grid.step[2] * small_x);
			ptr[0] += src.ptr<float>(y)[x]; ptr[1] += 1.0;
		}
	}

	//卷积
#ifdef _DEBUG
	cout << "convolution..." << endl;
#endif

	cv::Mat buffer(3, mat_size, CV_32FC2, Scalar::all(0.0));
	//x,y,z维度,step[0-2]每一维元素字节大小(面，线，点),step1(0-2)每一维元素通道数
	for (int dim = 2; dim >= 0; dim--)
	{
		int offset = grid.step1(dim);
		for (int iter = 0; iter < 2; iter++)//每个维度两次迭代
		{
			cv::swap(buffer, grid);
			for (int z = 1, z_end = small_depth - 1; z < z_end; z++)//z:面
			{
				for (int y = 1, y_end = small_height - 1; y < y_end; y++)//y:线
				{
					float *g_ptr = (float*)(grid.data + grid.step[0] * z + grid.step[1] * y + grid.step[2]);
					float *b_ptr = (float*)(buffer.data + buffer.step[0] * z + buffer.step[1] * y + buffer.step[2]);
					for (int x = 1, x_end = small_width - 1; x < x_end; x++, g_ptr++, b_ptr++)//x:点
					{
						*g_ptr = (*(b_ptr - offset) + *(b_ptr + offset) + 2 * (*b_ptr)) / 4.0;
						g_ptr++, b_ptr++;
						*g_ptr = (*(b_ptr - offset) + *(b_ptr + offset) + 2 * (*b_ptr)) / 4.0;
					}//end of x
				}//end of y
			}//end of z
		}//end of iter
	}//end of dim

	//非线性处理
#ifdef _DEBUG
	cout << "nonlinearities..." << endl;
#endif

	dst.create(src.size(), src.type());
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			const float z = src.ptr<float>(y)[x] - src_min;
			//三线性插值
			float xf = x / sigma_space + padding_xy;
			float yf = y / sigma_space + padding_xy;
			float zf = z / sigma_range + padding_z;
			const int xi = floor(xf);//小于等于x的最大整数
			const int yi = floor(yf);
			const int zi = floor(zf);
			xf -= xi;//小数部分
			yf -= yi;
			zf -= zi;
			cv::Vec2f trilinearInterpolation =
				grid.at<Vec2f>(zi, yi, xi)             *(1 - zf)*(1 - yf)*(1 - xf) +
				grid.at<Vec2f>(zi, yi, xi + 1)         *(1 - zf)*(1 - yf)*xf +
				grid.at<Vec2f>(zi, yi + 1, xi)         *(1 - zf)*yf      *(1 - xf) +
				grid.at<Vec2f>(zi, yi + 1, xi + 1)     *(1 - zf)*yf      *xf +
				grid.at<Vec2f>(zi + 1, yi, xi)         *zf      *(1 - yf)*(1 - xf) +
				grid.at<Vec2f>(zi + 1, yi, xi + 1)     *zf      *(1 - yf)*xf +
				grid.at<Vec2f>(zi + 1, yi + 1, xi)     *zf      *yf      *(1 - xf) +
				grid.at<Vec2f>(zi + 1, yi + 1, xi + 1) *zf      *yf      *xf;

			float result = trilinearInterpolation[0] / trilinearInterpolation[1];

			dst.ptr<float>(y)[x] = result;
			//dst.ptr<float>(y)[x] = result < 0 ? 0 : (result>1 ? 1 : result);
		}
	}

}
////************************************************************************////


////************************************************************************////
////双边纹理滤波
////所有的灰色图像都归一化到[0,1)(也就是CV_32F类型)
//由输入灰度图像计算色调范围图像，k必须是奇数
void creatTonalRangeImage(const Mat& src, Mat& TonalImage, const int k)
{
	assert(src.type() == CV_32FC1);

	TonalImage.create(src.size(), src.type());
	Mat imageRoi;
	double MaxV, MinV;
	int height = src.rows;
	int width = src.cols;
	int d = k / 2;
	for (int i = d; i < height - d; i++)
	{
		float* p = TonalImage.ptr<float>(i);
		for (int j = d; j < width - d; j++)
		{
			imageRoi = src(Range(i - d, i + d + 1), Range(j - d, j + d + 1));//range包括左边界，不包括右边界！！！
			minMaxLoc(imageRoi,&MinV,&MaxV);//选择最小和最大值
			p[j] = MaxV - MinV;
		}
	}
	imageRoi = TonalImage(Range(d,height-d),Range(d,width-d));//左行右列
	imageRoi.copyTo(TonalImage);//在拷贝前，目标图像会被重新分配内存
	copyMakeBorder(TonalImage, TonalImage, d, d, d, d, IPL_BORDER_REPLICATE);
}
//由输入灰色图像计算图像梯度幅度
void creatGradientMagnitudeImage(const Mat& src, Mat& GradImage)
{
	assert(src.type() == CV_32FC1);

	Mat GradImage_x, GradImage_y;
	Sobel(src, GradImage_x, CV_32F, 1, 0);
	Sobel(src, GradImage_y, CV_32F, 0, 1);
	magnitude(GradImage_x, GradImage_y, GradImage);

	//Mat dir;//梯度方向
	//cartToPolar(GradImage_x, GradImage_y, GradImage,dir);//计算L2范式及梯度方向

	//pow(GradImage_x, 2, GradImage_x);
	//pow(GradImage_y, 2, GradImage_y);
	//sqrt(GradImage_x + GradImage_y, GradImage);
}
//由输入梯度图像和色调范围图像计算mRTV
void creatmRTV(const Mat& GradImage, const Mat& TonoalImage, Mat& mRTV, const int k)
{
	mRTV.create(GradImage.size(), GradImage.type());
	Mat imageRoi;
	double MaxV, MinV,SumV;
	int height = GradImage.rows;
	int width = GradImage.cols;
	int d = k / 2;
	for (int i = d; i < height - d; i++)
	{
		float* p = mRTV.ptr<float>(i);
		const float* tonal = TonoalImage.ptr<float>(i);
		for (int j = d; j < width - d; j++)
		{
			imageRoi = GradImage(Range(i - d, i + d + 1), Range(j - d, j + d + 1));//range包括左边界，不包括右边界！！！
			minMaxLoc(imageRoi, &MinV, &MaxV);
			SumV = sum(imageRoi)[0];
			p[j] = tonal[j]* MaxV/(SumV+0.000000001);
		}
	}
	imageRoi = mRTV(Range(d, height - d), Range(d, width - d));//左行右列
	imageRoi.copyTo(mRTV);
	copyMakeBorder(mRTV, mRTV, d, d, d, d, IPL_BORDER_REPLICATE);
}
//由输入mRTV，B，计算G和alpha(灰色图像，单通道)
void creatGandAlpha_Gray(const Mat& mRTV,const Mat& B, Mat& G, Mat& alpha, const int k)
{
	G.create(B.size(), B.type());
	alpha.create(B.size(), B.type());
	Mat imageRoi_mrtv;
	Mat imageRoi_b;
	double MaxV, MinV;
	Point MaxL, MinL;
	int height = B.rows;
	int width = B.cols;
	int d = k / 2;
	for (int i = d; i < height - d; i++)
	{
		const float* p1 = mRTV.ptr<float>(i);//mRTV
		float* p2 = G.ptr<float>(i);//G
		float* p3 = alpha.ptr<float>(i);//alpha
		for (int j = d; j < width - d; j++)
		{
			imageRoi_mrtv = mRTV(Range(i - d, i + d + 1), Range(j - d, j + d + 1));
			imageRoi_b = B(Range(i - d, i + d + 1), Range(j - d, j + d + 1));
			minMaxLoc(imageRoi_mrtv, &MinV, &MaxV, &MinL, &MaxL);
			p2[j] = imageRoi_b.at<float>(MinL);//G图像,为具有最小mRTV处的B

			float tmp = -5.0*k*(p1[j] - (float)MinV);//p-q
			tmp = 1 / (exp(tmp) + 1) - 0.5;
			p3[j] = 2 * tmp;//alpha图像
		}
	}

}
//由输入mRTV，B，计算G和alpha(彩色图像，三通道)
void creatGandAlpha_Color(const Mat& mRTV,const Mat& B, Mat& G, Mat& alpha, const int k)
{
	G.create(B.size(), B.type());
	alpha.create(B.size(), CV_32FC1);
	Mat imageRoi_mrtv;
	Mat imageRoi_b;
	double MaxV, MinV;
	Point MaxL, MinL;
	int height = B.rows;
	int width = B.cols;
	int d = k / 2;
	for (int i = d; i < height - d; i++)
	{
		const float* p1 = mRTV.ptr<float>(i);
		float* p2 = G.ptr<float>(i);
		float* p3 = alpha.ptr<float>(i);
		for (int j = d; j < width - d; j++)
		{
			imageRoi_mrtv = mRTV(Range(i - d, i + d + 1), Range(j - d, j + d + 1));
			imageRoi_b = B(Range(i - d, i + d + 1), Range(j - d, j + d + 1));
			minMaxLoc(imageRoi_mrtv, &MinV, &MaxV, &MinL, &MaxL);
			p2[3 * j] = imageRoi_b.at<Vec3f>(MinL)[0];//G,三通道
			p2[3 * j+1] = imageRoi_b.at<Vec3f>(MinL)[1];
			p2[3 * j+2] = imageRoi_b.at<Vec3f>(MinL)[2];

			float tmp = p1[j] - MinV;//是否需要转换到1-255！？
			tmp = 1 / (exp(tmp*(-5.0*k)) + 1) - 0.5;//此处注意
			p3[j] = 2 * tmp;
		}
	}

}
//由输入彩色图像，计算引导图像(灰色)
void CreatGuidanceImage_Gray(const Mat& src,Mat& dst,const int k)
{
	//所求的的Guidance Image为灰度图像！！
	Mat image_gray;
	Mat B;
	Mat TonalRangeImage;
	Mat GradientMagnitudeImage;
	Mat mRTV;
	Mat G;
	Mat Alpha;
	//将彩色图像转换到灰度图像，以后的所有操作都将基于灰度图像操作，最终生成的guidance image也为灰度图像
	assert(src.type() == CV_32FC1);
	src.copyTo(image_gray);

	//将图像归一化0-1之内,谨记最终要转化回来！！！
	//image_gray.convertTo(image_gray, CV_32FC1, 1.0 / 255, 0);

	//计算B,即K*K邻域内均值
	blur(image_gray,B,Size(k,k));

	//计算获得Tonal Image，即K*K邻域内最大像素值与最小像素值的差值大小
	creatTonalRangeImage(image_gray,TonalRangeImage,k);

	//计算图像的梯度图像
	creatGradientMagnitudeImage(image_gray,GradientMagnitudeImage);

	//计算mRTV
	creatmRTV(GradientMagnitudeImage, TonalRangeImage, mRTV, k);

	//计算alpha和G
	creatGandAlpha_Gray(mRTV, B, G, Alpha, k);

	//利用alpha权重插值B和G
	dst = Alpha.mul(G)+(1-Alpha).mul(B);

	int height = dst.rows;
	int width = dst.cols;
	int d = k / 2;
	Mat imageRoi = dst(Range(d, height - d), Range(d, width - d));//左行右列
	imageRoi.copyTo(dst);
	copyMakeBorder(dst, dst, d, d, d, d, BORDER_REPLICATE);

	//将Guidance Image转化到0-255
	//dst.convertTo(dst,CV_8UC1,255,0);


	////调试用
	//{
	//	double minV, maxV;
	//	minMaxLoc(B, &minV, &maxV);
	//	cout << "B: " << minV << " " << maxV << endl;
	//	minMaxLoc(TonalRangeImage, &minV, &maxV);
	//	cout << "TonalRangeImage: " << minV << " " << maxV << endl;
	//	minMaxLoc(GradientMagnitudeImage, &minV, &maxV);
	//	cout << "Gradient: " << minV << " " << maxV << endl;
	//	minMaxLoc(mRTV, &minV, &maxV);
	//	cout << "mRTV: " << minV << " " << maxV << endl;
	//	minMaxLoc(G, &minV, &maxV);
	//	cout << "G: " << (float)minV << " " << (float)maxV << endl;
	//	minMaxLoc(Alpha, &minV, &maxV);
	//	cout << "Alpha: " << (float)minV << " " << (float)maxV << endl;
	//}
	//B.convertTo(B, CV_8UC1, 255, 0);
	//TonalRangeImage.convertTo(TonalRangeImage, CV_8UC1, 255, 0);
	//GradientMagnitudeImage.convertTo(GradientMagnitudeImage, CV_8UC1, 255, 0);
	//mRTV.convertTo(mRTV, CV_8UC1, 255 * k, 0);
	//Alpha.convertTo(Alpha, CV_8UC1, 255, 0);
	//G.convertTo(G, CV_8UC1, 255, 0);
	//imwrite("W_1Blur.jpg", B);
	//imwrite("W_2Tonal.jpg", TonalRangeImage);
	//imwrite("W_3Gradient.jpg", GradientMagnitudeImage);
	//imwrite("W_4mRTV.jpg", mRTV);
	//imwrite("W_5Alpha.jpg", Alpha);
	//imwrite("W_6G.jpg", G);
	//imwrite("W_7dst.jpg", dst);
}
//由输入彩色图像，计算引导图像（彩色）
void CreatGuidanceImage_Color(const Mat& src, Mat& dst, int k)
{
	//所求的的Guidance Image为彩色图像！！
	Mat imageNorm;
	Mat B;
	Mat TonalImage_r, TonalImage_g, TonalImage_b;
	Mat GradImage_r, GradImage_g, GradImage_b;
	Mat mRTV_r, mRTV_g, mRTV_b, mRTV;
	Mat G;
	Mat Alpha;

	//确保原图像是归一化后
	assert(src.type() == CV_32FC3);
	//归一化
	//src.convertTo(imageNorm, CV_32FC3, 1.0 / 255, 0);
	src.copyTo(imageNorm);

	//计算B,即K*K邻域内均值,B为三通道彩色图像！
	blur(imageNorm, B, Size(k, k));

	//######################################################################
	//分别求取三通道的mRTV
	//将图像划分为三通道的灰度图像
	vector<Mat> v;
	split(imageNorm, v);
	//v[0]为B通道；v[1]为G通道；v[2]为R通道分量
	//求取B通道mRTV
	creatTonalRangeImage(v[0], TonalImage_b, k);
	creatGradientMagnitudeImage(v[0], GradImage_b);
	creatmRTV(GradImage_b, TonalImage_b, mRTV_b, k);
	//求取G通道mRTV
	creatTonalRangeImage(v[1], TonalImage_g, k);
	creatGradientMagnitudeImage(v[1], GradImage_g);
	creatmRTV(GradImage_g, TonalImage_g, mRTV_g, k);
	//求取R通道mRTV
	creatTonalRangeImage(v[2], TonalImage_r, k);
	creatGradientMagnitudeImage(v[2], GradImage_r);
	creatmRTV(GradImage_r, TonalImage_r, mRTV_r, k);
	//求取总的mRTV
	mRTV = mRTV_r + mRTV_g + mRTV_b;
	//######################################################################

	creatGandAlpha_Color(mRTV, B, G, Alpha, k);

	vector<Mat> v2, v3;
	split(G, v2);
	split(B, v3);
	for (int i = 0; i < 3; i++)//B,G两通道或者B,G,R三通道
	{
		v2[i]=(Alpha.mul(v2[i]) + (1 - Alpha).mul(v3[i]));
	}
	merge(v2, dst);

	int height = dst.rows;
	int width = dst.cols;
	int d = k / 2;
	Mat imageRoi = dst(Range(d, height - d), Range(d, width - d));//左行右列
	imageRoi.copyTo(dst);
	copyMakeBorder(dst, dst, d, d, d, d, IPL_BORDER_REPLICATE);

	//将Guidance Image转化到0-255;
	//dst.convertTo(dst, CV_8UC3, 255, 0);
}
////双边纹理滤波
////src为8-bit Uchar 或者 32-bit Float(且是归一化的！！！)
void BilateralTextureFilter(const Mat &src, Mat &dst, const int k, const int num_iter)
{
	//彩色图像
	if (src.channels() == 3)
	{
		Mat src_f;
		//转化为32-float ，且归一化
		if (src.type() == CV_8UC3)
			src.convertTo(src_f, CV_32FC3, 1.0 / 255, 0);
		else
			src.copyTo(src_f);
		Mat guide;
		for (int i = 0; i < num_iter; i++)
		{
			CreatGuidanceImage_Color(src_f, guide, k);
			ximgproc::jointBilateralFilter(guide, src_f, dst, 2 * k - 1, 0.05*sqrtf(3), k - 1);
			dst.copyTo(src_f);
		}
		if (src.type() == CV_8UC3)
			dst.convertTo(dst, CV_8UC3, 255, 0);
	}
	//灰色图像（单通道）
	else
	{
		Mat src_f;
		if (src.type() == CV_8UC1)
			src.convertTo(src_f, CV_32FC1, 1.0 / 255, 0);
		else
			src.copyTo(src_f);
		Mat guide;
		for (int i = 0; i < num_iter; i++)
		{
			CreatGuidanceImage_Gray(src_f, guide, k);
			ximgproc::jointBilateralFilter(guide, src_f, dst, 2 * k - 1, 0.05, k - 1);
			src_f = dst;
		}
		if (src.type() == CV_8UC1)
			dst.convertTo(dst, CV_8UC1, 255, 0);
	}
}
////**************************************************************************////


////*************************************************************************////
//利用多尺度双边滤波分解，进行细节增强
//核大小通常为sigma的6*sigma + 1。因为离中心点3*sigma大小之外的系数与中点的系数只比非常小，
//可以认为此之外的点与中心点没有任何联系
//默认参数  尺度scales=5。0，1，2，3，4，5
//sigmaSpace=2, sigmaColor=0.1【0，1】
void MultiScalesDetailEnhance(const Mat &src, Mat &dst, const int scales, float sigmaSpace, float sigmaColor)
{

	//Lab颜色空间
	Mat src_lab;
	cvtColor(src, src_lab, COLOR_BGR2Lab);
	vector<Mat> lab_channels;
	split(src_lab, lab_channels);
	Mat src_f;
	lab_channels[0].convertTo(src_f, CV_32FC1, 1.0 / 255);

	//YUV颜色空间
	//Mat src_yuv;
	//cvtColor(src, src_yuv, COLOR_BGR2YUV);//RGB颜色空间转为YUV
	//vector<Mat> yuv_channels;
	//split(src_yuv, yuv_channels);//三通道分离

	//Mat src_f;
	//yuv_channels[0].convertTo(yuv_channels[0], CV_32FC1);//Y通道转为float型
	//yuv_channels[0] += 1.0f;
	//log(yuv_channels[0], src_f);//取自然数的对数
	//{double v_max, v_min;
	//minMaxLoc(src_f, &v_min, &v_max);
	//cout << "src_f: " << v_min << " " << v_max << endl;
	//}
	//src_f.convertTo(src_f, CV_32FC1, 1.0 / log(255.0));
	//normalize(src_f, src_f, 0, 1, NORM_MINMAX);//归一化到0-1
	//convertScaleAbs(src_f, src_f, 255);//dst= saturate_cast<uchar>(|src* alpha + beta|)
	//imwrite("PinkRose_logY0.png", src_f);
	//////src_f.convertTo(src_f, CV_8UC1, 255);//dst = saturate_cast<rType>( src* alpha + beta)
	//yuv_channels[0] = src_f;
	//merge(yuv_channels, src_f);
	//cvtColor(src_f, src_f, COLOR_YUV2BGR);
	//imwrite("PinkRose_logY.png", src_f);

	//Mat src_gray, src_f;
	//cvtColor(src, src_gray, CV_BGR2GRAY);//灰色图像
	//src_gray.convertTo(src_f, CV_32FC1, 1.0 / 255, 0);//归一化【0，1】

	const int width = src.cols;
	const int height = src.rows;

	vector<Mat> filteredImages, detailImages, gradientImages, weightImages;
	filteredImages.resize(scales + 1);
	detailImages.resize(scales);
	gradientImages.resize(scales);
	weightImages.resize(scales, Mat_<float>(src_f.size()));
	filteredImages[0] = src_f;

	////计算多尺度滤波图像
	for (int i = 1; i <= scales; i++)
	{
		//更新每个尺度的spatial and range kernel
		if (i == 1)
			sigmaSpace *= sqrtf(3.0);
		else
			sigmaSpace *= powf(2, i - 1);

		sigmaColor /= 2.0;

		//计算每个尺度的滤波图像
		string p1, p2, p3;
		p1 = to_string(i) + "_1.png";
		p2 = to_string(i) + "_2.png";
		p3 = to_string(i) + "_3.png";
		Mat result;
		clock_t start_t, end_t;
		start_t = clock();
		//BilateralFilter(filteredImages[i - 1], filteredImages[i], d, sigmaSpace, sigmaColor);//自己实现的
		//end_t = clock();
		//cout << "Mine:"<<(float)(end_t - start_t) / CLOCKS_PER_SEC << "s " << endl;
		//imwrite(p1, filteredImages[i]);
		//bilateralFilter(filteredImages[i - 1], filteredImages[i], 0, sigmaColor, sigmaSpace);//opencv自带的
		//end_t = clock();
		//cout << "Opencv:"<<(float)(end_t - start_t) / CLOCKS_PER_SEC << "s " << endl;
		//filteredImages[i].convertTo(result, CV_8UC1, 255, 0);
		//imwrite(p2, result);
		fastBilateralFilter(filteredImages[i - 1], filteredImages[i], sigmaSpace, sigmaColor);//快速双边滤波
		end_t = clock();
		cout << "FastBF:" << (float)(end_t - start_t) / CLOCKS_PER_SEC << "s " << endl << endl;
		filteredImages[i].convertTo(result, CV_8UC1, 255, 0);
		imwrite(p3, result);
	}

	////分离细节图像
	for (int i = 0; i < scales; i++)
	{
		detailImages[i] = filteredImages[i + 1] - filteredImages[i];

		double v_max, v_min;
		minMaxLoc(detailImages[i], &v_min, &v_max);
		cout << "d_" << to_string(i) << ":" << v_min << " " << v_max << endl;
		Mat result;
		convertScaleAbs(detailImages[i], result, 2550);
		string path = "d_" + to_string(i) + ".png";
		imwrite(path, result);
	}

	////计算细节图像权重
	for (int i = 0; i < scales; i++)
	{
		//计算图像的梯度
		Mat GradImage_x, GradImage_y;
		Sobel(filteredImages[i + 1], GradImage_x, CV_32F, 1, 0);
		Sobel(filteredImages[i + 1], GradImage_y, CV_32F, 0, 1);
		magnitude(GradImage_x, GradImage_y, gradientImages[i]);
		{double v_max, v_min;
		minMaxLoc(gradientImages[i], &v_min, &v_max);
		cout << "GradImage: " << v_min << " " << v_max << endl;
		}
		//Mat dir;//梯度方向
		//cartToPolar(GradImage_x, GradImage_y, GradImage,dir);//计算L2范式及梯度方向
		//pow(GradImage_x, 2, GradImage_x);
		//pow(GradImage_y, 2, GradImage_y);
		//sqrt(GradImage_x + GradImage_y,GradImage);
		//GradImage.convertTo(GradImage, CV_8UC1, 255, 0);
		//imwrite("grad2.png", GradImage);

		//计算权重
		int dx[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
		int dy[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
		float sigma_d = 8;//高斯滤波卷积核
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				//在3*3邻域内寻找最小值
				float min_src = 1;
				for (int k = 0; k < 9; k++)
				{
					if (row + dx[k] >= 0 && row + dx[k] < height && col + dy[k] >= 0 && col + dy[k] < width &&
						filteredImages[i + 1].ptr<float>(row + dx[k])[col + dy[k]] < min_src)
						min_src = filteredImages[i + 1].ptr<float>(row + dx[k])[col + dy[k]];
				}
				float C = gradientImages[i].ptr<float>(row)[col] / (min_src + 0.000000001);

				//计算scales个细节图像权重
				weightImages[i].ptr<float>(row)[col] = exp(abs(detailImages[i].ptr<float>(row)[col]) - C);
			}
		}
		GaussianBlur(weightImages[i], weightImages[i], Size(0, 0), sigma_d);//高斯卷积润滑权重
	}


	////细节放大参数
	Mat Detail(src_f.size(), CV_32FC1, Scalar::all(0.0));
	//float lambda[5] = { 0.75, 0.75, 0.8, 0.95, 0.95 };
	float lambda[5] = { 0.95, 0.95, 0.8, 0.75, 0.75 };
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			float productSum = 0.0, weightSum = 0.0;
			for (int i = 0; i < scales; i++)
			{
				productSum += weightImages[i].ptr<float>(y)[x] *
					(detailImages[i].ptr<float>(y)[x] > 0 ? 1 : -1)*powf(abs(detailImages[i].ptr<float>(y)[x]), lambda[i]);
				weightSum += weightImages[i].ptr<float>(y)[x];
			}
			if (productSum != 0.0)
				Detail.ptr<float>(y)[x] = productSum / weightSum;
		}
	}

	//最终结果图像
	multiply(filteredImages[scales], 0.2f, filteredImages[scales]);
	//multiply(Detail, 3.0f, Detail);
	//multiply(detailImages[0], 3.0f, detailImages[0]);
	//detailImages[0].copyTo(Detail);
	Mat result = filteredImages[scales] + Detail;

	double v_max, v_min;
	minMaxLoc(Detail, &v_min, &v_max);
	cout << "Detail:" << v_min << " " << v_max << endl;
	Mat temp;
	convertScaleAbs(Detail, temp, 255);
	imwrite("detail.png", temp);
	minMaxLoc(result, &v_min, &v_max);
	cout << "Result:" << v_min << " " << v_max << endl;

	int less = 0, mid = 0, more = 0;
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			if (result.ptr<float>(y)[x] < 0)
				less++;
			else if (result.ptr<float>(y)[x] > 1)
				more++;
			else
				mid++;
		}
	}
	cout << less << " " << mid << " " << more << endl;

	//normalize(result, result, 0, 255, NORM_MINMAX);
	convertScaleAbs(result, result, 255);
	//result.convertTo(result, CV_8UC1, 255);
	imwrite("PinkRose_logY0_result.png", result);

	lab_channels[0] = result;
	merge(lab_channels, dst);
	cvtColor(dst, dst, COLOR_Lab2BGR);
	imwrite("PinkRose_logY_result.png", dst);
}
////*************************************************************************////


////**************************************************************************////
//细节增强
//细节增强中用到的保边滤波类型
enum FilterTypes
{
	BilateralFilterType = 1,//双边滤波
	FastBilateralFilterType = 2,//快速双边滤波(双边网格)
	BilateralTextureFilterType = 3,//双边纹理滤波
	DomainTransformType = 4//域转换滤波
};

void DentailEnhance(const Mat &src, Mat &dst, const float sigmaSpace = 10.f, const float sigmaColor = 0.1f, int filterType = BilateralTextureFilterType)
{
	////将图像转化为单通道 浮点类型  归一化[0,1]
	Mat src_f;//32-float
	Mat src_u;//8-bit(uchar)
	Mat src_lab;
	vector<Mat> lab_channels;
	const int num_channels = src.channels();
	if (num_channels == 3)
	{
		cvtColor(src, src_lab, COLOR_BGR2Lab);
		split(src_lab, lab_channels);
		lab_channels[0].copyTo(src_u);//单通道, 8-bit
		lab_channels[0].convertTo(src_f, CV_32FC1, 1.0 / 255, 0);//单通道 32-float
	}
	else
	{
		src.copyTo(src_u);//单通道 8-bit
		src.convertTo(src_f, CV_32FC1, 1.0 / 255, 0);//单通道 32-float
	}

	////对图像进行保边滤波
	//双边滤波
	if (filterType == 1)
	{

		bilateralFilter(src_u, dst, 0, sigmaColor * 255, sigmaSpace);//8-bit 1-channels
		dst.convertTo(dst, CV_32FC1, 1.0 / 255, 0);

		//bilateralFilter(src_lab, dst, 0, sigmaColor * 255, sigmaSpace);//8-bit 3-channels
		//vector<Mat> dst_channels; split(dst, dst_channels);
		//dst_channels[0].convertTo(dst, CV_32FC1, 1.0 / 255, 0);

	}
	//快速双边滤波
	else if (filterType == 2)
	{
		fastBilateralFilter(src_f, dst, sigmaSpace, sigmaColor);//32-float [0,1]
	}
	//双边纹理滤波
	else if (filterType == 3)
	{
		BilateralTextureFilter(src_f, dst, sigmaSpace/2.0, 3);//32-float 1-channels
		//ximgproc::bilateralTextureFilter(src_f, dst, sigmaSpace + 1, 3);

		//BilateralTextureFilter(src_lab, dst, sigmaSpace/2.0, 3);//8-bit 3-channels
		//vector<Mat> dst_channels; split(dst, dst_channels);
		//dst_channels[0].convertTo(dst, CV_32FC1, 1.0 / 255, 0);
	}
	//域转换保边滤波
	else if (filterType == 4)
	{
		if (num_channels != 3)
		{
			vector<Mat> temp = { src_u, src_u, src_u };
			merge(temp, src_lab);
		}
		edgePreservingFilter(src_lab, dst, 1, sigmaSpace, sigmaColor);//8-bit 3-channels
		vector<Mat> dst_channels;
		split(dst, dst_channels);
		dst_channels[0].convertTo(dst, CV_32FC1, 1.0 / 255, 0);
	}

	////分离出细节图像(float型)，并增强（单通道分离和三通道分离差不多，统一用Lab颜色空间L通道分离细节图像）
	Mat detail = src_f - dst;
	multiply(detail, 3.0f, detail);
	dst = dst + detail;
	dst.convertTo(dst, CV_8UC1, 255, 0);

	if (num_channels == 3)
	{
		lab_channels[0] = dst;
		merge(lab_channels, dst);
		cvtColor(dst, dst, COLOR_Lab2BGR);
	}

	//////输出结果
	//if (filterType == 1)
	//{
	//	imwrite("DetailEnhance_bf.png", dst);
	//}
	//else if (filterType == 2)
	//{
	//	imwrite("DetailEnhance_fbf.png", dst);
	//}
	//else if (filterType == 3)
	//{
	//	imwrite("DetailEnhance_btf.png", dst);
	//}
	//else if (filterType == 4)
	//{
	//	imwrite("DetailEnhance_dt.png", dst);
	//}
}
////*************************************************************************////


////*************************************************************************////
//多尺度结构增强
void MultiscaleStructureEnhance(const Mat &src, Mat &dst, const bool useBilateralTextureFilter = true,
	const int scales = 4, const int k = 3, const int num_iter = 3)
{
	////将图像转化为单通道 浮点类型  归一化[0,1]
	Mat src_f;//32-float
	Mat src_u;//8-bit(uchar)
	Mat src_lab;
	vector<Mat> lab_channels;
	const int num_channels = src.channels();
	if (num_channels == 3)
	{
		cvtColor(src, src_lab, COLOR_BGR2Lab);
		split(src_lab, lab_channels);
		lab_channels[0].copyTo(src_u);//单通道, 8-bit
		lab_channels[0].convertTo(src_f, CV_32FC1, 1.0 / 255, 0);//单通道 32-float
	}
	else
	{
		src.copyTo(src_u);//单通道 8-bit
		src.convertTo(src_f, CV_32FC1, 1.0 / 255, 0);//单通道 32-float
	}

	//处理单通道
	vector<Mat> filterdImages, detailImages;
	filterdImages.resize(scales + 1);
	detailImages.resize(scales);
	src_f.copyTo(filterdImages[0]);


	//多尺度纹理滤波
	for (int i = 0; i < scales; i++)
	{
		if (useBilateralTextureFilter)
		{
			//双边纹理滤波   滤波后图像索引1，2，3，。。。scales
			BilateralTextureFilter(filterdImages[i], filterdImages[i + 1], k + 2 * i, num_iter);
		}
		else
		{
			//快速双边滤波
			fastBilateralFilter(filterdImages[i], filterdImages[i + 1], pow(2, i + 1), 0.1);
		}

		//分离出细节图   细节图像索引为0，1，2，。。。scales-1
		detailImages[i] = filterdImages[i] - filterdImages[i + 1];
	}


	//多尺度融合
    Mat result(src.size(), CV_32F, Scalar::all(0.0));
	float detailWeight[4] = { 3.0, 2.5, 2.0, 1.5};//保证尺度在4之内
	float baseWeight[4] = { 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6 };
	for (int i = scales - 1; i >= 0; i--)
	{
		multiply(filterdImages[i+1], baseWeight[i], filterdImages[i+1]);
		multiply(detailImages[i], detailWeight[i], detailImages[i]);

		result = result + filterdImages[i + 1] + detailImages[i];

		//if (i == scales - 1)
		//	result = filterdImages[i + 1] + detailImages[i];
		//else
		//    result = result + detailImages[i];
	}
	result.convertTo(dst, CV_8U, 255, 0);

	if (num_channels == 3)
	{
		lab_channels[0] = dst;
		merge(lab_channels, dst);
		cvtColor(dst, dst, COLOR_Lab2BGR);
	}
}
////************************************************************************////

#endif//BILATERALTEXTUREFILTER_H