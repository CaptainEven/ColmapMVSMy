#ifndef FASTBILATERALSOLVERME_H
#define FASTBILATERALSOLVERME_H

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>


#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
//#include<opencv2/ximgproc.hpp>

#include <ctime>
#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_map>

using namespace std;
using namespace cv;


//// class CV_EXPORTS_W FastBilateralSolverFilter : public Algorithm
//class FastBilateralSolverFilter : public Algorithm
//{
//public:
//	CV_WRAP virtual void filter(InputArray src, InputArray confidence, OutputArray dst) = 0;
//};
//
//CV_EXPORTS_W Ptr<FastBilateralSolverFilter> createFastBilateralSolverFilter(InputArray guide, double sigma_spatial = 8.0f, double sigma_luma = 8.0f, double sigma_chroma = 8.0f);
//CV_EXPORTS_W void fastBilateralSolverFilter(InputArray guide, InputArray src, InputArray confidence, OutputArray dst, double sigma_spatial, double sigma_luma, double sigma_chroma);


class FastBilateralSolverFilter 
{
public:

	//static Ptr<FastBilateralSolverFilterImpl> create(InputArray guide, double sigma_spatial, double sigma_luma, double sigma_chroma)
	//{
	//	CV_Assert(guide.type() == CV_8UC3);
	//	FastBilateralSolverFilterImpl *fbs = new FastBilateralSolverFilterImpl();
	//	Mat gui = guide.getMat();
	//	fbs->init(gui, sigma_spatial, sigma_luma, sigma_chroma);
	//	return Ptr<FastBilateralSolverFilterImpl>(fbs);
	//}
	//
	// FastBilateralSolverFilterImpl(){}

	FastBilateralSolverFilter(const cv::Mat &reference, const cv::Mat &target, const cv::Mat &confidence,
		double sigma_spatial=8.f, double sigma_luma=4.f, double sigma_chroma=4.f)
	{
		//滤波图像必须和参考图像大小相同
		CV_Assert(confidence.type() == CV_8UC1 && target.size() == confidence.size());

		grid_param.spatialSigma = sigma_spatial;
		grid_param.lumaSigma = sigma_luma;
		grid_param.chromaSigma = sigma_chroma;
		reference_bgr_ = reference.clone();
		target_ = target.clone();
		confident_ = confidence.clone();
	}

	~FastBilateralSolverFilter(){}

	//void filter(InputArray& src, InputArray& confidence, OutputArray& dst)
	//{
	//
	//	CV_Assert(src.type() == CV_8UC1 && confidence.type() == CV_8UC1 && src.size() == confidence.size());
	//	if (src.rows() != rows || src.cols() != cols)
	//	{
	//		CV_Error(Error::StsBadSize, "Size of the filtered image must be equal to the size of the guide image");
	//		return;
	//	}
	//
	//	dst.create(src.size(), src.type());
	//	Mat tar = src.getMat();
	//	Mat con = confidence.getMat();
	//	Mat out = dst.getMat();
	//
	//	solve(tar, con, out);
	//}

	cv::Mat run();

	cv::Mat run1();

	// protected:
	//void solve(cv::Mat& src, cv::Mat& confidence,cv::Mat &dst);
	//void init(cv::Mat& reference_bgr, double sigma_spatial, double sigma_luma, double sigma_chroma);

	void init();
	void solve(cv::Mat &dst);

	void Splat(Eigen::VectorXf& input, Eigen::VectorXf& dst);
	void Blur(Eigen::VectorXf& input, Eigen::VectorXf& dst);
	void Slice(Eigen::VectorXf& input, Eigen::VectorXf& dst);

private:
	cv::Mat reference_bgr_;
	cv::Mat target_;
	cv::Mat confident_;

	int npixels;
	int nvertices;
	int dim;
	int cols;
	int rows;
	std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > blurs;
	std::vector<int> splat_idx;
	std::vector<std::pair<int, int>> blur_idx;
	Eigen::VectorXf m;
	Eigen::VectorXf n;
	Eigen::SparseMatrix<float, Eigen::ColMajor> blurs_test;
	Eigen::SparseMatrix<float, Eigen::ColMajor> S;
	Eigen::SparseMatrix<float, Eigen::ColMajor> Dn;
	Eigen::SparseMatrix<float, Eigen::ColMajor> Dm;

	struct grid_params
	{
		float spatialSigma;
		float lumaSigma;
		float chromaSigma;
		grid_params()
		{
			spatialSigma = 8.0;
			lumaSigma = 4.0;
			chromaSigma = 4.0;
		}
	};

	struct bs_params
	{
		float lam;
		float A_diag_min;
		float cg_tol;
		int cg_maxiter;
		bs_params()
		{
			lam = 128.0;
			A_diag_min = float(1e-5);
			cg_tol = float(1e-5);
			cg_maxiter = 25;
		}
	};

	grid_params grid_param;
	bs_params bs_param;

};



#endif//FASTBILATERALSOLVERME_H