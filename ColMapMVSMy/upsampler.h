#ifndef UPSAMPLER_H
#define UPSAMPLER_H

#include <opencv2/core/core.hpp>
#include <memory>
#include <math.h>


class Upsampler {
public:
	Upsampler(int high_width, int high_height, float upsampling_rate)//upsampling_rate > 0
		: high_width_(high_width), high_height_(high_height), upsampling_rate_(upsampling_rate){}
	virtual ~Upsampler() {}

	int getLowWidth()const { return std::round(high_width_ / upsampling_rate_); }
	int getLowHeight()const { return std::round(high_height_ / upsampling_rate_); }
	int getHighWidth() const { return high_width_; }
	int getHighHeight()const { return high_height_; }
	float getUpsamplingRate() const { return upsampling_rate_; }

	virtual cv::Mat upsampling(const cv::Mat& highin, const cv::Mat& lowin, const cv::Mat& lowout) = 0;

private:
	int high_width_, high_height_;
	float upsampling_rate_;

};

//双边网格上采样
class BilateralGuidedUpsampler : public Upsampler
{
public:
	BilateralGuidedUpsampler(int high_width, int high_height, float upsampling_rate,
		float r_sigma = 1.f / 8.f, int s_sigma = 16);
	cv::Mat upsampling(const cv::Mat& highin, const cv::Mat& lowin, const cv::Mat& lowout);

private:
	float r_sigma_;
	int s_sigma_;
};

//联合双边上采样
class JointBilateralUpsamplinger:public Upsampler
{
public:
	JointBilateralUpsamplinger(int high_width, int high_height, float upsampling_rate, int r,
		float r_sigma = 25.f, int s_sigma = 10);
	cv::Mat upsampling(const cv::Mat &joint, const cv::Mat &lowin, const cv::Mat &lowout);
private:
	int radius;
	float sigma_color;
	int sigma_space;
};

#endif