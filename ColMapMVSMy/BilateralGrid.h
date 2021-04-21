#ifndef BILATERALGRID_H
#define BILATERALGRID_H

#include <iostream>
#include <vector>

#include <opencv2\core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

Mat bilateralGridFilter(const Mat &low_res_in,const Mat &low_res_out,const Mat &high_res_in, const float s_sigma, const float l_sigma);

class Grid
{
public:
	//把整幅图像转化为一个类似于体素的三维双边网格
	//每行有width个，每列有height个，每竖有depth个网格，每个网格有ceils个float类型的数据
	//这ceile个数据，存储着最终得到的3*4矩阵
	Grid(int width, int height, int depth, int cells)
		:width_(width), height_(height), depth_(depth), cells_(cells),
		data(width*height*depth*cells,0.0f){}
	Grid() :width_(0), height_(0), depth_(0), cells_(0), data(0,0.0f){}

	int width() const { return width_; }
	int height() const { return height_; }
	int depth() const { return depth_; }
	int cells() const { return cells_; }
	int size() const { return data.size(); }

	float& at(int x, int y, int z, int c)
	{
		assert(x >= 0 && x < width_);
		assert(y >= 0 && y < height_);
		assert(z >= 0 && z < depth_);
		assert(c >= 0 && c < cells_);
		//int p = ((x * height_ + y) * depth_ + z) * cells_ + c;
		int p = (z*height_*width_ + y*width_ + x) * cells_ + c;
		return data[p];
	}
	float at(int x, int y, int z, int c)const
	{
		assert(x >= 0 && x < width_);
		assert(y >= 0 && y < height_);
		assert(z >= 0 && z < depth_);
		assert(c >= 0 && c < cells_);
		//int p = ((x * height_ + y) * depth_ + z) * cells_ + c;
		int p = (z*height_*width_ + y*width_ + x) * cells_ + c;
		return data[p];
	}
	void set(int x, int y, int z, int c,float value)
	{
		assert(x >= 0 && x < width_);
		assert(y >= 0 && y < height_);
		assert(z >= 0 && z < depth_);
		assert(c >= 0 && c < cells_);
		//int p = ((x * height_ + y) * depth_ + z) * cells_ + c;
		int p = (z*height_*width_ + y*width_ + x) * cells_ + c;
		data[p] = value;
	}

	float& operator()(int x, int y, int z, int c)
	{
		return at(x, y, z, c);
	}
	float operator()(int x, int y, int z, int c)const
	{
		return at(x, y, z, c);
	}


	inline int clamp(int val, int low, int high) const
	{
		return val <= low ? low : (val >= high ? high : val);
	}
	float clamp_at(int x, int y, int z, int c)const
	{
		x = clamp(x, 0, width_ - 1);
		y = clamp(y, 0, height_ - 1);
		z = clamp(z, 0, depth_ - 1);
		c = clamp(c, 0, cells_ - 1);
		return at(x, y, z, c);
	}
private:
	//class 默认是私有的，struct默认的是公有的
	std::vector<float> data;
	int width_, height_, depth_, cells_;
};


#endif//BILATERALGRID_H