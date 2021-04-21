#include <opencv2/opencv.hpp>
#include <assert.h>
#include <stdexcept>
#include <math.h>
#include <algorithm>
#include <iomanip>

#include "upsampler.h"


using namespace cv;

inline int clamp(int val, int low, int high)
{
	return val <= low ? low : (val >= high ? high : val);
}

class Voxel 
{
	//class 默认是私有的，struct默认的是公有的
	std::vector<float> data;
	int width_, height_, depth_, cells_;

public:
	//把整幅图像转化为一个类似于体素的三维双边网格
	//每行有width个，每列有height个，每竖有depth个网格，每个网格有ceils个float类型的数据
	//这ceile个数据，存储着最终得到的3*4矩阵
	Voxel(int width, int height, int depth, int cells)
		:width_(width), height_(height), depth_(depth), cells_(cells),
		data(width*height*depth*cells){}

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
	float& operator()(int x, int y, int z, int c)
	{
		return at(x, y, z, c);
	}
	float operator()(int x, int y, int z, int c)const
	{
		return at(x, y, z, c);
	}
	float clamp_at(int x, int y, int z, int c)const
	{
		x = clamp(x, 0, width_ - 1);
		y = clamp(y, 0, height_ - 1);
		z = clamp(z, 0, depth_ - 1);
		c = clamp(c, 0, cells_ - 1);
		return at(x, y, z, c);
	}
};

//输出运算符重载
std::ostream& operator<<(std::ostream& os, const Voxel& voxel)
{
	for (int x = 0; x < voxel.width(); x++) {
		for (int y = 0; y < voxel.height(); y++) {
			for (int z = 0; z < voxel.depth(); z++) {
				os << "(" << std::setw(3) << x << "," << std::setw(3) << y << "," << std::setw(3) << z << ")\t";
				for (int c = 0; c < voxel.cells(); c++) {
					os << std::setw(7) << std::fixed << std::setprecision(3) << voxel(x, y, z, c) << "  ";
				}
				os << std::endl;
			}
		}
	}
	return os;
}


void print2darr(float** a, int m, int n)
{
	std::cout << '[';
	for (int i = 0; i < m; i++) {
		std::string pre;
		for (int j = 0; j < n; j++) {
			std::cout << pre << std::setw(7) << std::fixed << *((float*)a + i*n + j);
			pre = ", ";
		}
		std::cout << (i == m - 1 ? "]" : ";") << std::endl;
	}
	std::cout << std::endl;

}


void voxel2grid(const Voxel& coef, std::vector<float>& row0, std::vector<float>& row1, std::vector<float>& row2)
{
	int w = coef.width();
	int h = coef.height();
	int d = coef.depth();

	int rowsize = w*h*d * 4;
	row0 = std::vector<float>(rowsize);
	row1 = std::vector<float>(rowsize);
	row2 = std::vector<float>(rowsize);

	for (int z = 0; z < d; z++) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int loc = (z*w*h + y*w + x) * 4;
				row0[loc] = coef(x, y, z, 0);
				row0[loc + 1] = coef(x, y, z, 1);
				row0[loc + 2] = coef(x, y, z, 2);
				row0[loc + 3] = coef(x, y, z, 3);

				row1[loc] = coef(x, y, z, 4);
				row1[loc + 1] = coef(x, y, z, 5);
				row1[loc + 2] = coef(x, y, z, 6);
				row1[loc + 3] = coef(x, y, z, 7);

				row2[loc] = coef(x, y, z, 8);
				row2[loc + 1] = coef(x, y, z, 9);
				row2[loc + 2] = coef(x, y, z, 10);
				row2[loc + 3] = coef(x, y, z, 11);
			}
		}
	}
}


//高斯消元法求解方程组的解 Solve Ax = b
template<int M,int N>
void solve(const float A[M][M], const float b[M][N], float x[M][N])
{
	//const int M = 4;
	//const int N = 3;
	float e[M][M + N];
	// fill elimination matrix
	//生成  增广矩阵
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			e[i][j] = A[i][j];
		}
		for (int j = 0; j < N; j++) {
			e[i][j + M] = b[i][j];
		}
	}

	// eliminate lower left左下
	//从第一行开始，每次 乘加运算 消除左下数据
	//消除顺序为，列，从左到右
	for (int k = 0; k < M - 1; k++) {
		for (int i = k + 1; i < M; i++) {

			float t = e[i][k] / e[k][k];
			for (int j = k + 1; j < M + N; j++) {
				e[i][j] -= e[k][j] * t;
			}
			e[i][k] = 0.0f;
		}
	}

	// eliminate upper right右下
	//从最后一行开始，每次 乘加运算 消除右上数据
	//消除顺序为，列，从右到左
	for (int k = M - 1; k > 0; k--) {
		for (int i = 0; i < k; i++) {
			float t = e[i][k] / e[k][k];
			for (int j = k + 1; j < M + N; j++) {
				e[i][j] -= e[k][j] * t;
			}
			e[i][k] = 0.0f;
		}
	}


	// Divide by diagonal and put it in the output matrix.
	//float x[4][3];
	for (int i = 0; i < M; i++) {
		e[i][i] = 1.0f / e[i][i];
		for (int j = 0; j < N; j++) {
			x[i][j] = e[i][j + M] * e[i][i];
		}
	}
}

//splat
Voxel splat(const cv::Mat& lowin, const cv::Mat& lowout, float r_sigma, int s_sigma)
{
	const int lowWidth = lowin.cols;
	const int lowHeight = lowin.rows;

	//转为单通道
	Mat gray_lowin;
	Mat bgr[3];   //destination array
	split(lowin, bgr);//split source  
	gray_lowin = 0.25f*bgr[0] + 0.5f*bgr[1] + 0.25f*bgr[2];

	// bilateral grid size
	int grid_width, grid_height, grid_depth, grid_cellsize;
	grid_width = lowWidth / s_sigma + 1;
	grid_height = lowHeight / s_sigma + 1;
	grid_depth = std::round(1.0f / r_sigma);
	grid_cellsize = 3 * (3 + 1);//3*4=12

	//每个网格存储着累计的参数
	//三通道每个网格参数为22（10+3*4），单通道参数为14(10+1*4)个
	const bool channels3 = lowout.channels() == 3 ? true : false;
	const int numPara = channels3 ? 22 : 14;
	Voxel histogram(grid_width, grid_height, grid_depth, numPara);

	//splatting the input values at location determined by the guide image
	for (int y = 0; y < grid_height; y++) {//each gird row
		for (int x = 0; x < grid_width; x++) {//each gird col

			for (int ry = 0; ry < s_sigma; ry++) {//each row of one gird
				for (int rx = 0; rx < s_sigma; rx++) {//each col of one gird
					int sx = x * s_sigma + rx - s_sigma / 2;
					//int sx = x * s_sigma + rx;
					sx = clamp(sx, 0, lowWidth - 1);
					int sy = y * s_sigma + ry - s_sigma / 2;
					//int sy = y * s_sigma + ry;
					sy = clamp(sy, 0, lowHeight - 1);
					float pos = gray_lowin.at<float>(sy, sx);
					int zi = int(std::round(pos * (1.f / r_sigma)));
					zi = clamp(zi, 0, grid_depth - 1);

					//三通道数据
					if (channels3)
					{
						cv::Vec3f s = lowin.at<cv::Vec3f>(sy, sx);
						float sr = s[0], sg = s[1], sb = s[2];
						cv::Vec3f v = lowout.at<cv::Vec3f>(sy, sx);
						float vr = v[0], vg = v[1], vb = v[2];

						float mat[22] = {
							// A
							sr*sr, sr*sg, sr*sb, sr,
							       sg*sg, sg*sb, sg,
							              sb*sb, sb,
							                    1.0f,
							// b
							vr*sr, vr*sg, vr*sb, vr,
							vg*sr, vg*sg, vg*sb, vg,
							vb*sr, vb*sg, vb*sb, vb
						};

						// fill histogram
						for (int c = 0; c < 22; c++) {
							histogram(x, y, zi, c) += mat[c];
						}
					}
					else
					{
						cv::Vec3f s = lowin.at<cv::Vec3f>(sy, sx);
						float sr = s[0], sg = s[1], sb = s[2];
						float v = lowout.at<float>(sy, sx);

						float mat[14] = {
							// A
							sr*sr, sr*sg, sr*sb, sr,
							       sg*sg, sg*sb, sg,
							              sb*sb, sb,
							                     1.0f,
							// b
							v*sr, v*sg, v*sb, v
						};

						// fill histogram
						for (int c = 0; c < 14; c++) {
							histogram(x, y, zi, c) += mat[c];
						}
					}


				}
			}
		}
	}

	return histogram;
}
//blur
Voxel blur(Voxel voxel)
{
	// 1/(r+1)^3
	const float t0 = 1.0f / 64.f;
	const float t1 = 1.0f / 27.f;
	const float t2 = 1.0f / 8.f;
	const float t3 = 1.0f;
	//const float ts[7] = { t0, t1, t2, t3, t2, t1, t0 };

	int w = voxel.width(), h = voxel.height(), d = voxel.depth(), m = voxel.cells();
	// blur z
	Voxel blurz(w, h, d, m);
	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
			for (int z = 0; z < d; z++)
				for (int c = 0; c < m; c++) {
		blurz(x, y, z, c) =
			voxel.clamp_at(x, y, z - 3, c)*t0 +
			voxel.clamp_at(x, y, z - 2, c)*t1 +
			voxel.clamp_at(x, y, z - 1, c)*t2 +
			voxel.clamp_at(x, y, z, c)*t3 +
			voxel.clamp_at(x, y, z + 1, c)*t2 +
			voxel.clamp_at(x, y, z + 2, c)*t1 +
			voxel.clamp_at(x, y, z + 3, c)*t0;
				}


	// blur y
	Voxel blury(w, h, d, m);
	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
			for (int z = 0; z < d; z++)
				for (int c = 0; c < m; c++) {
		blury(x, y, z, c) =
			blurz.clamp_at(x, y - 3, z, c)*t0 +
			blurz.clamp_at(x, y - 2, z, c)*t1 +
			blurz.clamp_at(x, y - 1, z, c)*t2 +
			blurz.clamp_at(x, y, z, c)*t3 +
			blurz.clamp_at(x, y + 1, z, c)*t2 +
			blurz.clamp_at(x, y + 2, z, c)*t1 +
			blurz.clamp_at(x, y + 3, z, c)*t0;
				}

	// blur x
	Voxel blurx(w, h, d, m);
	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
			for (int z = 0; z < d; z++)
				for (int c = 0; c < m; c++) {
		blurx(x, y, z, c) =
			blury.clamp_at(x - 3, y, z, c)*t0 +
			blury.clamp_at(x - 2, y, z, c)*t1 +
			blury.clamp_at(x - 1, y, z, c)*t2 +
			blury.clamp_at(x, y, z, c)*t3 +
			blury.clamp_at(x + 1, y, z, c)*t2 +
			blury.clamp_at(x + 2, y, z, c)*t1 +
			blury.clamp_at(x + 3, y, z, c)*t0;
				}

	return blurx;
}
//solve
Voxel solveAffainModels(Voxel blurred, const bool lowoutChannel3)
{
	// bilateral grid
	const int grid_width = blurred.width();
	const int grid_height = blurred.height();
	const int grid_depth = blurred.depth();
	const int numPara1 = lowoutChannel3 ? 12 : 4;
	Voxel bgrid(grid_width, grid_height, grid_depth, numPara1);

	// Regularize by pushing the solution towards the average gain
	// in this cell = (average output luma + eps) / (average input luma + eps).
	const float lambda = 1e-6f;
	const float epsilon = 1e-6f;

	for (int x = 0; x < grid_width; x++) {
		for (int y = 0; y < grid_height; y++) {
			for (int z = 0; z < grid_depth; z++) {
				// fill affine matrix
				float A[4][4] = {
					{ blurred(x, y, z, 0), blurred(x, y, z, 1), blurred(x, y, z, 2), blurred(x, y, z, 3) },
					{ blurred(x, y, z, 1), blurred(x, y, z, 4), blurred(x, y, z, 5), blurred(x, y, z, 6) },
					{ blurred(x, y, z, 2), blurred(x, y, z, 5), blurred(x, y, z, 7), blurred(x, y, z, 8) },
					{ blurred(x, y, z, 3), blurred(x, y, z, 6), blurred(x, y, z, 8), blurred(x, y, z, 9) }};

				//三通道 b的3*4的数组
				if (lowoutChannel3)
				{
					float b[4][3] = {
							{ blurred(x, y, z, 10), blurred(x, y, z, 14), blurred(x, y, z, 18) },
							{ blurred(x, y, z, 11), blurred(x, y, z, 15), blurred(x, y, z, 19) },
							{ blurred(x, y, z, 12), blurred(x, y, z, 16), blurred(x, y, z, 20) },
							{ blurred(x, y, z, 13), blurred(x, y, z, 17), blurred(x, y, z, 21) }};


					// The bottom right entry of A is a count of the number of
					// constraints affecting this cell.
					float N = A[3][3];

					// The last row of each matrix is the sum of input and output
					// RGB values for the pixels affecting this cell. Instead of
					// dividing them by N+1 to get averages, we'll multiply
					// epsilon by N+1. This saves two divisions.
					float output_luma = b[3][0] + 2 * b[3][1] + b[3][2] + epsilon * (N + 1);
					float input_luma = A[3][0] + 2 * A[3][1] + A[3][2] + epsilon * (N + 1);
					float gain = output_luma / input_luma;


					// Add lambda and lambda*gain to the diagonal of the
					// matrices. The matrices are sums/moments rather than
					// means/covariances, so just like above we need to multiply
					// lambda by N+1 so that it's equivalent to adding a constant
					// to the diagonal of a covariance matrix. Otherwise it does
					// nothing in cells with lots of linearly-dependent
					// constraints.
					float weighted_lambda = lambda * (N + 1);
					A[0][0] += weighted_lambda;
					A[1][1] += weighted_lambda;
					A[2][2] += weighted_lambda;
					A[3][3] += weighted_lambda;

					b[0][0] += weighted_lambda * gain;
					b[1][1] += weighted_lambda * gain;
					b[2][2] += weighted_lambda * gain;

					float result[4][3];
					solve<4, 3>(A, b, result);

					// transpose
					int c = 0;
					for (int j = 0; j < 3; j++) {
						for (int i = 0; i < 4; i++) {
							bgrid(x, y, z, c) = result[i][j];
							c++;
						}
					}
				}
				else//单通道，b为1*4的数组
				{
					float b[4][1] = {
							{ blurred(x, y, z, 10) },
							{ blurred(x, y, z, 11) },
							{ blurred(x, y, z, 12) },
							{ blurred(x, y, z, 13) },
					};


					// The bottom right entry of A is a count of the number of
					// constraints affecting this cell.
					float N = A[3][3];

					// The last row of each matrix is the sum of input RGB values
					// and output luma values for the pixels affecting this
					// cell. Instead of dividing them by N+1 to get averages,
					// we'll multiply epsilon by N+1. This saves two
					// divisions. We'll also need to multiply the luma values by
					// four, because using 1 2 1 for the input weights actually
					// computes luma*4.
					float output_luma = 4*b[3][0] + epsilon * (N + 1);
					float input_luma = A[3][0] + 2 * A[3][1] + A[3][2] + epsilon * (N + 1);
					float gain = output_luma / input_luma;


					// Add lambda and lambda*gain to the diagonal of the matrices.
					// In the rgb -> rgb case we regularized the transform to be
					// close to gain * identity. In the rgb -> gray case we
					// regularize the transform to be close to gain * (some
					// reasonable conversion of the input to luma).
					float weighted_lambda = lambda * (N + 1);
					A[0][0] += weighted_lambda;
					A[1][1] += weighted_lambda;
					A[2][2] += weighted_lambda;
					A[3][3] += weighted_lambda;

					b[0][0] += weighted_lambda * gain*0.25f;
					b[1][0] += weighted_lambda * gain*0.5f;
					b[2][0] += weighted_lambda * gain*0.25f;

					float result[4][1];
					solve<4, 1>(A, b, result);

					// transpose
					int c = 0;
					for (int j = 0; j < 1; j++) {
						for (int i = 0; i < 4; i++) {
							bgrid(x, y, z, c) = result[i][j];
							c++;
						}
					}
				}

			} // z
		} // y
	} // x
	return bgrid;
}
//slice 
Mat slice(const Mat &highin,  const Voxel& voxel, const bool channels3,
	 const float upsamplefactor, const float  r_sigma, const float s_sigma)
{

	int highW = highin.cols, highH = highin.rows, highD = highin.channels();

	int w = voxel.width(), h = voxel.height(), d = voxel.depth(), m = voxel.cells();

	Mat gray_highin, bgr[3];   //destination array
	split(highin, bgr);//split source  
	gray_highin = 0.25f*bgr[0] + 0.5f*bgr[1] + 0.25f*bgr[2];

	float big_sigma = upsamplefactor*s_sigma;

	Mat result;
	channels3 ? result.create(highin.size(), CV_32FC3) : result.create(highin.size(), CV_32FC1);
	for (int y = 0; y < highH; y++)
	{
		for (int x = 0; x < highW; x++)
		{
			////如果低分辨处没有深度值，就不进行插值
			//const int yy = clamp(0, round(yy / upsamplefactor), lowH);
			//const int xx = clamp(0, round(xx / upsamplefactor), lowW);
			//if (lowout.ptr<float>(yy)[xx] == 0.0f)
			//{
			//	result.ptr<float>(y)[x] = 0.0f;
			//	continue;
			//}

			float trilerp[12] = { 0.0f };//三线性插值的网格矩阵

			//求取三个方向插值权重
			float xf = (float)x / big_sigma;
			int xi = floor(xf);//小于等于xf的最大整数
			xf -= xi;//小数部分
			float yf = (float)y / big_sigma;
			int yi = floor(yf);//小于等于xf的最大整数
			yf -= yi;//小数部分
			float zf = gray_highin.at<float>(y, x);
			int zi = int(std::round(zf* (1.f / r_sigma)));
			zf -= zi;

			//三线性插值网格矩阵
			for (int c = 0; c < m; c++)
			{
				float  c1 = 0, c2 = 0, c3 = 0, c4 = 0;

				//z方向插值
				c1 = voxel.clamp_at(xi, yi, zi, c)*(1 - zf) +
					voxel.clamp_at(xi, yi, zi + 1, c)*zf;
				c2 = voxel.clamp_at(xi + 1, yi, zi, c)*(1 - zf) +
					voxel.clamp_at(xi + 1, yi, zi + 1, c)*zf;
				c3 = voxel.clamp_at(xi, yi + 1, zi, c)*(1 - zf) +
					voxel.clamp_at(xi, yi + 1, zi + 1, c)*zf;
				c4 = voxel.clamp_at(xi + 1, yi + 1, zi, c)*(1 - zf) +
					voxel.clamp_at(xi + 1, yi + 1, zi + 1, c)*zf;
				//x方向插值
				c1 = c1*(1 - xf) + c2*xf;
				c3 = c3*(1 - xf) + c4*xf;
				//y方向插值
				c1 = c1*(1 - yf) + c3*yf;

				trilerp[c] = c1;
			}

			//利用插值后的矩阵进行大图像求解
			if (channels3)
			{
				Vec3f rColor;
				for (int i = 0; i < 3; i++)
				{
					Vec3f color = highin.at<Vec3f>(y, x);
					rColor[i] = trilerp[i * 4 + 0] * color[0] +
						trilerp[i * 4 + 1] * color[1] +
						trilerp[i * 4 + 2] * color[2] +
						trilerp[i * 4 + 3];
				}
				result.at<Vec3f>(y, x) = rColor;
			}
			else//单通道
			{
				float rColor;
				Vec3f color = highin.at<Vec3f>(y, x);
				rColor = trilerp[0] * color[0] +
					trilerp[1] * color[1] +
					trilerp[2] * color[2] +
					trilerp[3];
				result.at<float>(y, x) = rColor;
			}
		}
	}

	return result;
}


//splat(灰度图像，lowin和lowout都是单通道归一化[0,1])
Voxel splat_gray(const cv::Mat& lowin, const cv::Mat& lowout, float r_sigma, int s_sigma)
{
	assert(lowin.channels() == 1 && lowout.channels() == 1);
	const int lowWidth = lowin.cols;
	const int lowHeight = lowin.rows;

	// bilateral grid size
	int grid_width, grid_height, grid_depth;
	grid_width = lowWidth / s_sigma + 1;
	grid_height = lowHeight / s_sigma + 1;
	grid_depth = std::round(1.0f / r_sigma);

	//每个网格存储着累计的参数
	//单通道参数为5个
	const int numPara = 5;
	Voxel histogram(grid_width, grid_height, grid_depth, numPara);

	//splatting the input values at location determined by the guide image
	for (int y = 0; y < grid_height; y++) {//each gird row
		for (int x = 0; x < grid_width; x++) {//each gird col

			for (int ry = 0; ry < s_sigma; ry++) {//each row of one gird
				for (int rx = 0; rx < s_sigma; rx++) {//each col of one gird
					int sx = x * s_sigma + rx - s_sigma / 2;
					//int sx = x * s_sigma + rx;
					sx = clamp(sx, 0, lowWidth - 1);
					int sy = y * s_sigma + ry - s_sigma / 2;
					//int sy = y * s_sigma + ry;
					sy = clamp(sy, 0, lowHeight - 1);
					float pos = lowin.at<float>(sy, sx);
					int zi = int(std::round(pos * (1.f / r_sigma)));
					zi = clamp(zi, 0, grid_depth - 1);

					float s = lowin.at<float>(sy, sx);
					float v = lowout.at<float>(sy, sx);

					float mat[5] = {
						// A
						s*s,s,
						    1,
						// b
						v*s,v
					};

					// fill histogram
					for (int c = 0; c < 5; c++) {
						histogram(x, y, zi, c) += mat[c];
					}
				
				}
			}
		}
	}

	return histogram;
}
//solve(灰度图像)
Voxel solveAffainModels_gray(Voxel blurred)
{
	// bilateral grid
	const int grid_width = blurred.width();
	const int grid_height = blurred.height();
	const int grid_depth = blurred.depth();
	const int numPara = 2;//仿射矩阵为2*1
	Voxel bgrid(grid_width, grid_height, grid_depth, numPara);

	// Regularize by pushing the solution towards the average gain
	// in this cell = (average output luma + eps) / (average input luma + eps).
	const float lambda = 1e-6f;
	const float epsilon = 1e-6f;

	for (int x = 0; x < grid_width; x++) {
		for (int y = 0; y < grid_height; y++) {
			for (int z = 0; z < grid_depth; z++) {
				// fill affine matrix
				float A[2][2] = {
						{ blurred(x, y, z, 0), blurred(x, y, z, 1) },
						{ blurred(x, y, z, 1), blurred(x, y, z, 2) } };

				float b[2][1] = {
						{ blurred(x, y, z, 3) },
						{ blurred(x, y, z, 4) }};


				// The bottom right entry of A is a count of the number of
				// constraints affecting this cell.
				float N = A[1][1];

				// The last row of each matrix is the sum of input RGB values
				// and output luma values for the pixels affecting this
				// cell. Instead of dividing them by N+1 to get averages,
				// we'll multiply epsilon by N+1. This saves two
				// divisions. 
				float output_luma = b[1][0] + epsilon * (N + 1);
				float input_luma = A[1][0] + epsilon * (N + 1);
				float gain = output_luma / input_luma;

				// Add lambda and lambda*gain to the diagonal of the matrices.
				float weighted_lambda = lambda * (N + 1);
				A[0][0] += weighted_lambda;
				A[1][1] += weighted_lambda;
	
				b[0][0] += weighted_lambda * gain;
				b[1][0] += weighted_lambda * gain;

				float result[2][1];
				solve<2, 1>(A, b, result);

				// transpose
				int c = 0;
				for (int j = 0; j < 1; j++) {
					for (int i = 0; i < 2; i++) {
						bgrid(x, y, z, c) = result[i][j];
						c++;
					}
				}
				
			} // z
		} // y
	} // x
	return bgrid;
}
//slice (灰度图像)
Mat slice_gray(const Mat &highin, const Voxel& voxel,
	const float upsamplefactor, const float  r_sigma, const float s_sigma)
{
	assert(highin.channels() == 1);
	int highW = highin.cols, highH = highin.rows, highD = highin.channels();
	int w = voxel.width(), h = voxel.height(), d = voxel.depth(), m = voxel.cells();
	float big_sigma = upsamplefactor*s_sigma;

	Mat result(highin.size(), CV_32FC1);
	for (int y = 0; y < highH; y++)
	{
		for (int x = 0; x < highW; x++)
		{
			float trilerp[2] = { 0.0f };//三线性插值的网格矩阵

			//求取三个方向插值权重
			float xf = (float)x / big_sigma;
			int xi = floor(xf);//小于等于xf的最大整数
			xf -= xi;//小数部分
			float yf = (float)y / big_sigma;
			int yi = floor(yf);//小于等于xf的最大整数
			yf -= yi;//小数部分
			float zf = highin.at<float>(y, x);
			int zi = int(std::round(zf* (1.f / r_sigma)));
			zf -= zi;

			//三线性插值网格矩阵
			for (int c = 0; c < m; c++)
			{
				float  c1 = 0, c2 = 0, c3 = 0, c4 = 0;

				//z方向插值
				c1 = voxel.clamp_at(xi, yi, zi, c)*(1 - zf) +
					voxel.clamp_at(xi, yi, zi + 1, c)*zf;
				c2 = voxel.clamp_at(xi + 1, yi, zi, c)*(1 - zf) +
					voxel.clamp_at(xi + 1, yi, zi + 1, c)*zf;
				c3 = voxel.clamp_at(xi, yi + 1, zi, c)*(1 - zf) +
					voxel.clamp_at(xi, yi + 1, zi + 1, c)*zf;
				c4 = voxel.clamp_at(xi + 1, yi + 1, zi, c)*(1 - zf) +
					voxel.clamp_at(xi + 1, yi + 1, zi + 1, c)*zf;
				//x方向插值
				c1 = c1*(1 - xf) + c2*xf;
				c3 = c3*(1 - xf) + c4*xf;
				//y方向插值
				c1 = c1*(1 - yf) + c3*yf;

				trilerp[c] = c1;
			}

			float color = highin.at<float>(y, x);
			result.at<float>(y, x) = trilerp[0] * color + trilerp[1];
			
		}
	}

	return result;
}


BilateralGuidedUpsampler::BilateralGuidedUpsampler(int high_width, int high_height, float upsampling_rate,
	float r_sigma, int s_sigma)
	:Upsampler(high_width, high_height, upsampling_rate), r_sigma_(r_sigma), s_sigma_(s_sigma)
{}

cv::Mat BilateralGuidedUpsampler::upsampling(const cv::Mat& highin, const cv::Mat& lowin, const cv::Mat& lowout)
{
	if (highin.cols != getHighWidth() || highin.rows != getHighHeight())
		throw std::invalid_argument("upsampling error: high res image size mismatch");
	if (lowin.cols != getLowWidth() || lowin.rows != getLowHeight())
		throw std::invalid_argument("upsampling error: low res in image size mismatch");
	if (lowout.cols != getLowWidth() || lowout.rows != getLowHeight())
		throw std::invalid_argument("upsampling error: low res out image size mismatch");
	if (highin.channels() != lowin.channels()) 
		throw std::invalid_argument("upsampling error: low in and high in must have the same number of channels");
	if (lowout.channels() != 1 && lowout.channels() != 3) 
		throw std::invalid_argument("upsampling error: low out must have only 1 or 3 channels");

	Mat fhighin, fresult, flowin, flowout;//归一化[0,1]
	if (highin.channels() == 1 && lowin.channels() == 1)
	{
		highin.convertTo(fhighin, CV_32FC1, 1 / 255.f);
		lowin.convertTo(flowin, CV_32FC1, 1 / 255.f);
		if (lowout.type() == CV_8UC1)
			lowout.convertTo(flowout, CV_32FC1, 1 / 255.f);
		else if (lowout.type() == CV_32FC1)
			lowout.copyTo(flowout);

		//splat
		Voxel splatGrid = splat_gray(flowin, flowout, r_sigma_, s_sigma_);
		std::cout << "splat done..." << std::endl;
		Voxel blurGrid = blur(splatGrid);
		std::cout << "blur done..." << std::endl;
		Voxel model = solveAffainModels_gray(blurGrid);
		Mat result = slice_gray(fhighin, model, getUpsamplingRate(), r_sigma_, s_sigma_);
		std::cout << "slice done..." << std::endl;

		if (lowout.type() == CV_8UC1)
			result.convertTo(fresult, CV_8UC1, 255);
		else if (lowout.type() == CV_32FC1)
			result.copyTo(fresult);
	}
	else
	{
		highin.convertTo(fhighin, CV_32FC3, 1 / 255.f);//0.0-1.0f
		lowin.convertTo(flowin, CV_32FC3, 1 / 255.f);
		const bool lowoutChannels3 = lowout.channels() == 3 ? true : false;
		if (lowout.type() == CV_8UC3)//低分辨率输出为3通道
			lowout.convertTo(flowout, CV_32FC3, 1 / 255.f);
		else if (lowout.type() == CV_8UC1)//低分辨率输出为单通道的
			lowout.convertTo(flowout, CV_32FC1, 1 / 255.f);
		else if (lowout.type() == CV_32FC1 || lowout.type() == CV_32FC3)
			lowout.copyTo(flowout);

		//splat
		Voxel splatGrid = splat(flowin, flowout, r_sigma_, s_sigma_);
		std::cout << "splat done..." << std::endl;
		Voxel blurGrid = blur(splatGrid);
		std::cout << "blur done..." << std::endl;
		Voxel model = solveAffainModels(blurGrid, lowoutChannels3);
		Mat result = slice(fhighin, model, lowoutChannels3, getUpsamplingRate(), r_sigma_, s_sigma_);
		std::cout << "slice done..." << std::endl;

		//将结果转化为opencv格式方便输出,如果低分辨率输出是图像类型的，那么进行转换
		if (lowout.type() == CV_8UC3)
			result.convertTo(fresult, CV_8UC3, 255);//0-255
		else if (lowout.type() == CV_8UC1)
			result.convertTo(fresult, CV_8UC1, 255);//0-255
		else if (lowout.type() == CV_32FC1 || lowout.type() == CV_32FC3)//如果低分辨率输出不是图像类型的，就不用转换
			result.copyTo(fresult);
	}
	return fresult;
}


JointBilateralUpsamplinger::JointBilateralUpsamplinger(int high_width, int high_height, float upsampling_rate,
	int r, float r_sigma, int s_sigma)
	:Upsampler(high_width, high_height, upsampling_rate),radius(r), sigma_color(r_sigma), sigma_space(s_sigma)
{}

cv::Mat JointBilateralUpsamplinger::upsampling(const cv::Mat &joint, const cv::Mat &lowin, const cv::Mat &lowout)
{
	cv::Mat highout(joint.size(), lowin.type());
	const int highRow = joint.rows;
	const int highCol = joint.cols;
	const int lowRow = lowin.rows;
	const int lowCol = lowin.cols;
	float upsamling_rate = 1 / getUpsamplingRate();

	if (radius <= 0)
		radius = round(sigma_space * 1.5);
	const int d = 2 * radius + 1;

	//原联合图像的通道数
	const int cnj = joint.channels();

	float *color_weight = new float[cnj * 256];
	float *space_weight = new float[d*d];
	int *space_ofs_row = new int[d*d];//坐标的差值
	int *space_ofs_col = new int[d*d];

	double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
	double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
	// initialize color-related bilateral filter coefficients  
	// 色差的高斯权重  
	for (int i = 0; i < 256 * cnj; i++)
		color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

	int maxk = 0;   // 0 - (2*radius + 1)^2  
	// initialize space-related bilateral filter coefficients  
	//空间差的高斯权重
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double r = std::sqrt((double)i * i + (double)j * j);
			if (r > radius)
				continue;
			//空间权重是作用在小图像上的
			space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff * upsamling_rate*upsamling_rate);
			space_ofs_row[maxk] = i;
			space_ofs_col[maxk++] = j;
		}
	}

	for (int r = 0; r < highRow; r++)
	{
		for (int l = 0; l < highCol; l++)
		{
			int px = l, py = r;//窗口中心像素
			const cv::Vec3b color0 = joint.ptr<cv::Vec3b>(py)[px];
			float sum_w = 0;
			float sum_value[3] = { 0 };
			for (int k = 0; k < maxk; k++)
			{
				const int qy = py + space_ofs_row[k];
				const int qx = px + space_ofs_col[k];

				if (qx < 0 || qx >= highCol || qy < 0 || qy >= highRow)
					continue;

				float fqx = qx * upsamling_rate;//低分辨率图像对应坐标
				float fqy = qy * upsamling_rate;
				int iqx = roundf(fqx);//四舍五入
				int iqy = roundf(fqy);
				if (iqx >= lowCol || iqy >= lowRow)
					continue;

				//颜色距离权重，是作用在高分辨率图像上的
				cv::Vec3b color1 = joint.ptr<cv::Vec3b>(qy)[qx];
				// 根据joint当前像素和邻域像素的 距离权重 和 色差权重，计算综合的权重  
				float w = space_weight[k] * color_weight[abs(color0[0] - color1[0]) + abs(color0[1] - color1[1]) + abs(color0[2] - color1[2])];

				if (lowin.type() == CV_8UC3)
				{
					sum_value[0] += lowin.ptr<cv::Vec3b>(iqy)[iqx][0] * w;
					sum_value[1] += lowin.ptr<cv::Vec3b>(iqy)[iqx][1] * w;
					sum_value[2] += lowin.ptr<cv::Vec3b>(iqy)[iqx][2] * w;
				}
				else if (lowin.type() == CV_8UC1)
				{
					sum_value[0] += lowin.ptr<uchar>(iqy)[iqx] * w;
				}
				else if (lowin.type() == CV_32FC3)
				{
					sum_value[0] += lowin.ptr<cv::Vec3f>(iqy)[iqx][0] * w;
					sum_value[1] += lowin.ptr<cv::Vec3f>(iqy)[iqx][1] * w;
					sum_value[2] += lowin.ptr<cv::Vec3f>(iqy)[iqx][2] * w;
				}
				else if (lowin.type() == CV_32FC1)
				{
					sum_value[0] += lowin.ptr<float>(iqy)[iqx] * w;
				}
				sum_w += w;
			}

			sum_w = 1.f / sum_w;
			if (lowin.type() == CV_8UC3)
			{
				highout.ptr<cv::Vec3b>(py)[px] = cv::Vec3b(sum_value[0] * sum_w, sum_value[1] * sum_w, sum_value[2] * sum_w);
			}
			else if (lowin.type() == CV_8UC1)
			{
				highout.ptr<uchar>(py)[px] = sum_value[0] * sum_w;
			}
			else if (lowin.type() == CV_32FC3)
			{
				highout.ptr<cv::Vec3f>(py)[px] = cv::Vec3f(sum_value[0] * sum_w, sum_value[1] * sum_w, sum_value[2] * sum_w);
			}
			else if (lowin.type() == CV_32FC1)
			{
				highout.ptr<float>(py)[px] = sum_value[0] * sum_w;
			}

		}
	}

	return highout;
}