#include "BilateralGrid.h"

//所有的图像值域在[0,1]区间内
Grid splat(const Mat &low_res_in,const Mat &low_res_out, const float s_sigma, const float l_sigma)
{
	const int lowRow = low_res_in.rows;
	const int lowCol = low_res_in.cols;
	const int lowInChannel = low_res_in.channels();
	const int lowOutChannel = low_res_out.channels();


	////三通道变单通道
	Mat gray_low_res_in;
	Mat bgr[3];  
	if (lowInChannel==3)
	{
		split(low_res_in, bgr);//split source  
		gray_low_res_in = 0.25f*bgr[0] + 0.5f*bgr[1] + 0.25f*bgr[2];
	}
	else
		gray_low_res_in = low_res_in;
	//if (channels==3)
	//	cvtColor(low_res_in, gray_low_res_in, CV_BGR2GRAY);

	// bilateral grid size
	int grid_width, grid_height, grid_depth, grid_cellsize;
	grid_width = ceilf((float)lowCol / s_sigma);
	grid_height = ceilf((float)lowRow / s_sigma);
	grid_depth = ceilf(1.f / l_sigma);
	grid_cellsize = lowOutChannel;//网格内数据个数为低分辨率输出图像通道数


	Grid grid(grid_width, grid_height, grid_depth,grid_cellsize);
	for (int y = 0; y < grid_height; y++) 
	{
		for (int x = 0; x < grid_width; x++)
		{

			std::vector<std::vector<float>> values(grid_depth, std::vector<float>(grid_cellsize, 0.0f));//统计每个网格数据和
			std::vector<int> nums(grid_depth, 0);//统计每个网格有多少数据
			for (int ry = 0; ry < s_sigma; ry++)
			{
				for (int rx = 0; rx < s_sigma; rx++)
				{
					//int sx = x * s_sigma + rx;
					int sx = x * s_sigma + rx - s_sigma / 2;				
					sx = grid.clamp(sx, 0, lowCol - 1);

					//int sy = y * s_sigma + ry;
					int sy = y * s_sigma + ry - s_sigma / 2;
					sy = grid.clamp(sy, 0, lowRow - 1);

					float pos = gray_low_res_in.at<float>(sy, sx);
					int zi = int(std::round(pos / l_sigma));
					zi = grid.clamp(zi, 0, grid_depth - 1);

					//根据引导图像low_res_in的颜色属性，向网格中填入结果图像数据low_res_out
					if (lowOutChannel == 1)
						values[zi][0] += low_res_out.ptr<float>(sy)[sx];
					else
					{
						Vec3f color = low_res_out.ptr<Vec3f>(sy)[sx];
						values[zi][0] += color[0];
						values[zi][1] += color[1];
						values[zi][2] += color[2];
					}
					nums[zi]++;
				}
			}
			//填充网格
			for (int i = 0; i < grid_depth; i++)
			{
				//除数不能为0
				if (nums[i] == 0)
					continue;
				for (int j = 0; j < grid_cellsize; j++)
				{
					grid.set(x, y, i, j, values[i][j] / nums[i]);
				}
			}
		}
	}
	return grid;
}

Grid blur(const Grid &grid)
{
	const float t0 = 0.0f / 64.f;
	const float t1 = 0.0f / 27.f;
	const float t2 = 1.0f / 8.f;
	const float t3 = 1.0f;
	//const float ts[7] = { t0, t1, t2, t3, t2, t1, t0 };
	const float sumW = 1.0f / (t0 + t1 + t2 + t3);
	//const float sumW = 1.0f;

	int w = grid.width(), h = grid.height(), d = grid.depth(), m = grid.cells();
	// blur z
	Grid blurz(w, h, d, m);
	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
			for (int z = 0; z < d; z++)
				for (int c = 0; c < m; c++)
				{
		blurz.set(x, y, z, c,
			(grid.clamp_at(x, y, z - 3, c)*t0 +
			grid.clamp_at(x, y, z - 2, c)*t1 +
			grid.clamp_at(x, y, z - 1, c)*t2 +
			grid.clamp_at(x, y, z, c)*t3 +
			grid.clamp_at(x, y, z + 1, c)*t2 +
			grid.clamp_at(x, y, z + 2, c)*t1 +
			grid.clamp_at(x, y, z + 3, c)*t0)*sumW);
				}


	// blur y
	Grid blury(w, h, d, m);
	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
			for (int z = 0; z < d; z++)
				for (int c = 0; c < m; c++) {
		blury.set(x, y, z, c,
			(blurz.clamp_at(x, y - 3, z, c)*t0 +
			blurz.clamp_at(x, y - 2, z, c)*t1 +
			blurz.clamp_at(x, y - 1, z, c)*t2 +
			blurz.clamp_at(x, y, z, c)*t3 +
			blurz.clamp_at(x, y + 1, z, c)*t2 +
			blurz.clamp_at(x, y + 2, z, c)*t1 +
			blurz.clamp_at(x, y + 3, z, c)*t0)*sumW);
				}

	// blur x
	Grid blurx(w, h, d, m);
	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
			for (int z = 0; z < d; z++)
				for (int c = 0; c < m; c++) {
		blurx.set(x, y, z, c,
			(blury.clamp_at(x - 3, y, z, c)*t0 +
			blury.clamp_at(x - 2, y, z, c)*t1 +
			blury.clamp_at(x - 1, y, z, c)*t2 +
			blury.clamp_at(x, y, z, c)*t3 +
			blury.clamp_at(x + 1, y, z, c)*t2 +
			blury.clamp_at(x + 2, y, z, c)*t1 +
			blury.clamp_at(x + 3, y, z, c)*t0)*sumW);
				}

	return blurx;
}

Mat slice(const Mat &high_res_in,const Mat &low_res_out,const Grid &grid, const float  s_sigma, const float l_sigma, const float upsamplefactor)
{

	const int highRow = high_res_in.rows;
	const int highCol = high_res_in.cols;
	const int highInChannel = high_res_in.channels();
	const int highOutChannel = low_res_out.channels();

	const int w = grid.width();
	const int h = grid.height();
	const int d = grid.depth();
	const int m = grid.cells();


	Mat gray_high_res_in, bgr[3];   //destination array
	if (highInChannel == 3)
	{
		split(high_res_in, bgr);//split source  
		gray_high_res_in = 0.25f*bgr[0] + 0.5f*bgr[1] + 0.25f*bgr[2];
	}
	else
		gray_high_res_in = high_res_in;
	//Mat gray_high_res_in; cvtColor(high_res_in, gray_high_res_in, CV_BGR2GRAY);

	float big_sigma = upsamplefactor*s_sigma;

	Mat result(high_res_in.size(), low_res_out.type());
	for (int y = 0; y < highRow; y++)
	{
		for (int x = 0; x < highCol; x++)
		{

			vector<float> trilerp(m, 0.0f);//三线性插值的网格矩阵

			//求取三个方向插值权重
			float xf = (float)x / big_sigma;
			int xi = floor(xf);//小于等于xf的最大整数
			xf -= xi;//小数部分

			float yf = (float)y / big_sigma;
			int yi = floor(yf);//小于等于xf的最大整数
			yf -= yi;//小数部分

			float pos = gray_high_res_in.at<float>(y, x);
			float zf = pos / l_sigma;
			int zi = floor(zf);
			zf -= zi;

			//三线性插值网格矩阵
			for (int c = 0; c < m; c++)
			{
				float  c1 = 0, c2 = 0, c3 = 0, c4 = 0;

				//z方向插值
				c1 = grid.clamp_at(xi, yi, zi, c)*(1 - zf) +
					grid.clamp_at(xi, yi, zi + 1, c)*zf;
				c2 = grid.clamp_at(xi + 1, yi, zi, c)*(1 - zf) +
					grid.clamp_at(xi + 1, yi, zi + 1, c)*zf;
				c3 = grid.clamp_at(xi, yi + 1, zi, c)*(1 - zf) +
					grid.clamp_at(xi, yi + 1, zi + 1, c)*zf;
				c4 = grid.clamp_at(xi + 1, yi + 1, zi, c)*(1 - zf) +
					grid.clamp_at(xi + 1, yi + 1, zi + 1, c)*zf;
				//x方向插值
				c1 = c1*(1 - xf) + c2*xf;
				c3 = c3*(1 - xf) + c4*xf;
				//y方向插值
				c1 = c1*(1 - yf) + c3*yf;

				trilerp[c] = c1;
			}
			if (highOutChannel == 1)
				result.at<float>(y, x) = trilerp[0];
			else
				result.at<Vec3f>(y, x) = Vec3f(trilerp[0], trilerp[1], trilerp[2]);
		}
	}

	return result;
}

Mat bilateralGridFilter(const Mat &low_res_in, const Mat &low_res_out, const Mat &high_res_in, const float s_sigma, const float l_sigma)
{
	Grid grid;
	const int upsamplingfactor = roundf(high_res_in.rows / low_res_in.rows);//上采样率
	grid = splat(low_res_in,low_res_out, s_sigma, l_sigma);
	grid = blur(grid);
	return slice(high_res_in, low_res_out,grid, s_sigma, l_sigma, upsamplingfactor);
}