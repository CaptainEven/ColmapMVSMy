#include "JointBilateralFilter.h"

bool JointBilateralFilter::isInputRight()
{
	//原图像和联合图像不能为空，原图像和目标图像不能只想相同的物理空间
	CV_Assert(src_.empty() == false && joint_.empty() == false && src_.data != dst_.data);

	//如果是滤波，原图像，联合图像，最终输出图像大小必须一样

	if (dst_.empty())
		dst_ = Mat::zeros(src_.size(), src_.type());
	CV_Assert(src_.size() == joint_.size());


	if (sigma_color_ <= 0)
		sigma_color_ = 1;
	if (sigma_space_ <= 0)
		sigma_space_ = 1;

	if (d_ <= 0)
		radius_ = cvRound(sigma_space_ * 1.5);    // 根据 sigma_space 计算 radius  
	else
		radius_ = d_ / 2;
	radius_ = MAX(radius_, 1);
	d_ = radius_ * 2 + 1; // 重新计算 像素“矩形”邻域的直径d，确保是奇数

	return true;
}

void JointBilateralFilter::computerWeight()
{
	// 扩展 src 和 joint 长宽各2*radius  
	copyMakeBorder(joint_, jim_, radius_, radius_, radius_, radius_, BORDER_REPLICATE);
	copyMakeBorder(src_, sim_, radius_, radius_, radius_, radius_, BORDER_REPLICATE);

	const int cn = src_.channels();
	const int cnj = joint_.channels();// cnj: joint的通道数  

	//vector<float> _color_weight(cnj * 256);//每个通道的颜色差范围是【0，255】
	//vector<float> _space_weight(d_ * d_);    // (2*radius + 1)^2  
	//vector<int> _space_ofs_jnt(d_ * d_);
	//vector<int> _space_ofs_src(d_ * d_);
	//color_weight = &_color_weight[0];
	//space_weight = &_space_weight[0];
	//space_ofs_jnt = &_space_ofs_jnt[0];
	//space_ofs_src = &_space_ofs_src[0];

	color_weight = new float[cnj * 256];
	space_weight = new float[d_*d_];
	space_ofs_jnt = new int[d_*d_];
	space_ofs_src = new int[d_*d_];


	double gauss_color_coeff = -0.5 / (sigma_color_ * sigma_color_);
	double gauss_space_coeff = -0.5 / (sigma_space_ * sigma_space_);
	// initialize color-related bilateral filter coefficients  
	// 色差的高斯权重  
	for (int i = 0; i < 256 * cnj; i++)
		color_weight[i] = (float)std::exp(i * i * gauss_color_coeff);

	maxk = 0;   // 0 - (2*radius + 1)^2  
	// initialize space-related bilateral filter coefficients  
	//空间差的高斯权重
	for (int i = -radius_; i <= radius_; i++)
	{
		for (int j = -radius_; j <= radius_; j++)
		{
			double r = std::sqrt((double)i * i + (double)j * j);
			if (r > radius_)
				continue;
			space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
			// joint 邻域内的相对坐标 (i, j)【偏移量】左上角为(-radius, -radius)，右下角为(radius, radius)  
			space_ofs_jnt[maxk] = (int)(i * jim_.step1(0) + j * jim_.step1(1)); 
			//step是每一维数据字节数，而step1是通道数量	。此处应该用step1来作为取地址的偏移量												
			space_ofs_src[maxk++] = (int)(i * sim_.step1(0) + j * sim_.step1(1));// src 邻域内的相对坐标 (i, j)  
		}
	}
}

Mat JointBilateralFilter::runJBF()
{
	isInputRight();
	computerWeight();

	const int cn = src_.channels();
	const int cnj = joint_.channels();// cnj: joint的通道数
	bool srcType;
	if (src_.type() == CV_8UC1 || src_.type() == CV_8UC3)
	{
		srcType = true;//如果原图像类型是char型的，为true
	}
	else if (src_.type() == CV_32FC1 || src_.type() == CV_32FC3)
	{
		srcType = false;//如果原图像类型是float型的，为false
	}


	for (int i = 0; i < dst_.rows; i++)
	{
		//因为joint图像往边界处扩展了，所以行列索引都要加上radius_
		const uchar *jptr = jim_.data + (i + radius_) * jim_.step[0] + radius_ * jim_.step[1];  // &jim.ptr(i+radius)[radius]  
		
		//以uchar为指针寻址，没增加一个地址偏移量，就挪动1以字节
		const uchar *sptr = sim_.data + (i + radius_) * sim_.step[0] + radius_ * sim_.step[1]; // &sim.ptr(i+radius)[radius]  
		uchar *dptr = dst_.data + i * dst_.step[0];                                  // dst.ptr(i)  
		
		//以float为指针寻址，每增加一个地址偏移量，就挪动4个字节
		const float *sptrf = (float *)(sim_.data + (i + radius_) * sim_.step[0] + radius_ * sim_.step[1]); // &sim.ptr(i+radius)[radius]  
		float *dptrf = (float *)(dst_.data + i * dst_.step[0]);                                  // dst.ptr(i)  
	

		// src 和 joint 通道数不同的四种情况  
		if (cn == 1 && cnj == 1)
		{
			for (int j = 0; j < dst_.cols; j++)
			{
				float sum = 0, wsum = 0;
				int val0 = jptr[j]; // jim.ptr(i + radius)[j + radius]  
				float val2;
				for (int k = 0; k < maxk; k++)
				{
					int val = jptr[j + space_ofs_jnt[k]];// jim.ptr(i + radius + offset_x)[j + radius + offset_y]  
					
					if (srcType)
						val2 = sptr[j + space_ofs_src[k]];// sim.ptr(i + radius + offset_x)[j + radius + offset_y]  
					else
						val2 = sptrf[j + space_ofs_src[k]];

					// 根据joint当前像素和邻域像素的 距离权重 和 色差权重，计算综合的权重  
					float w = space_weight[k]* color_weight[std::abs(val - val0)];
					sum += val2 * w;    // 统计 src 邻域内的像素带权和  
					wsum += w;          // 统计权重和  
				}
				// overflow is not possible here => there is no need to use CV_CAST_8U  
				// 归一化 src 邻域内的像素带权和，并赋给 dst对应的像素 
				if (srcType)
					dptr[j] = (uchar)cvRound(sum / wsum);
				else
					dptrf[j] = (sum / wsum);

			}
		}
		else if (cn == 3 && cnj == 3)
		{
			for (int j = 0; j < dst_.cols * 3; j += 3)
			{
				float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
				int b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2]; // jim.ptr(i + radius)[j + radius][0...2]  
				for (int k = 0; k < maxk; k++)
				{
					const uchar *sptr_k = jptr + j + space_ofs_jnt[k];
					const uchar *sptr_k2 = sptr + j + space_ofs_src[k];
					const float *sptrf_k2 = sptrf + j + space_ofs_src[k];

					int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2]; // jim.ptr(i + radius + offset_x)[j + radius + offset_y][0...2]  
					float w = space_weight[k] * color_weight[std::abs(b - b0) + std::abs(g - g0) + std::abs(r - r0)];
					sum_b += (srcType ? sptr_k2[0] : sptrf_k2[0]) * w;;   // sim.ptr(i + radius + offset_x)[j + radius + offset_y][0...2]  
					sum_g += (srcType ? sptr_k2[1] : sptrf_k2[1]) * w;
					sum_r += (srcType ? sptr_k2[2] : sptrf_k2[2]) * w;
					wsum += w;
				}
				wsum = 1.f / wsum;
				if (srcType)
				{
					dptr[j] = (uchar)cvRound(sum_b * wsum);
					dptr[j + 1] = (uchar)cvRound(sum_g * wsum);
					dptr[j + 2] = (uchar)cvRound(sum_r * wsum);
				}
				else
				{
					dptrf[j] = (sum_b * wsum);
					dptrf[j + 1] = (sum_g * wsum);
					dptrf[j + 2] = (sum_r * wsum);
				}
			}
		}
		else if (cn == 1 && cnj == 3)
		{
			for (int j = 0, l = 0; j < dst_.cols * 3; j += 3, l++)
			{
				float sum_b = 0, wsum = 0;
				float val;
				int b0 = jptr[j], g0 = jptr[j + 1], r0 = jptr[j + 2];   // jim.ptr(i + radius)[j + radius][0...2]  
				for (int k = 0; k < maxk; k++)
				{
					if (srcType)
						val = sptr[l + space_ofs_src[k]];  // sim.ptr(i + radius + offset_x)[l + radius + offset_y]  
					else
						val = sptrf[l + space_ofs_src[k]];

					const uchar *sptr_k = jptr + j + space_ofs_jnt[k];
					int b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];// jim.ptr(i + radius + offset_x)[j + radius + offset_y][0...2]  

					float w = space_weight[k]* color_weight[std::abs(b - b0) + std::abs(g - g0)+ std::abs(r - r0)];
					sum_b += val * w;
					wsum += w;
				}
				wsum = 1.f / wsum;
				if (srcType)
					dptr[l] = (uchar)cvRound(sum_b * wsum);
				else
					dptrf[l] = (sum_b*wsum);
			}
		}
		else if (cn == 3 && cnj == 1)
		{
			for (int j = 0, l = 0; j < dst_.cols * 3; j += 3, l++)
			{
				float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
				int val0 = jptr[l]; // jim.ptr(i + radius)[l + radius]  
				for (int k = 0; k < maxk; k++)
				{
					int val = jptr[l + space_ofs_jnt[k]]; // jim.ptr(i + radius + offset_x)[l + radius + offset_y]  

					const uchar *sptr_k = sptr + j + space_ofs_src[k];// sim.ptr(i + radius + offset_x)[j + radius + offset_y]   
					const float *sptrf_k = sptrf + j + space_ofs_src[k];

					float w = space_weight[k] * color_weight[std::abs(val - val0)];

					sum_b += (srcType ? sptr_k[0] : sptrf_k[0]) * w; // sim.ptr(i + radius + offset_x)[j + radius + offset_y] [0...2]  
					sum_g += (srcType ? sptr_k[1] : sptrf_k[1]) * w;
					sum_r += (srcType ? sptr_k[2] : sptrf_k[2]) * w;
					wsum += w;
				}

				// overflow is not possible here => there is no need to use CV_CAST_8U  
				wsum = 1.f / wsum;
				if (srcType)
				{
					dptr[j] = (uchar)cvRound(sum_b * wsum);
					dptr[j + 1] = (uchar)cvRound(sum_g * wsum);
					dptr[j + 2] = (uchar)cvRound(sum_r * wsum);
				}
				else
				{
					dptrf[j] = (sum_b * wsum);
					dptrf[j + 1] = (sum_g * wsum);
					dptrf[j + 2] = (sum_r * wsum);
				}
			}
		}
	}

	return dst_;
}