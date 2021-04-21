#ifndef UTILS_H_
#define UTILS_H_

#include<corecrt_io.h>
#include<direct.h>
#include<iomanip>
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>


bool FindOrCreateDirectory(const char* path);

// 字符串替换
void StringReplace(std::string &str_1,
	const std::string &str_2,
	const std::string &str_3);

// 字符串分割
void StringSplit(const std::string& str,
	const std::string& sep,
	std::vector<std::string>& result);

// 生成等差数列
inline void Linspace(const std::vector<int>& array,
	const int num_bins,
	std::vector<float>& linspace)
{
	auto min = std::min_element(std::begin(array), std::end(array));
	auto max = std::min_element(std::begin(array), std::end(array));

	const float step = (float(*max) - float(*min)) / float(num_bins - 1);
	for (int i = 0; i < num_bins; ++i)
	{
		linspace[i] = float(*min) + float(i) * step;
	}
}


#endif // !UTILS_H
