#ifndef COLMAP_SRC_MVS_MAT_H_
#define COLMAP_SRC_MVS_MAT_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "opencv2/core.hpp"

namespace colmap {
	namespace mvs {

		template <typename T>
		class Mat {
		public:
			Mat();
			Mat(const size_t width, const size_t height, const size_t depth);
			Mat(const size_t width, const size_t height, const size_t depth, std::vector<T> &values);

			size_t GetWidth() const;
			size_t GetHeight() const;
			size_t GetDepth() const;

			size_t GetNumBytes() const;

			T Get(const size_t row, const size_t col, const size_t slice = 0) const;
			void GetSlice(const size_t row, const size_t col, T* values) const;
			T* GetPtr();
			const T* GetPtr() const;

			const std::vector<T>& GetData() const;

			void Set(const int row, const int col, const T value);
			void Set(const size_t row, const size_t col, const size_t slice,
				const T value);
			void SetSlice(const size_t row, const size_t col, const T* values);

			void Fill(const T value);

			void Read(const std::string& path);
			void Write(const std::string& path) const;

			void ReadBinary(const std::string& path);
			void WriteBinary(const std::string& path) const;

			////////////////////////////////////////////////////////////////
			void setNewData(cv::Mat mat);

		protected:
			size_t width_ = 0;
			size_t height_ = 0;
			size_t depth_ = 0;
			std::vector<T> data_;  // Mat的数据是用一个vector容器存放
		};

		////////////////////////////////////////////////////////////////////////////////
		// Implementation
		////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		Mat<T>::Mat() : Mat(0, 0, 0) {}

		template <typename T>
		Mat<T>::Mat(const size_t width, const size_t height, const size_t depth)
			: width_(width), height_(height), depth_(depth) {
			data_.resize(width_ * height_ * depth_, 0);
		}

		template <typename T>
		Mat<T>::Mat(const size_t width, const size_t height, const size_t depth, std::vector<T> &values)
			: width_(width), height_(height), depth_(depth)
		{
			data_.swap(values);
		}

		template <typename T>
		size_t Mat<T>::GetWidth() const {
			return width_;
		}

		template <typename T>
		size_t Mat<T>::GetHeight() const {
			return height_;
		}

		template <typename T>
		size_t Mat<T>::GetDepth() const {
			return depth_;
		}

		template <typename T>
		size_t Mat<T>::GetNumBytes() const {
			return data_.size() * sizeof(T);
		}



		template <typename T>
		T Mat<T>::Get(const size_t row, const size_t col, const size_t slice) const {
			return data_.at(slice * width_ * height_ + row * width_ + col);
		}

		template <typename T>
		void Mat<T>::GetSlice(const size_t row, const size_t col, T* values) const {
			for (size_t slice = 0; slice < depth_; ++slice) {
				values[slice] = Get(row, col, slice);
			}
		}

		template <typename T>
		T* Mat<T>::GetPtr() {
			return data_.data();
		}

		template <typename T>
		const T* Mat<T>::GetPtr() const
		{
			return data_.data();
		}

		template <typename T>
		const std::vector<T>& Mat<T>::GetData() const
		{
			return data_;
		}

		template <typename T>
		void Mat<T>::Set(const int row, const int col, const T value)
		{
			Set(row, col, 0, value);
		}

		template <typename T>
		void Mat<T>::Set(const size_t row, const size_t col, const size_t slice,
			const T value) 
		{
			data_.at(slice * width_ * height_ + row * width_ + col) = value;
		}

		template<typename T>
		void Mat<T>::SetSlice(const size_t row, const size_t col, const T* values)
		{
			for (size_t slice = 0; slice < depth_; ++slice) {
				data_.at(slice * width_ * height_ + row * width_ + col) = values[slice];
			}
		}

		///////////////////////////////////////////////
		template <typename T>
		void Mat<T>::setNewData(cv::Mat mat)
		{
			width_ = mat.cols;
			height_ = mat.rows;
			depth_ = mat.channels();
			data_.resize(width_*height_*depth_);
			if (depth_ == 1)
				memcpy(GetPtr(), mat.data, width_*height_ * sizeof(float));
			else
			{
				std::vector<cv::Mat> splitMat(3);
				cv::split(mat, splitMat);
				memcpy(GetPtr(), splitMat.at(0).data, width_*height_ * sizeof(float));
				memcpy(GetPtr() + width_ * height_, splitMat.at(1).data, width_*height_ * sizeof(float));
				memcpy(GetPtr() + 2 * width_*height_, splitMat.at(2).data, width_*height_ * sizeof(float));
			}
			//for (int i = 0; i < height_; i++)
			//{ 
			//	for (int j = 0; j < width_; j++)
			//	{
			//		if (depth_==1)
			//			Set(i, j, mat.ptr<float>(i)[j]);
			//		else
			//		{
			//			for (int k = 0; k < depth_;k++)
			//			Set(i, j, k,mat.ptr<cv::Vec3f>(i)[j][k]);
			//		}
			//	}
			//}
		}
		/////////////////////////////////////////


		template <typename T>
		void Mat<T>::Fill(const T value) 
		{
			std::fill(data_.begin(), data_.end(), value);
		}

		template <typename T>
		void Mat<T>::Read(const std::string& path) 
		{
			std::fstream text_file(path, std::ios::in);
			if (!text_file.is_open())
			{
				std::cout << path << " Open failed !" << std::endl;
				std::system("pause");
				std::exit(EXIT_FAILURE);
			}

			char unused_char;
			text_file >> width_ >> unused_char >> height_ >> unused_char >> depth_ >> unused_char;

			// 需要给data分配存储空间
			data_.resize(width_*height_*depth_);

			for (int r = 0; r < height_; r++)
			{
				for (int c = 0; c < width_; c++)
				{
					for (int d = 0; d < depth_; d++)
					{
						T value;
						text_file >> value;
						Set(r, c, d, value);
					}
				}
			}

			text_file.close();
		}

		template <typename T>
		void Mat<T>::ReadBinary(const std::string& path)
		{
			//std::cout << "reading image: " << path << std::endl;

			std::fstream text_file(path, std::ios::in | std::ios::binary);
			if (!text_file.is_open())
			{
				std::cout << path << " Open failed !" << std::endl;
				std::system("pause");
				std::exit(EXIT_FAILURE);
			}

			char unused_char;
			text_file >> width_ >> unused_char >> height_ >> unused_char >> depth_ >> unused_char;

			//std::cout << "width, height, depth: " << width_ << " " << height_ << " " << depth_
			//	<< std::endl;

			std::streampos pos = text_file.tellg();
			text_file.close();

			assert(width_ > 0);
			assert(height_ > 0);
			assert(depth_ > 0);

			data_.resize(width_ * height_ * depth_);

			std::fstream binary_file(path, std::ios::in | std::ios::binary);
			if (!binary_file.is_open())
			{
				std::cout << path << " Open failed !" << std::endl;
				std::system("pause");
				std::exit(EXIT_FAILURE);
			}
			binary_file.seekg(pos);

			binary_file.read((char *)data_.data(), width_*height_*depth_ * sizeof(T));
			binary_file.close();
		}

		template <typename T>
		void Mat<T>::Write(const std::string& path) const 
		{
			std::fstream text_file(path, std::ios::out);
			if (!text_file.is_open())
			{
				std::cout << path << " Open failed !" << std::endl;
				std::system("pause");
				std::exit(1);
			}
			text_file << width_ << "&" << height_ << "&" << depth_ << "&" << std::endl;

			for (int r = 0; r < height_; r++)
			{
				for (int c = 0; c < width_; c++)
				{
					for (int d = 0; d < depth_; d++)
					{
						text_file << Get(r, c, d) << " ";
					}
				}
				text_file << std::endl;
			}

			text_file.close();
		}

		template <typename T>
		void Mat<T>::WriteBinary(const std::string& path) const 
		{
			std::fstream text_file(path, std::ios::out);
			if (!text_file.is_open())
			{
				std::cout << path << " Open failed !" << std::endl;
				std::system("pause");
				std::exit(1);
			}

			text_file << width_ << "&" << height_ << "&" << depth_ << "&";
			text_file.close();

			std::fstream binary_file(path,
				std::ios::out | std::ios::binary | std::ios::app);

			binary_file.write((char *)data_.data(), width_*height_*depth_ * sizeof(T));
			binary_file.close();
		}

	}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_MAT_H_
