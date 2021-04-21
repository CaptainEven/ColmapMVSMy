#include "patch_match.h"
#include "fusion.h"
#include "BilateralTextureFilter.h"
//#include "MyPatchMatch.h"
//#include "JointBilateralFilter.h"
//#include <opencv2\ximgproc.hpp>
#include "upsampler.h"
//#include "BilateralGrid.h"
//#include "FastBilateralSolverMe.h"

#include"Utils.h"

using namespace colmap;


////***************************************************************************////
//对输入图像进行细节和结构增强
//1:src+BTF 2:src+fastBF  3:down + BGU
void detailAndStructureEnhance(const std::string& path,
	int bStructureEnhance = 1,
	bool bDetailEnhance = false)
{
	const std::string bundler_list_path = path + "/bundler.out.list.txt";
	std::ifstream f_img_names(bundler_list_path);

	assert(f_img_names.is_open());

	std::string image_name;
	const std::string img_enhance_dir = path + "/images_enhance/";

	// 查找或创建增强图像目录
	FindOrCreateDirectory(img_enhance_dir.c_str());

	clock_t T_start, T_end;

	int num = -1;
	while (getline(f_img_names, image_name))
	{
		++num;

		const std::string img_path = path + "/images/" + image_name;
		const std::string img_path_output = img_enhance_dir + image_name;

		cv::Mat bitmap = cv::imread(img_path);
		if (bitmap.empty())
		{
			cout << "[Err]: empty bitmap." << endl;
			return;
		}

		const cv::Size& src_size = bitmap.size();
		const cv::Size& down_size = cv::Size(src_size.width / 3.0f, src_size.height / 3.0f);
		cv::resize(bitmap, bitmap, down_size);

		if (bDetailEnhance)
		{
			cout << "=> Detail Enhance: " << num << "...";

			T_start = clock();

			DentailEnhance(bitmap, bitmap, 3.0 * 2, 0.1f, BilateralTextureFilterType);
			//detailEnhance(bitmap, bitmap);  // opencv

			T_end = clock();
			cout << " | time: " << (float)(T_end - T_start) / CLOCKS_PER_SEC << "s" << endl;
		}
		else if (bStructureEnhance == 1)
		{
			cout << "=> Structure Enhance (BTextureF): " << num << "...";
			T_start = clock();

			MultiscaleStructureEnhance(bitmap, bitmap);

			T_end = clock();
			cout << " | time: " << (float)(T_end - T_start) / CLOCKS_PER_SEC << "s" << endl;
		}
		else if (bStructureEnhance == 2)
		{
			cout << "Structure Enhance (fastBF): " << num << "...";
			T_start = clock();

			MultiscaleStructureEnhance(bitmap, bitmap, false);

			T_end = clock();
			cout << " | time: " << (float)(T_end - T_start) / CLOCKS_PER_SEC << "s" << endl;
		}
		else if (bStructureEnhance == 3)
		{
			cout << "Structure Enhance (BGU): " << num << "...";
			T_start = clock();

			cv::Size down_size2 = cv::Size(down_size.width / 4.0, down_size.height / 4.0);
			cv::Mat lowin; resize(bitmap, lowin, down_size2);
			cv::Mat lowout;

			MultiscaleStructureEnhance(lowin, lowout, true, 4, 3, 1);
			BilateralGuidedUpsampler bgu(down_size.width, down_size.height, 4.0f, 1 / 4.f, 16);

			bitmap = bgu.upsampling(bitmap, lowin, lowout);

			//JointBilateralUpsamplinger jbu(down_size1.width, down_size1.height, 4.0f, 10, 10.f, 10);
			//cv::Mat bitmap = jbu.upsampling(bitmap, lowout, lowout);

			T_end = clock();
			cout << " | time: " << (float)(T_end - T_start) / CLOCKS_PER_SEC << "s" << endl;
		}

		cv::resize(bitmap, bitmap, src_size);
		cv::imwrite(img_path_output, bitmap);
	}
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("[Err]: wrong number of cmd line parameters.\n");
		return -1;
	}

	const std::string path = string(argv[1]);

	const std::string format = "colmap";
	const std::string pmvsOptionsName = "option-all";

	//// ------------- Patch match stereo
	//mvs::PatchMatch::Options PMoptions;
	//mvs::PatchMatchController PM(PMoptions, path, format, pmvsOptionsName);
	//PM.Run();
	////PM.RunMyPm();
	//// ------------- 

	int enhance = atoi(argv[2]);  // 1 or -1
	if (enhance == 1)
	{
		detailAndStructureEnhance(path, 1, false);  // argc1:细节, agrc2:结构
	}
	else
	{
		mvs::StereoFusion::Options SFoptions;
		const std::string inputType = "geometric";
		mvs::StereoFusion SF(SFoptions, path, format, inputType);

		SF.Run();
	}

	std::system("pause");
	return 0;
}