#include "Utils.h"


// ----------------- Util functions
bool FindOrCreateDirectory(const char* path)
{
	//�������Ŀ¼����Ϊ0�����򷵻�-1
	if (_access(path, 0) == 0)
	{
		std::cout << "Find directory��" << path << std::endl;
		return true;
	}
	else
	{
		//�����ɹ�����0������Ϊ-1
		if (_mkdir(path) == 0)
		{
			std::cout << "Create directory success:" << path << std::endl;
			return true;
		}
		else
		{
			std::cout << "No find directory��" << path << "��and can not create it." << std::endl;
			return false;
		}
	}
}

// ���ַ���str_1������str_2�滻Ϊstr_3
void StringReplace(std::string& str_1,
	const std::string& str_2,
	const std::string& str_3)
{
	std::string::size_type pos = 0;
	std::string::size_type a = str_2.size();
	std::string::size_type b = str_3.size();

	while ((pos = str_1.find(str_2, pos)) != std::string::npos)
	{
		str_1.replace(pos, a, str_3);
		pos += b;
	}
}

// ���ַ���sep�з��ַ���str, �����result 
void StringSplit(const std::string& str,
	const std::string& sep,
	std::vector<std::string>& result)
{
	std::string::size_type pos1, pos2;
	pos1 = 0;
	pos2 = str.find(sep, pos1);
	while (std::string::npos != pos2)
	{
		result.push_back(str.substr(pos1, pos2 - pos1));
		pos1 = pos2 + sep.size();
		pos2 = str.find(sep, pos1);
	}
	if (pos1 != str.size())
	{
		result.push_back(str.substr(pos1));
	}
}
