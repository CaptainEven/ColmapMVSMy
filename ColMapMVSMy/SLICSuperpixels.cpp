#include "SLICSuperpixels.h"


void SLIC::initClusterCenter(const int &imageIndex)
{
	//该索引所对应的图像
	image = imread(path_);

	//如果要输出超像素分割结果，就克隆一个图像，专门用于显示,后续要用到该图像的都要先判断是否输出
	if (bSlicOut)
	{
		showImage = image.clone();
	}

	Mat kernel = (Mat_<char>(3, 3) << 0, 1, 0,
										1, -4, 1,
										0, 1, 0);//拉普拉斯滤波核
	cvtColor(image, imageLab, CV_BGR2Lab);//将输入图像位BGR颜色空间转化为Lab
	filter2D(image, imageLapcian, image.depth(), kernel);//将输入图像进行拉普拉斯滤波处理

	//如果没有给定步长
	if (bGivenStep == false)
	{
		S = sqrtf(image.rows*image.cols / float(k));//聚簇中心间隔
	}
	else
	{
		S = slicStep;
	}

	int colNum = ceil(image.cols / S);//x方向超像素个数
	int rowNum = ceil(image.rows / S);//y方向超像素个数

	//初始化聚簇中心
	vCC.clear();
	for (int y = 0; y < rowNum; y++)
	{
		for (int x = 0; x < colNum; x++)//从左到右，从上到下
		{
			xylab cc;
			if (x != colNum - 1)
			{
				cc.x = (int)(x*S + S / 2);
			}
			else
			{
				cc.x = (int)((x*S + image.cols - 1) / 2);
			}
			if (y != rowNum - 1)
			{
				cc.y = (int)(y*S + S / 2);
			}
			else
			{
				cc.y = (int)((y*S + image.rows - 1) / 2);
			}

			//是否输出超像素分割图像
			if (bSlicOut)
			{
				circle(showImage, Point(cc.x, cc.y), 1, Scalar(255, 0, 0), 1, 8, 0);//聚簇中心画蓝色
			}

			//在3X3的区域内，移动聚簇中心到梯度最小处;
			int tempx = 0;
			int tempy = 0;
			int tempgradient = INT_MAX;
			for (int i = cc.x - 1; i <= cc.x + 1; i++)
			{
				for (int j = cc.y - 1; j <= cc.y + 1; j++)
				{
					if (i > -1 && i < image.cols  && j > -1 && j < image.rows )
					{
						int gradient = abs(imageLapcian.at<Vec3b>(j, i)[0] +
							               imageLapcian.at<Vec3b>(j, i)[1] +
							               imageLapcian.at<Vec3b>(j, i)[2]);
						if (gradient < tempgradient)
						{
							tempx = i;
							tempy = j;
							tempgradient = gradient;
						}
					}  
				}
			}

			//调整过后的每个聚簇中心的位置和颜色
			cc.x = tempx;
			cc.y = tempy;
			cc.l = imageLab.at<Vec3b>(tempy, tempx)[0];
			cc.a = imageLab.at<Vec3b>(tempy, tempx)[1];
			cc.b = imageLab.at<Vec3b>(tempy, tempx)[2];

			vCC.push_back(cc);

			//是否输出超像素分割图像
			if (bSlicOut)
			{
				circle(showImage, Point(cc.x, cc.y), 1, Scalar(0, 0, 255), 1, 8, 0);//移动聚簇中心后画红色
			}
		}
	}
}

void SLIC::performSegmentation(const int &imageIndex)
{
	label = Mat_<int>(image.rows, image.cols, -1);//刚开始，每个像素的聚簇中心标签为-1
	Mat distance = Mat_<float>(image.rows, image.cols, -1.0);//刚开始，每个像素到聚簇中心距离为-1
	float residualError = FLT_MAX;
	int numIte = 10;

	//开始迭代
	while (residualError >= vCC.size()*0.1)//以残差小于某个阈值为迭代标准
	//while (numIte--)//以迭代次数为标准
	{
		//assignment
		//对每个聚簇中心在2S*2S的区域内为每个像素分配标签（属于哪个聚族中心）
    	int index = 0;
		for (vector<xylab>::iterator it = vCC.begin(); it != vCC.end(); it++, index++)
		{
			for (int x = int((*it).x - S); x <= int((*it).x + S); x++)
			{
				for (int y = int((*it).y - S); y <= int((*it).y + S); y++)
				{
					if (x < 0 || y < 0 || x > image.cols - 1 || y > image.rows - 1)
					{
						continue;
					}

					xylab cc = { x, y, imageLab.at<Vec3b>(y,x)[0], imageLab.at<Vec3b>(y,x)[1], imageLab.at<Vec3b>(y,x)[2] };
					float dist = xylabDistance(cc, *it);

					if (label.at<int>(y, x) == -1)
					{
						distance.at<float>(y, x) = dist;
						label.at<int>(y, x) = index;
						continue;
					}
					
					if (dist < distance.at<float>(y, x))
					{
						distance.at<float>(y, x) = dist;
						label.at<int>(y, x) = index;
					}
				}
			}
		}

		//updata
		//计算新的聚簇中心(求每个超像素的均值)
		vector<xylab> vC1;//每个超像素的x,y,l,a,b的和
		vC1.assign(vCC.size(), {0.0,0.0,0.0,0.0,0.0});
		vector<int> vClusterNum;//每个聚簇中心包含的像素个数
		vClusterNum.assign(vCC.size(), 0);

		residualError = 0.0;
		for (int i = 0; i < image.cols; i++)
		{
			for (int j = 0; j < image.rows; j++)
			{
				int id = label.at<int>(j, i);
				vC1[id].x += i;
				vC1[id].y += j;
				vC1[id].l += imageLab.at<Vec3b>(j, i)[0];
				vC1[id].a += imageLab.at<Vec3b>(j, i)[1];
				vC1[id].b += imageLab.at<Vec3b>(j, i)[2];
				vClusterNum[id]++;
			}
		}
		for (int i = 0; i < vC1.size(); i++)
		{
			vC1[i].x /= vClusterNum[i];
			vC1[i].y /= vClusterNum[i];
			vC1[i].l /= vClusterNum[i];
			vC1[i].a /= vClusterNum[i];
			vC1[i].b /= vClusterNum[i];
			//所有均值点和聚族中心之间的距离(残差)
			residualError += pow(vC1[i].x - vCC[i].x, 2) + pow(vC1[i].y - vCC[i].y, 2);
		}
		vCC = vC1;//每个超像素的均值点作为新的聚族中心
	}

	////调试看超像素分割效果
	//Mat maskImage(label.rows, label.cols, CV_8UC3);
	//for (int r = 0; r < label.rows; r++)
	//{
	//	for (int l = 0; l < label.cols; l++)
	//	{
	//		int n = label.at<int>(r, l);
	//		switch (n%5)
	//		{
	//			case 0:maskImage.at<Vec3b>(r, l) = Vec3b(0, 0, 255); break;
	//			case 1:maskImage.at<Vec3b>(r, l) = Vec3b(0, 255, 0); break;
	//			case 2:maskImage.at<Vec3b>(r, l) = Vec3b(255, 0, 0); break;
	//			case 3:maskImage.at<Vec3b>(r, l) = Vec3b(0, 0, 0); break;
	//			case 4:maskImage.at<Vec3b>(r, l) = Vec3b(255, 255, 255); break;
	//			default:
	//				break;
	//		}
	//	}
	//}
	//imwrite("slic.jpg", maskImage);

}
 
void SLIC::enforceConnectivity(const int &imageIndex)
{
	Mat curLabel = Mat_<int>(label.rows, label.cols, -1);//增连通性强后的标签
	int mask=0;//当前聚簇中心标签号
	int adjlabel=0;//相邻超像素聚簇中心标号
	
	int dx4[4] = { -1, 0, 1, 0 };
	int dy4[4] = { 0, -1, 0, 1 };

	vLabelPixelCount.clear();

	//首先找出每个超像素的起点
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)//从左到右，从上到下
		{
			if (curLabel.at<int>(y, x) < 0)
			{
				curLabel.at<int>(y, x) = mask;
				//从起点的四邻域中找到被标记过的像素，用adjlabel记录相邻超像素的标号
				for (int i = 0; i < 4; i++)
				{
					int nx = x + dx4[i];
					int ny = y + dy4[i];
					if (nx > -1 && nx <image.cols && ny > -1 && ny <image.rows)
					{
						if (curLabel.at<int>(ny, nx) >= 0)
						{
							adjlabel = curLabel.at<int>(ny, nx);//最后选择的是上面相邻超像素标号！！！
							break;//左上右下的顺序，找到邻域就退出
						}
					}
				}
				int count=1;
				vector<Point2i> vClustPoints;
				vClustPoints.push_back(Point2i(x, y));
				//开始计算当前起点超像素的个数
				//先以起点为中心，找四邻域
				for (int m = 0; m < count; m++)
				{
					for (int i = 0; i < 4; i++)
					{
						int nx = vClustPoints[m].x + dx4[i];
						int ny = vClustPoints[m].y + dy4[i];
						if (nx > -1 && nx <image.cols && ny > -1 && ny < image.rows)
						{
							//同属一个超像素的像素新标号未被标记，同时旧标号和当前操作中心点旧标号相同
							if (curLabel.at<int>(ny, nx) < 0 && label.at<int>(y, x) == label.at<int>(ny, nx))
							{
								curLabel.at<int>(ny, nx) = curLabel.at<int>(y, x);
								vClustPoints.push_back(Point2i(nx, ny));
								count++;//再以新成员为中心
							}
						}
					}
				}
				//如果当前超像素尺度小于S*S的1/4，把像素标号改为adjlabel，与前一个超像素融合
				int superSize = S*S;
				if (count <= superSize >> 2  )
				{
					for (int m = 0; m < count; m++)
					{
						Point2i curPoint = vClustPoints[m];
						curLabel.at<int>(curPoint.y, curPoint.x) = adjlabel;
					}

					if (vLabelPixelCount.size()==0)//如果是第一个超像素，直接放到容器里
					{
						vLabelPixelCount.push_back(count);
						mask++;//此时，一个聚簇中心已经找完所有像素点，进行下一个聚簇中心的寻找
					}
					else//如果不是第一个超像素，修改被融合的超像素的大小
					{
						vLabelPixelCount[adjlabel] += count;//此时，没有添加新的超像素，mask值不增加
					}
				}
				else//如果尺寸大于固定值，就直接把这个超像素放在容器里
				{
					vLabelPixelCount.push_back(count);
					mask++;
				}
			}//每一个没有被标记的像素都进行上述相同的操作
		}
	}//for循环，直到所有像素被查找完

	label = curLabel.clone();



	////调试看超像素分割效果
	//Mat maskImage(label.rows, label.cols, CV_8UC3);
	//for (int r = 0; r < label.rows; r++)
	//{
	//	for (int l = 0; l < label.cols; l++)
	//	{
	//		int n = label.at<int>(r, l);
	//		switch (n % 5)
	//		{
	//		case 0:maskImage.at<Vec3b>(r, l) = Vec3b(0, 0, 255); break;
	//		case 1:maskImage.at<Vec3b>(r, l) = Vec3b(0, 255, 0); break;
	//		case 2:maskImage.at<Vec3b>(r, l) = Vec3b(255, 0, 0); break;
	//		case 3:maskImage.at<Vec3b>(r, l) = Vec3b(0, 0, 0); break;
	//		case 4:maskImage.at<Vec3b>(r, l) = Vec3b(255, 255, 255); break;
	//		default:
	//			break;
	//		}
	//	}
	//}
	//imwrite("slic1.jpg", maskImage);
}

float SLIC::xylabDistance(xylab &a, xylab &b)
{
	float spatial = pow(a.x - b.x, 2) + pow(a.y - b.y, 2);//空间距离平方和
	float lab = pow(a.l - b.l, 2) + pow(a.a - b.a, 2) + pow(a.b - b.b, 2);//颜色距离平方和
	return sqrtf(spatial / pow(S, 2) + lab / pow(m, 2));
}

void SLIC::drawContour(const int &imageIndex)
{
	int dx4[4] = { -1, 0, 1, 0 };
	int dy4[4] = { 0, -1, 0, 1 };

	int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	for (int i = 0; i < image.cols ; i++)
	{
		for (int j = 0; j < image.rows ; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				//int x = i + dx8[k];
				//int y = j + dy8[k];

				int x = i + dx4[k];
				int y = j + dy4[k];

				if (x>=0 && x<image.cols && y>=0 && y<image.rows)
				{
					if (label.at<int>(j, i) != label.at<int>(y, x))
					{
						//showImage.at<Vec3b>(j, i) = Vec3b(255, 255, 255);//在轮廓边界画白线
						showImage.at<Vec3b>(j, i) = Vec3b(0, 0, 0);//在轮廓边界画黑线

						break;
					}
				}
			}
		}
	}

	char filename[20];
	sprintf_s(filename, "SLIC%d.jpg", imageIndex);
	string output = outPutPath_ + filename;
	imwrite(output, showImage);
	//imshow("Image", showImage);
	//waitKey(0);
}

void SLIC::drawContour1(const int &imageIndex)
{
	int dx4[4] = { -1, 0, 1, 0 };
	int dy4[4] = { 0, -1, 0, 1 };

	int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	Mat istaken(image.rows, image.cols, CV_8UC1, Scalar::all(false));

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int np = 0;
			for (int k = 0; k < 8; k++)
			{
			    int x = j + dx8[k];
				int y = i + dy8[k];

				if (x > -1 && x < image.cols && y > -1 && y < image.rows)
				{
					if (istaken.at<bool>(y,x) == false)
					{
						if (label.at<int>(i, j) != label.at<int>(y, x))
						{
							np++;
						}
					}
				}
			}
			if (np > 1)//增大可减细超像素分割线
			{
				showImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);//白线
				//showImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);//黑线
				istaken.at<bool>(i, j) = true;	
			}
		}
	}

	char filename[20];
	sprintf_s(filename, "SLIC%d.jpg", imageIndex);
	const string outputP = outPutPath_ + filename;
	imwrite(outputP, showImage);
	//imshow("Image", showImage);
	//waitKey(0);
}

Mat SLIC::run(const int &imageIndex)
{
	//开始执行该图像索引所对应的图像的超像素分割
	initClusterCenter(imageIndex);
	performSegmentation(imageIndex);
	enforceConnectivity(imageIndex);
	
	//是否输出超像素分割图像
	if (bSlicOut)
	{
		drawContour1(imageIndex);
	}

	return label;
}



////多线程的终极函数
//void SLIC::run1()
//{
//	cout << endl << endl << "SLIC Begin !" << endl << endl;
//
//	//初始化静态变量的大小
//	vLabel.resize(numImage);
//	vvLabelPixelCount.resize(numImage);
//	vNumSP.resize(numImage);
//
//	numLeftThread = numImage;
//
//	//准备多线程
//	ThreadedExecutor executor;
//	try
//	{
//		for (int i = 0; i < numImage; i++)
//		{
//			SLICThread *aThread = new SLICThread(new SLIC, i);
//			executor.execute(aThread);
//		}
//	}
//	catch (Synchronization_Exception &e)
//	{
//		cerr << e.what() << endl;
//	}
//
//	while (numLeftThread)
//	{
//		Thread::sleep(1000);
//	}
//	cout << endl << endl << "SLIC Done !" << endl << endl;
//}
////线程执行函数
//void SLICThread::run()
//{
//	//获取该索引下的原图像
//	parent->srcLock.acquire();
//
//	parent->image = imread(vImageFile[id]);
//
//	parent->srcLock.release();
//
//	//开始就该图像进行超像素分割
//	parent->initClusterCenter(id);
//	parent->performSegmentation(id);
//	parent->enforceConnectivity(id);
//
//	//是否输出超像素分割图像
//	if (bSlicOut)
//	{
//		parent->drawContour(id);
//	}
//
//	//把深度合成中要用到的信息保存到静态变量中
//	parent->detLock.acquire();
//
//	parent->vLabel[id] = parent->label;
//	parent->vvLabelPixelCount[id] = parent->vLabelPixelCount;
//	parent->vNumSP[id] = parent->vLabelPixelCount.size();
//
//	//执行完一个线程之后，就对线程数量统计量减一
//	parent->numLeftThread--;
//
//	parent->detLock.release();
//
//}
