#include "SLICSuperpixels.h"


void SLIC::initClusterCenter(const int &imageIndex)
{
	//����������Ӧ��ͼ��
	image = imread(path_);

	//���Ҫ��������طָ������Ϳ�¡һ��ͼ��ר��������ʾ,����Ҫ�õ���ͼ��Ķ�Ҫ���ж��Ƿ����
	if (bSlicOut)
	{
		showImage = image.clone();
	}

	Mat kernel = (Mat_<char>(3, 3) << 0, 1, 0,
										1, -4, 1,
										0, 1, 0);//������˹�˲���
	cvtColor(image, imageLab, CV_BGR2Lab);//������ͼ��λBGR��ɫ�ռ�ת��ΪLab
	filter2D(image, imageLapcian, image.depth(), kernel);//������ͼ�����������˹�˲�����

	//���û�и�������
	if (bGivenStep == false)
	{
		S = sqrtf(image.rows*image.cols / float(k));//�۴����ļ��
	}
	else
	{
		S = slicStep;
	}

	int colNum = ceil(image.cols / S);//x�������ظ���
	int rowNum = ceil(image.rows / S);//y�������ظ���

	//��ʼ���۴�����
	vCC.clear();
	for (int y = 0; y < rowNum; y++)
	{
		for (int x = 0; x < colNum; x++)//�����ң����ϵ���
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

			//�Ƿ���������طָ�ͼ��
			if (bSlicOut)
			{
				circle(showImage, Point(cc.x, cc.y), 1, Scalar(255, 0, 0), 1, 8, 0);//�۴����Ļ���ɫ
			}

			//��3X3�������ڣ��ƶ��۴����ĵ��ݶ���С��;
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

			//���������ÿ���۴����ĵ�λ�ú���ɫ
			cc.x = tempx;
			cc.y = tempy;
			cc.l = imageLab.at<Vec3b>(tempy, tempx)[0];
			cc.a = imageLab.at<Vec3b>(tempy, tempx)[1];
			cc.b = imageLab.at<Vec3b>(tempy, tempx)[2];

			vCC.push_back(cc);

			//�Ƿ���������طָ�ͼ��
			if (bSlicOut)
			{
				circle(showImage, Point(cc.x, cc.y), 1, Scalar(0, 0, 255), 1, 8, 0);//�ƶ��۴����ĺ󻭺�ɫ
			}
		}
	}
}

void SLIC::performSegmentation(const int &imageIndex)
{
	label = Mat_<int>(image.rows, image.cols, -1);//�տ�ʼ��ÿ�����صľ۴����ı�ǩΪ-1
	Mat distance = Mat_<float>(image.rows, image.cols, -1.0);//�տ�ʼ��ÿ�����ص��۴����ľ���Ϊ-1
	float residualError = FLT_MAX;
	int numIte = 10;

	//��ʼ����
	while (residualError >= vCC.size()*0.1)//�Բв�С��ĳ����ֵΪ������׼
	//while (numIte--)//�Ե�������Ϊ��׼
	{
		//assignment
		//��ÿ���۴�������2S*2S��������Ϊÿ�����ط����ǩ�������ĸ��������ģ�
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
		//�����µľ۴�����(��ÿ�������صľ�ֵ)
		vector<xylab> vC1;//ÿ�������ص�x,y,l,a,b�ĺ�
		vC1.assign(vCC.size(), {0.0,0.0,0.0,0.0,0.0});
		vector<int> vClusterNum;//ÿ���۴����İ��������ظ���
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
			//���о�ֵ��;�������֮��ľ���(�в�)
			residualError += pow(vC1[i].x - vCC[i].x, 2) + pow(vC1[i].y - vCC[i].y, 2);
		}
		vCC = vC1;//ÿ�������صľ�ֵ����Ϊ�µľ�������
	}

	////���Կ������طָ�Ч��
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
	Mat curLabel = Mat_<int>(label.rows, label.cols, -1);//����ͨ��ǿ��ı�ǩ
	int mask=0;//��ǰ�۴����ı�ǩ��
	int adjlabel=0;//���ڳ����ؾ۴����ı��
	
	int dx4[4] = { -1, 0, 1, 0 };
	int dy4[4] = { 0, -1, 0, 1 };

	vLabelPixelCount.clear();

	//�����ҳ�ÿ�������ص����
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)//�����ң����ϵ���
		{
			if (curLabel.at<int>(y, x) < 0)
			{
				curLabel.at<int>(y, x) = mask;
				//���������������ҵ�����ǹ������أ���adjlabel��¼���ڳ����صı��
				for (int i = 0; i < 4; i++)
				{
					int nx = x + dx4[i];
					int ny = y + dy4[i];
					if (nx > -1 && nx <image.cols && ny > -1 && ny <image.rows)
					{
						if (curLabel.at<int>(ny, nx) >= 0)
						{
							adjlabel = curLabel.at<int>(ny, nx);//���ѡ������������ڳ����ر�ţ�����
							break;//�������µ�˳���ҵ�������˳�
						}
					}
				}
				int count=1;
				vector<Point2i> vClustPoints;
				vClustPoints.push_back(Point2i(x, y));
				//��ʼ���㵱ǰ��㳬���صĸ���
				//�������Ϊ���ģ���������
				for (int m = 0; m < count; m++)
				{
					for (int i = 0; i < 4; i++)
					{
						int nx = vClustPoints[m].x + dx4[i];
						int ny = vClustPoints[m].y + dy4[i];
						if (nx > -1 && nx <image.cols && ny > -1 && ny < image.rows)
						{
							//ͬ��һ�������ص������±��δ����ǣ�ͬʱ�ɱ�ź͵�ǰ�������ĵ�ɱ����ͬ
							if (curLabel.at<int>(ny, nx) < 0 && label.at<int>(y, x) == label.at<int>(ny, nx))
							{
								curLabel.at<int>(ny, nx) = curLabel.at<int>(y, x);
								vClustPoints.push_back(Point2i(nx, ny));
								count++;//�����³�ԱΪ����
							}
						}
					}
				}
				//�����ǰ�����س߶�С��S*S��1/4�������ر�Ÿ�Ϊadjlabel����ǰһ���������ں�
				int superSize = S*S;
				if (count <= superSize >> 2  )
				{
					for (int m = 0; m < count; m++)
					{
						Point2i curPoint = vClustPoints[m];
						curLabel.at<int>(curPoint.y, curPoint.x) = adjlabel;
					}

					if (vLabelPixelCount.size()==0)//����ǵ�һ�������أ�ֱ�ӷŵ�������
					{
						vLabelPixelCount.push_back(count);
						mask++;//��ʱ��һ���۴������Ѿ������������ص㣬������һ���۴����ĵ�Ѱ��
					}
					else//������ǵ�һ�������أ��޸ı��ںϵĳ����صĴ�С
					{
						vLabelPixelCount[adjlabel] += count;//��ʱ��û������µĳ����أ�maskֵ������
					}
				}
				else//����ߴ���ڹ̶�ֵ����ֱ�Ӱ���������ط���������
				{
					vLabelPixelCount.push_back(count);
					mask++;
				}
			}//ÿһ��û�б���ǵ����ض�����������ͬ�Ĳ���
		}
	}//forѭ����ֱ���������ر�������

	label = curLabel.clone();



	////���Կ������طָ�Ч��
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
	float spatial = pow(a.x - b.x, 2) + pow(a.y - b.y, 2);//�ռ����ƽ����
	float lab = pow(a.l - b.l, 2) + pow(a.a - b.a, 2) + pow(a.b - b.b, 2);//��ɫ����ƽ����
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
						//showImage.at<Vec3b>(j, i) = Vec3b(255, 255, 255);//�������߽续����
						showImage.at<Vec3b>(j, i) = Vec3b(0, 0, 0);//�������߽续����

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
			if (np > 1)//����ɼ�ϸ�����طָ���
			{
				showImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);//����
				//showImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);//����
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
	//��ʼִ�и�ͼ����������Ӧ��ͼ��ĳ����طָ�
	initClusterCenter(imageIndex);
	performSegmentation(imageIndex);
	enforceConnectivity(imageIndex);
	
	//�Ƿ���������طָ�ͼ��
	if (bSlicOut)
	{
		drawContour1(imageIndex);
	}

	return label;
}



////���̵߳��ռ�����
//void SLIC::run1()
//{
//	cout << endl << endl << "SLIC Begin !" << endl << endl;
//
//	//��ʼ����̬�����Ĵ�С
//	vLabel.resize(numImage);
//	vvLabelPixelCount.resize(numImage);
//	vNumSP.resize(numImage);
//
//	numLeftThread = numImage;
//
//	//׼�����߳�
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
////�߳�ִ�к���
//void SLICThread::run()
//{
//	//��ȡ�������µ�ԭͼ��
//	parent->srcLock.acquire();
//
//	parent->image = imread(vImageFile[id]);
//
//	parent->srcLock.release();
//
//	//��ʼ�͸�ͼ����г����طָ�
//	parent->initClusterCenter(id);
//	parent->performSegmentation(id);
//	parent->enforceConnectivity(id);
//
//	//�Ƿ���������طָ�ͼ��
//	if (bSlicOut)
//	{
//		parent->drawContour(id);
//	}
//
//	//����Ⱥϳ���Ҫ�õ�����Ϣ���浽��̬������
//	parent->detLock.acquire();
//
//	parent->vLabel[id] = parent->label;
//	parent->vvLabelPixelCount[id] = parent->vLabelPixelCount;
//	parent->vNumSP[id] = parent->vLabelPixelCount.size();
//
//	//ִ����һ���߳�֮�󣬾Ͷ��߳�����ͳ������һ
//	parent->numLeftThread--;
//
//	parent->detLock.release();
//
//}
