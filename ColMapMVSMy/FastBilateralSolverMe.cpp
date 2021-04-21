#include "FastBilateralSolverMe.h"

bool bInfo = true;


cv::Mat FastBilateralSolverFilter::run1()
{
#ifdef _DEBUG
	std::cout << "Construct Simplified Bilateral Grid Using Reference Image..." << std::endl;
#endif


	cv::Mat reference_yuv;
	cv::cvtColor(reference_bgr_, reference_yuv, COLOR_BGR2YCrCb);

	std::chrono::steady_clock::time_point begin_grid_construction = std::chrono::steady_clock::now();

	cols = reference_yuv.cols;
	rows = reference_yuv.rows;
	npixels = cols*rows;
	int64_t hash_vec[5];
	for (int i = 0; i < 5; ++i)
		hash_vec[i] = static_cast<int64_t>(std::pow(255, i));

	std::unordered_map<int64_t /* hash */, int /* vert id */> hashed_coords;
	hashed_coords.reserve(cols*rows);

	const unsigned char* pref = (const unsigned char*)reference_yuv.data;
	int vert_idx = 0;
	int pix_idx = 0;

	clock_t Tstart, Tend;
	Tstart = clock();
	// construct Splat(Slice) matrices
	splat_idx.resize(npixels);
	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			int64_t coord[5];
			coord[0] = int(x / grid_param.spatialSigma);
			coord[1] = int(y / grid_param.spatialSigma);
			coord[2] = int(pref[0] / grid_param.lumaSigma);
			coord[3] = int(pref[1] / grid_param.chromaSigma);
			coord[4] = int(pref[2] / grid_param.chromaSigma);

			// convert the coordinate to a hash value
			int64_t hash_coord = 0;
			for (int i = 0; i < 5; ++i)
				hash_coord += coord[i] * hash_vec[i];

			// pixels whom are alike will have the same hash value.
			// We only want to keep a unique list of hash values, therefore make sure we only insert
			// unique hash values.
			std::unordered_map<int64_t, int>::iterator it = hashed_coords.find(hash_coord);
			if (it == hashed_coords.end())
			{
				hashed_coords.insert(std::pair<int64_t, int>(hash_coord, vert_idx));
				splat_idx[pix_idx] = vert_idx;
				++vert_idx;
			}
			else
			{
				splat_idx[pix_idx] = it->second;
			}

			pref += 3; // skip 3 bytes (y u v)
			++pix_idx;
		}
	}
	nvertices = hashed_coords.size();

	if (bInfo)
	{
		Tend = clock();
		cout << "Splat Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Tstart = clock();
	// construct Blur matrices
	std::chrono::steady_clock::time_point begin_blur_construction = std::chrono::steady_clock::now();
	Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
	Eigen::VectorXf ones_npixels = Eigen::VectorXf::Ones(npixels);
	blurs_test = ones_nvertices.asDiagonal();
	blurs_test *= 10;
	for (int offset = -1; offset <= 1; ++offset)
	{
		if (offset == 0) continue;
		for (int i = 0; i < 5; ++i)
		{
			Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coords.size(), hashed_coords.size());
			blur_temp.reserve(Eigen::VectorXi::Constant(nvertices, 6));
			int64_t offset_hash_coord = offset * hash_vec[i];
			for (std::unordered_map<int64_t, int>::iterator it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
			{
				int64_t neighb_coord = it->first + offset_hash_coord;
				std::unordered_map<int64_t, int>::iterator it_neighb = hashed_coords.find(neighb_coord);
				if (it_neighb != hashed_coords.end())
				{
					blur_temp.insert(it->second, it_neighb->second) = 1.0f;
					blur_idx.push_back(std::pair<int, int>(it->second, it_neighb->second));
				}
			}
			blurs_test += blur_temp;
		}
	}
	blurs_test.finalize();

	if (bInfo)
	{
		Tend = clock();
		cout << "Blur Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Tstart = clock();
	//bistochastize
	int maxiter = 10;
	n = ones_nvertices;
	m = Eigen::VectorXf::Zero(nvertices);
	for (int i = 0; i < splat_idx.size(); i++) {
		m(splat_idx[i]) += 1.0f;
	}

	Eigen::VectorXf bluredn(nvertices);

	for (int i = 0; i < maxiter; i++) {
		Blur(n, bluredn);
		n = ((n.array()*m.array()).array() / bluredn.array()).array().sqrt();
	}
	Blur(n, bluredn);

	m = n.array() * (bluredn).array();
	Dm = m.asDiagonal();
	Dn = n.asDiagonal();

	if (bInfo)
	{
		Tend = clock();
		cout << "Bistochastize Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

#ifdef _DEBUG
	std::cout << "Splat:" << splat_idx.size() << '\n';
	std::cout << "Blur:" << blurs_test.nonZeros() << '\n';
	std::cout << "Dn:" << Dn.nonZeros() << '\n';
	std::cout << "Dm:" << Dm.nonZeros() << '\n';
#endif


#ifdef _DEBUG
	std::cout << "Solve Sparse Linear System..." << std::endl;
#endif


	Eigen::VectorXf x(npixels);
	Eigen::VectorXf w(npixels);

	if (target_.type() == CV_8UC1)
	{
		const uchar *pft = reinterpret_cast<const uchar*>(target_.data);
		for (int i = 0; i < npixels; i++)
		{
			x(i) = float(pft[i]) / 255.0f;
		}
	}
	else
	{
		//目标图像修改位float类型
		const float *pft = reinterpret_cast<const float*>(target_.data);
		for (int i = 0; i < npixels; i++)
		{
			x(i) = pft[i];
		}
	}

	const uchar *pfc = reinterpret_cast<const uchar*>(confident_.data);
	for (int i = 0; i < npixels; i++)
	{
		w(i) = float(pfc[i]) / 255.0f;
	}
	if (bInfo)
	{
		Tend = clock();
		cout << "PreProcess target Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices, nvertices);
	Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices, nvertices);
	Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices, nvertices);
	Eigen::VectorXf b(nvertices);
	Eigen::VectorXf y(nvertices);
	Eigen::VectorXf w_splat(nvertices);
	Eigen::VectorXf xw(x.size());

	Tstart = clock();
	//construct A
	Splat(w, w_splat);
	A_data = (w_splat).asDiagonal();
	A = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data;
	if (bInfo)
	{
		Tend = clock();
		cout << "Construct A Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Tstart = clock();
	//construct b
	b.setZero();
	for (int i = 0; i < splat_idx.size(); i++) {
		b(splat_idx[i]) += x(i) * w(i);
	}
	if (bInfo)
	{
		Tend = clock();
		cout << "Construct b Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Tstart = clock();
	// solve Ay = b
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
	cg.compute(A);
	cg.setMaxIterations(bs_param.cg_maxiter);
	cg.setTolerance(bs_param.cg_tol);
	y = cg.solve(b);

	if (bInfo)
	{
		Tend = clock();
		cout << "Solve Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

#ifdef _DEBUG
	std::cout << "#iterations:     " << cg.iterations() << std::endl;
	std::cout << "estimated error: " << cg.error() << std::endl;
#endif

	Tstart = clock();
	cv::Mat output(target_.size(), target_.type());
	//slice
	if (target_.type() == CV_8UC1)
	{
		uchar *pftar = (uchar*)(output.data);
		uchar *ptarget = (uchar *)(target_.data);
		for (int i = 0; i < splat_idx.size(); i++)
		{
			if (ptarget[i] == 0.0f)
				continue;
			pftar[i] = y(splat_idx[i]) * 255;
		}
	}
	else
	{
		float *pftar = (float*)(output.data);
		float *ptarget = (float *)(target_.data);
		for (int i = 0; i < splat_idx.size(); i++)
		{
			if (ptarget[i] == 0.0f)
				continue;
			pftar[i] = y(splat_idx[i]);
		}
	}
	if (bInfo)
	{
		Tend = clock();
		cout << "Slice Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	return output;
}


cv::Mat FastBilateralSolverFilter::run()
{
	clock_t Tstart, Tend;
	Tstart = clock();
	//利用参考图像初始化简化的双边网格
	//init(this->reference_, this->grid_param.spatialSigma, this->grid_param.lumaSigma, this->grid_param.chromaSigma);
	init();
	if (bInfo)
	{
		Tend = clock();
		cout << "Init Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}
	Tstart = clock();
	cv::Mat dst(target_.size(), target_.type());
	//利用快速双边求解器，进行深度超分辨率
	//solve(this->target_, this->confident_,dst);
	solve(dst);
	if (bInfo)
	{
		Tend = clock();
		cout << "Solve Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}
	return dst;
}

//void FastBilateralSolverFilter::init(cv::Mat& reference_bgr, double sigma_spatial, double sigma_luma, double sigma_chroma)
void FastBilateralSolverFilter::init()
{

#ifdef _DEBUG
	std::cout << "Construct Simplified Bilateral Grid Using Reference Image..." << std::endl;
#endif


	cv::Mat reference_yuv;
	cv::cvtColor(reference_bgr_, reference_yuv, COLOR_BGR2YCrCb);

	std::chrono::steady_clock::time_point begin_grid_construction = std::chrono::steady_clock::now();

	cols = reference_yuv.cols;
	rows = reference_yuv.rows;
	npixels = cols*rows;
	int64_t hash_vec[5];
	for (int i = 0; i < 5; ++i)
		hash_vec[i] = static_cast<int64_t>(std::pow(255, i));

	std::unordered_map<int64_t /* hash */, int /* vert id */> hashed_coords;
	hashed_coords.reserve(cols*rows);

	const unsigned char* pref = (const unsigned char*)reference_yuv.data;
	int vert_idx = 0;
	int pix_idx = 0;

	clock_t Tstart, Tend;
	Tstart = clock();
	// construct Splat(Slice) matrices
	splat_idx.resize(npixels);
	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			int64_t coord[5];
			coord[0] = int(x / grid_param.spatialSigma);
			coord[1] = int(y / grid_param.spatialSigma);
			coord[2] = int(pref[0] / grid_param.lumaSigma);
			coord[3] = int(pref[1] / grid_param.chromaSigma);
			coord[4] = int(pref[2] / grid_param.chromaSigma);

			// convert the coordinate to a hash value
			int64_t hash_coord = 0;
			for (int i = 0; i < 5; ++i)
				hash_coord += coord[i] * hash_vec[i];

			// pixels whom are alike will have the same hash value.
			// We only want to keep a unique list of hash values, therefore make sure we only insert
			// unique hash values.
			std::unordered_map<int64_t, int>::iterator it = hashed_coords.find(hash_coord);
			if (it == hashed_coords.end())
			{
				hashed_coords.insert(std::pair<int64_t, int>(hash_coord, vert_idx));
				splat_idx[pix_idx] = vert_idx;
				++vert_idx;
			}
			else
			{
				splat_idx[pix_idx] = it->second;
			}

			pref += 3; // skip 3 bytes (y u v)
			++pix_idx;
		}
	}
	nvertices = hashed_coords.size();

	if (bInfo)
	{
		Tend = clock();
		cout << "Splat Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Tstart = clock();
	// construct Blur matrices
	std::chrono::steady_clock::time_point begin_blur_construction = std::chrono::steady_clock::now();
	Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
	Eigen::VectorXf ones_npixels = Eigen::VectorXf::Ones(npixels);
	blurs_test = ones_nvertices.asDiagonal();
	blurs_test *= 10;
	for (int offset = -1; offset <= 1; ++offset)
	{
		if (offset == 0) continue;
		for (int i = 0; i < 5; ++i)
		{
			Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coords.size(), hashed_coords.size());
			blur_temp.reserve(Eigen::VectorXi::Constant(nvertices, 6));
			int64_t offset_hash_coord = offset * hash_vec[i];
			for (std::unordered_map<int64_t, int>::iterator it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
			{
				int64_t neighb_coord = it->first + offset_hash_coord;
				std::unordered_map<int64_t, int>::iterator it_neighb = hashed_coords.find(neighb_coord);
				if (it_neighb != hashed_coords.end())
				{
					blur_temp.insert(it->second, it_neighb->second) = 1.0f;
					blur_idx.push_back(std::pair<int, int>(it->second, it_neighb->second));
				}
			}
			blurs_test += blur_temp;
		}
	}
	blurs_test.finalize();

	if (bInfo)
	{
		Tend = clock();
		cout << "Blur Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Tstart = clock();
	//bistochastize
	int maxiter = 10;
	n = ones_nvertices;
	m = Eigen::VectorXf::Zero(nvertices);
	for (int i = 0; i < splat_idx.size(); i++) {
		m(splat_idx[i]) += 1.0f;
	}

	Eigen::VectorXf bluredn(nvertices);

	for (int i = 0; i < maxiter; i++) {
		Blur(n, bluredn);
		n = ((n.array()*m.array()).array() / bluredn.array()).array().sqrt();
	}
	Blur(n, bluredn);

	m = n.array() * (bluredn).array();
	Dm = m.asDiagonal();
	Dn = n.asDiagonal();

	if (bInfo)
	{
		Tend = clock();
		cout << "Bistochastize Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

#ifdef _DEBUG
	std::cout << "Splat:" << splat_idx.size() << '\n';
	std::cout << "Blur:" << blurs_test.nonZeros() << '\n';
	std::cout << "Dn:" << Dn.nonZeros() << '\n';
	std::cout << "Dm:" << Dm.nonZeros() << '\n';
#endif
}

void FastBilateralSolverFilter::Splat(Eigen::VectorXf& input, Eigen::VectorXf& output)
{
	output.setZero();
	for (int i = 0; i < splat_idx.size(); i++) {
		output(splat_idx[i]) += input(i);
	}

}

void FastBilateralSolverFilter::Blur(Eigen::VectorXf& input, Eigen::VectorXf& output)
{
	output.setZero();
	output = input * 10;
	for (int i = 0; i < blur_idx.size(); i++)
	{
		output(blur_idx[i].first) += input(blur_idx[i].second);
	}
}


void FastBilateralSolverFilter::Slice(Eigen::VectorXf& input, Eigen::VectorXf& output)
{
	output.setZero();
	for (int i = 0; i < splat_idx.size(); i++) {
		output(i) = input(splat_idx[i]);
	}
}


//void FastBilateralSolverFilter::solve(cv::Mat& target,cv::Mat& confidence,cv::Mat &output)
void FastBilateralSolverFilter::solve(cv::Mat &output)
{

#ifdef _DEBUG
	std::cout << "Solve Sparse Linear System..." << std::endl;
#endif

	clock_t Tstart, Tend;
	Tstart = clock();

	Eigen::VectorXf x(npixels);
	Eigen::VectorXf w(npixels);

	if (target_.type() == CV_8UC1)
	{
		const uchar *pft = reinterpret_cast<const uchar*>(target_.data);
		for (int i = 0; i < npixels; i++)
		{
			x(i) = float(pft[i]) / 255.0f;
		}
	}
	else
	{
		//目标图像修改位float类型
		const float *pft = reinterpret_cast<const float*>(target_.data);
		for (int i = 0; i < npixels; i++)
		{
			x(i) = pft[i];
		}
	}

	const uchar *pfc = reinterpret_cast<const uchar*>(confident_.data);
	for (int i = 0; i < npixels; i++)
	{
		w(i) = float(pfc[i]) / 255.0f;
	}
	if (bInfo)
	{
		Tend = clock();
		cout << "PreProcess target Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices, nvertices);
	Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices, nvertices);
	Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices, nvertices);
	Eigen::VectorXf b(nvertices);
	Eigen::VectorXf y(nvertices);
	Eigen::VectorXf w_splat(nvertices);
	Eigen::VectorXf xw(x.size());

	Tstart = clock();
	//construct A
	Splat(w, w_splat);
	A_data = (w_splat).asDiagonal();
	A = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data;
	if (bInfo)
	{
		Tend = clock();
		cout << "Construct A Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Tstart = clock();
	//construct b
	b.setZero();
	for (int i = 0; i < splat_idx.size(); i++) {
		b(splat_idx[i]) += x(i) * w(i);
	}
	if (bInfo)
	{
		Tend = clock();
		cout << "Construct b Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

	Tstart = clock();
	// solve Ay = b
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
	cg.compute(A);
	cg.setMaxIterations(bs_param.cg_maxiter);
	cg.setTolerance(bs_param.cg_tol);
	y = cg.solve(b);

	if (bInfo)
	{
		Tend = clock();
		cout << "Solve Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}

#ifdef _DEBUG
	std::cout << "#iterations:     " << cg.iterations() << std::endl;
	std::cout << "estimated error: " << cg.error() << std::endl;
#endif

	Tstart = clock();
	//slice
	if (target_.type() == CV_8UC1)
	{
		uchar *pftar = (uchar*)(output.data);
		for (int i = 0; i < splat_idx.size(); i++)
		{
			pftar[i] = y(splat_idx[i]) * 255;
		}
	}
	else
	{
		float *pftar = (float*)(output.data);
		for (int i = 0; i < splat_idx.size(); i++)
		{
			pftar[i] = y(splat_idx[i]);
		}
	}
	if (bInfo)
	{
		Tend = clock();
		cout << "Slice Time:" << (float)(Tend - Tstart) / CLOCKS_PER_SEC << "s" << endl;
	}
}