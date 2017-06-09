#ifndef  _ZCA_WHITENING_HPP

// Image.hpp ÇÊÇËâ∫Ç≈include Ç∑ÇÈïKóvÇ™Ç†ÇÈÅB
template<class T>
inline T* zca_whitening_image(Image* img, const double eps = 1.0E-5)
{
	Matrix<T> IMG(img->height, img->width * 3);

#pragma omp parallel for
	for (int i = 0; i < img->height; i++)
	{
		for (int j = 0; j < img->width; j++)
		{
			int pos = (i*img->width + j);
			IMG(i, 3 * j + 0) = img->data[pos].r;
			IMG(i, 3 * j + 1) = img->data[pos].g;
			IMG(i, 3 * j + 2) = img->data[pos].b;
		}
	}

	//Matrix<T> ZCAIMG = zca_whitening_matrix<T>(IMG, eps)*IMG;
	Matrix<T> ZCAIMG = zca_whitening_matrix2(IMG, eps)*IMG;

	T* zcaimg = new T[3 * img->height*img->width];
#pragma omp parallel for
	for (int i = 0; i < img->height; i++)
	{
		for (int j = 0; j < img->width; j++)
		{
			int pos = (i*img->width + j);

			zcaimg[3 * pos + 0] = ZCAIMG(i, 3 * j + 2);
			zcaimg[3 * pos + 1] = ZCAIMG(i, 3 * j + 1);
			zcaimg[3 * pos + 2] = ZCAIMG(i, 3 * j + 0);
		}
	}

	return zcaimg;
}

template<class T>
inline void zca_whitening_image(T* d, const int x, const int y, const double eps = 1.0E-5)
{
	double* tmp = pca_normalizedData<double>(d, x*y * 3);

	Image* img = ToImage<T>(tmp, x, y);
	T* ret = zca_whitening_image<T>(img, eps);
	delete img;

	memcpy(d, ret, sizeof(T)*x*y * 3);
}

#endif // ! _ZCA_WHITENING_HPP
