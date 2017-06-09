#include <iostream>
#include <functional>
#include <memory>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <random>

#include "../include/util/Image.hpp"
#include "../include/Matrix.hpp"

#include "../include/util/pca_normalizedData.hpp"
#include "../include/util/zca_whitening_image.hpp"

int main(int argc, char** argv)
{
	if (0)	//SVD Test
	{
#define M 5
#define N 5
#define LDA M
#define LDU M
#define LDVT N
		double a[LDA*N] = {
			8.79,  9.93,  9.83, 5.45,  3.16,
			6.11,  6.91,  5.04, -0.27,  7.98,
			-9.15, -7.93,  4.86, 4.85,  3.01,
			9.57,  1.64,  8.83, 0.74,  5.80,
			-3.49,  4.02,  9.80, 10.00,  4.27/*,
			9.84,  0.15, -8.99, -6.02, -5.31*/
		};

		Matrix<double>X(M, N);
		for (int i = 0; i < X.m; ++i) {
			for (int j = 0; j < X.n; ++j) {
				X(i, j) = a[i*N + j];
			}
		}
		for (int i = 0; i < X.m; ++i) {
			for (int j = 0; j < X.n; ++j) {
				fprintf(stdout, "%f ", X(i, j));
			}
			printf("\n");
		}
		printf("\n");

		SVD<double> svd(X);

		Matrix<double> Y = svd.U;
		for (int i = 0; i < Y.m; ++i) {
			for (int j = 0; j < Y.n; ++j) {
				fprintf(stdout, "%f ", Y(i, j));
			}
			printf("\n");
		}

		printf("\n");
		Y = svd.V.transpose();
		for (int i = 0; i < Y.m; ++i) {
			for (int j = 0; j < Y.n; ++j) {
				fprintf(stdout, "%f ", Y(i, j));
			}
			printf("\n");
		}

		printf("\n");
		Y = svd.W;
		for (int i = 0; i < Y.m; ++i) {
			for (int j = 0; j < Y.n; ++j) {
				fprintf(stdout, "%f ", Y(i, j));
			}
			printf("\n");
		}

		printf("\n");
		Y = (svd.U*svd.W)*svd.V.transpose();
		for (int i = 0; i < Y.m; ++i) {
			for (int j = 0; j < Y.n; ++j) {
				fprintf(stdout, "%f ", Y(i, j));
			}
			printf("\n");
		}

		printf("\n");
		Y = X*svd.inv();
		for (int i = 0; i < Y.m; ++i) {
			for (int j = 0; j < Y.n; ++j) {
				fprintf(stdout, "%f ", Y(i, j));
			}
			printf("\n");
		}

		exit(0);

	}

	if (1)
	{
		Image* img = readImage("aa.bmp");

		//double* whitening_img = ImageTo<double>(img);
		double* whitening_img = image_whitening<double>(img);

		zca_whitening_image(whitening_img, img->width, img->height, 0.01);


		double maxvalue = 0.0;
		for (int i = 0; i < 3 * img->height*img->width; i++)
		{
			if (fabs(whitening_img[i]) > maxvalue) maxvalue = fabs(whitening_img[i]);
		}
		unsigned char *data = new unsigned char[3 * img->height * img->width];

		for (int i = 0; i < img->height; i++)
		{
			for (int j = 0; j < img->width; j++)
			{
				int pos = (i*img->width + j);

				data[3 * pos + 0] = whitening_img[3 * pos + 0] / maxvalue * 127. + 128.;
				data[3 * pos + 1] = whitening_img[3 * pos + 1] / maxvalue * 127. + 128.;
				data[3 * pos + 2] = whitening_img[3 * pos + 2] / maxvalue * 127. + 128.;
			}
		}
		stbi_write_bmp("bb.bmp", img->width, img->height, 3, (void*)data);
		delete[] data;
		delete[] whitening_img;
	}

	return 0;
}