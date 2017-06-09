#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#include <assert.h>

#include "matrix/SVD.h"

#define USE_LAPACK

#ifdef USE_LAPACK
#include "util/clapack_util.h"
#endif

template<class T>
struct Matrix;

#include "matcalc.hpp"

#include <omp.h>

template<class T>
struct Matrix
{
	int m, n;
	T* v;
	
	inline Matrix(): m(0), n(0), v(NULL) { }
	inline Matrix( const int& m, const int& n ) :m(m), n(n)
	{
		v = new T[m*n];
	}

	inline Matrix( const std::vector<T>& v ) :m(v.size()), n(1)
	{
		this->v = new T[m*n];
		for( int i = 0; i < m; ++i ) this->v[i] = v[i];
	}

	inline Matrix( const Matrix<T>& mat )
	{
		m = mat.m; n = mat.n;
		if( m == 0 || n == 0 ){
			v = NULL;
		}
		else{
			v = new T[m*n];
			const int mn = m*n;
#pragma omp parallel for
			for( int i = 0; i < mn; ++i )  v[i] = mat.v[i];
		}
	}

	Matrix<T>& operator = ( const Matrix& mat )
	{
		if( v != NULL ) delete [] v;

		m = mat.m; n = mat.n;
		if( m == 0 || n == 0 ){
			v = NULL;
			return *this;
		}
			
		v = new T[m*n];
		const int mn = m*n;
#pragma omp parallel for
		for( int i = 0; i < mn; ++i )  v[i] = mat.v[i];

		return *this;
	}

	~Matrix ()
	{
		if( v != NULL ) delete [] v;
	}
	
	static Matrix<T> eye ( const int& m, const int& n )
	{
		Matrix<T> ret(m, n);
#pragma omp parallel for
		for( int i = 0; i < std::min(m,n); ++i ) ret(i,i) = 1.0;
		return ret;
	}

	static Matrix<T> values(const int& m, const int& n, const double a)
	{
		Matrix<T> ret(m, n);
		const int mn = m*n;
#pragma omp parallel for
		for (int i = 0; i < mn; ++i) ret.v[i] = a;
		return ret;
	}

	static Matrix<T> zeros ( const int& m, const int& n )
	{
		Matrix<T> ret(m, n);
		const int mn = m*n;
#pragma omp parallel for
		for( int i = 0; i < mn; ++i ) ret.v[i] = 0.0;
		return ret;
	}

	static Matrix<T> transpose( const Matrix<T>& mat )
	{
 		int m = mat.m, n = mat.n;
 		Matrix<T> ret(n, m);

		transpose_(mat.v, m, n, ret.v);
		return ret;
	}
	
	inline Matrix<T> transpose()
	{
		Matrix<T> ret(n, m);
		transpose_(v, m, n, ret.v);
		return ret;
	}

	static Matrix<T> hadamard ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);


		hadamard_(m1.v, m1.m, m1.n, m2.v, m2.m, m2.n, ret.v);

		return ret;
	}
	Matrix<T> hadamard(const Matrix<T>& mat)
	{
		Matrix<T> ret(m, n);

		hadamard_(v, m, n, mat.v, mat.m, mat.n, ret.v);

		return ret;
	}

	inline const T& operator () ( const int i, const int j ) const
	{
		return v[i*n + j];
	}

	inline T& operator () ( const int i, const int j )
	{
		return v[i*n + j];
	}

	Matrix<T>& operator += ( const Matrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;

		plus_eqd_(v, m, n, m1.v, m1.m, m1.n);

		return *this;
	}

	Matrix<T>& operator -= ( const Matrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
		const int mn = m*n;
#pragma omp parallel for
		for( int i = 0; i < mn; ++i ) this->v[i] -= m1.v[i];

		return *this;
	}

	Matrix<T>& operator *= ( const Matrix<T>& m1 )
	{
		*this = *this*m1;
		return *this;
	}

	Matrix<T>& operator *= ( const T& c )
	{
		const int mn = m*n;
		scara_prod_(&v[0], m, n, c);

		return *this;
	}
	
	Matrix<T>& operator /= ( const T& c )
	{
		const int mn = m*n;
#pragma omp parallel for
		for( int i = 0; i < mn; ++i ) this->v[i] /= c;
		return *this;
	}

	friend Matrix<T> operator + ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);

		plus_(m1.v, m1.m, m1.n, m2.v, m2.m, m2.n, ret.v);
		return ret;
	}
	
	friend Matrix<T> operator - ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);

		const int mn = m*n;
#pragma omp parallel for
		for( int i = 0; i < mn; ++i ) ret.v[i] = m1.v[i] - m2.v[i];

		return ret;
	}

	friend Matrix<T> operator * ( const T& c, const Matrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);
		const int mn = m*n;
		prod_(m1.v, m1.m, m1.n, c, ret.v);

		return ret;
	}

	friend Matrix<T> operator * ( const Matrix<T>& m1, const T& c )
	{
		return c*m1;
	}
	 
	friend Matrix<T> operator / ( const Matrix<T>& m1, const T& c )
	{
		return (1.0/c)*m1;
	}

	friend std::ostream& operator << ( std::ostream& os, const Matrix<T>& A )
	{
		for( int i = 0; i < A.m; ++i ){
			for( int j = 0; j < A.n; ++j ){
				if( j != 0 ) os << " ";
				os << std::scientific << std::setprecision(3) << std::setw(10) << A(i,j);
			}
			std::cout << std::endl;
		}
		return os;
	}

	Matrix<T> diag(Matrix<T>& X)
	{
		Matrix<T> d = Matrix<T>::zeros(X.n, X.n);

		for (int i = 0; i < X.n; i++) d(i, i) = X(i, i);

		return d;
	}

	void apply ( const std::function<double(const double&)>& func )
	{
		const int mn = m*n;
		int i = 0;
#pragma omp parallel for private(i)
		for( i = 0; i < mn; i += 4 )
		{
			v[i  ] = func(v[i  ]);
			v[i+1] = func(v[i+1]);
			v[i+2] = func(v[i+2]);
			v[i+3] = func(v[i+3]);
		}
		for ( i; j < mn; ++i)
		{
			v[i  ] = func(this->v[i  ]);
		}
	}
	Matrix<T> Abs()
	{
		Matrix<T> d = *this;

		for (int i = 0; i < d.m*d.n; i++) d.v[i] = fabs(d.v[i]);

		return d;
	}
	double Max() const
	{
		double mx = -1.0E32;
		for (int i = 0; i < m*n; i++)
		{
			if (mx < v[i]) mx = v[i];
		}
		return mx;
	}
	double Min() const
	{
		double mx = 1.0E32;
		for (int i = 0; i < m*n; i++)
		{
			if (mx > v[i]) mx = v[i];
		}
		return mx;
	}
};

Matrix<double> operator * ( const Matrix<double>& m1, const Matrix<double>& m2 )
{
	int m = m1.m, n = m2.n, l = m1.n;
		
	Matrix<double> ret(m, n);
	mull_(m1.v, m1.m, m1.n, m2.v, m2.m, m2.n, ret.v);		

	return ret;
}


#ifndef USE_LAPACK
template<class T>
class SVD
{
	nrSVD<T> svd;
public:
	Matrix<T> U;
	Matrix<T> V;
	Matrix<T> W;
	Matrix<T> *A;

	//void svdcmp(float **a, int m, int n, float w[], float **v)
	SVD(Matrix<T>& mat)
	{
		A = &mat;

		const int m = mat.m;
		const int n = mat.n;

		T **u = svd.matrix(1, m, 1, n);
		T **v = svd.matrix(1, n, 1, n);
		T *w = svd.vector(1, n);

		U = Matrix<T>(m, n);
		V = Matrix<T>(n, n);
		W = Matrix<T>(n, n);

		for (int i = 1; i < m + 1; i++)
			for (int j = 1; j < n + 1; j++)
				u[i][j] = mat(i - 1, j - 1);

		svd.svdcmp(u, m, n, w, v);
		
		for (int i = 1; i < m + 1; i++)
			for (int j = 1; j < n + 1; j++)
				U(i - 1, j - 1) = u[i][j];

		for (int i = 1; i < n + 1; i++)
			for (int j = 1; j < n + 1; j++)
				V(i - 1, j - 1) = v[i][j];
		
		//W.zeros(n, n);
		for (int i = 1; i < n + 1; i++)
			for (int j = 1; j < n + 1; j++)
				W(i - 1, j - 1) = 0.0;

		for (int i = 1; i < n + 1; i++)
				W(i - 1, i - 1) = w[i];

#if 0
		printf("Product u*w*(v-transpose):\n");
		float **aa = svd.matrix(1, m, 1, n);
		for (int k = 1; k <= m; k++) {
			for (int l = 1; l <= n; l++) {
				aa[k][l] = 0.0;
				for (int j = 1; j <= n; j++)
					aa[k][l] += u[k][j] * w[j] * v[l][j];
			}
			for (int l = 1; l <= n; l++) printf("%.6f ", fabs(mat(k-1,l-1) - aa[k][l]));
			printf("\n");
		}
		svd.free_matrix(aa, 1, m, 1, n);
#endif

		svd.free_matrix(v, 1, n, 1, n);
		svd.free_matrix(u, 1, m, 1, n);
		svd.free_vector(w, 1, n);
	}

	Matrix<T> inv() const
	{
		//A^-1 = V * [diag(1/Wj)] *Vt;

		Matrix<T> Ws = W;
		for (int i = 0; i < W.n; i++)
		{
			Ws(i, i) = 1.0 / W(i, i);
		}
		Matrix<T> Ut = U.transpose(U);
#if 0
		Matrix<T> t = *A*(V*Ws*Ut);
		std::cout << t << std::endl;
#endif
		return V*Ws*Ut;
	}

};
#endif

#ifdef USE_LAPACK

class SingularValueDecomposition
{
public:
	long int numberOfSingularValues;
	double *s;
	double *u;
	double *vt;

	SingularValueDecomposition(int m_, int n_, double *a)
	{
		char jobu = 'A', jobvt = 'S';
		long int m, n, lda, ldu, ldvt, lwork, info;
		m = lda = ldu = m_;
		n = ldvt = n_;
		lwork = std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n));

		u = new double[m*m];
		vt = new double[n*n];
		numberOfSingularValues = std::min(m, n);
		s = new double[numberOfSingularValues];
		double* work = new double[lwork];

		double* B = new double[m*n];
		memcpy(B, a, sizeof(double)*m*n);

		long int lwork_tmp = -1;
		dgesvd_(&jobu, &jobvt, &m, &n, B, &lda, s, u, &ldu, vt, &ldvt, work, &lwork_tmp, &info);
		if (info == 0)
		{
			lwork = static_cast<int>(work[0]);
			delete[] work;
			work = new double[lwork];
		}

		//memcpy(B, a, sizeof(double)*m*n);
		//memset(u, '\0', sizeof(double)*m*m);
		//memset(s, '\0', sizeof(double)*numberOfSingularValues);
		//memset(vt, '\0', sizeof(double)*n*n);
		dgesvd_(&jobu, &jobvt, &m, &n, B, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
		if (info != 0)
		{
			printf("ERROR dgesvd_ %d\n", info);
		}
		delete[] B;
		delete[] work;
	}

	~SingularValueDecomposition()
	{
		free(s);
		free(u);
		free(vt);
	}
};

template<class T>
class SVD
{
	SingularValueDecomposition* svd;
public:
	Matrix<T> U;
	Matrix<T> V;
	Matrix<T> W;
	Matrix<T> *A;

	~SVD()
	{
		delete svd;
	}

	//void svdcmp(float **a, int m, int n, float w[], float **v)
	SVD(Matrix<T>& mat)
	{
		A = &mat;

		const int m = mat.m;
		const int n = mat.n;

		svd = new SingularValueDecomposition(m, n, A->transpose().v);

		U = Matrix<T>(m, n);
		V = Matrix<T>(n, n);
		W = Matrix<T>(svd->numberOfSingularValues, svd->numberOfSingularValues);


		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				U(i, j) = svd->u[j*m + i];

		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				V(j, i) = svd->vt[j*n + i];

		//W.zeros(n, n);
		for (int i = 0; i < svd->numberOfSingularValues; i++)
			for (int j = 0; j < svd->numberOfSingularValues; j++)
				W(i, j) = 0.0;

		for (int i = 0; i < svd->numberOfSingularValues; i++)
			W(i, i) = svd->s[i];

#if 0
		printf("u*w*(v-transpose):\n");
		{
			Matrix<T> x = U*W*V;
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < n; j++)
				{
					fprintf(stdout, "%.4f ", fabs(x(i, j) - mat(i, j)));
				}
				printf("\n");
			}
		}
#endif
	}

	Matrix<T> inv() const
	{
		//A^-1 = V * [diag(1/Wj)] *Vt;

		Matrix<T> Ws = W;
		for (int i = 0; i < W.n; i++)
		{
			Ws(i, i) = 1.0 / W(i, i);
		}
		Matrix<T> Ut = U.transpose(U);
#if 0
		Matrix<T> t = *A*(V*Ws*Ut);
		std::cout << t << std::endl;
#endif
		return V*Ws*Ut;
	}

};
#endif

template<class T>
inline Matrix<double> zca_whitening_matrix(Matrix<T>& X, const T eps = 1.0e-5, bool bias = false)
{

	//const double eps = 1.0E-5;	//Whitening constant: prevents division by zero

	const int sz = X.m*X.n;
	std::vector<T> sum(X.m, 0.0);

	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			sum[i] += X(i, j);
		}
	}

	std::vector<T> mu(X.m, 0.0);
#pragma omp parallel for
	for (int i = 0; i < X.m; i++)
	{
		mu[i] = sum[i] / X.n;
		//printf("%f\n", mu[i]);
	}


	//•s•Î‹¤•ªŽUs—ñ	Sigma = (X-mu) * (X-mu)' / N
	Matrix<T> sigma = Matrix<T>::zeros(X.m, X.n);
#pragma omp parallel for
	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			sigma(i, j) = (X(i, j) - mu[i]);
		}
	}
	//std::cout << sigma << "\n";
	if (bias)
	{
		sigma = sigma*Matrix<T>::transpose(sigma) / X.n;
	}
	else
	{
		sigma = sigma*Matrix<T>::transpose(sigma) / (X.n - 1.0);
	}

	//std::cout << sigma << "\n";

	//Singular Value Decomposition. X = U * np.diag(W) * V
	SVD<T> svd(sigma);
	//std::cout << "U" << svd.U << "\n";
	//std::cout << "W" << svd.W << "\n";
	//std::cout << "V" << svd.V << "\n";

	for (int i = 0; i < svd.W.n; i++) svd.W(i, i) = 1.0 / sqrt(svd.W(i, i) + eps);

	//std::cout << "W" << svd.W << "\n";

	//ZCA Whitening matrix: U * diag(1/sqrt(W+eps) * U'
	return svd.U*(svd.W * Matrix<T>::transpose(svd.U));
}

#if 10
template<class T>
inline Matrix<T> zca_whitening_matrix2(Matrix<T>& X, const double eps = 1.0e-5, bool bias = false)
{

	//const double eps = 1.0E-8;	//Whitening constant: prevents division by zero

	const int sz = X.m*X.n;
	double sum = 0.0;

	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			sum += X(i, j);
		}
	}

	double mu = 0.0;
	mu = sum / sz;


	//•s•Î‹¤•ªŽUs—ñ	Sigma = (X-mu) * (X-mu)' / N
	Matrix<T> sigma = Matrix<T>::zeros(X.m, X.n);
#pragma omp parallel for
	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			sigma(i, j) = (X(i, j) - mu);
		}
	}
	//std::cout << sigma << "\n";
	if (bias)
	{
		sigma = sigma*Matrix<T>::transpose(sigma) / X.n;
	}
	else
	{
		sigma = sigma*Matrix<T>::transpose(sigma) / (X.n - 1.0);
	}

	//std::cout << sigma << "\n";

	//Singular Value Decomposition. X = U * np.diag(W) * V
	SVD<T> svd(sigma);
	//std::cout << "U" << svd.U << "\n";
	//std::cout << "W" << svd.W << "\n";
	//std::cout << "V" << svd.V << "\n";

	for (int i = 0; i < svd.W.n; i++) svd.W(i, i) = 1.0 / sqrt(svd.W(i, i) + eps);

	//std::cout << "W" << svd.W << "\n";

	//ZCA Whitening matrix: U * diag(1/sqrt(W+eps) * U'
	return svd.U*(svd.W * svd.U.transpose());
}
#endif

#endif
