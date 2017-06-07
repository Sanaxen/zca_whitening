#if !defined(_SVD_H)
#define _SVD_H

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#define NR_END 1
#define FREE_ARG char*

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

template<class T>
class nrSVD
{
	T maxarg1, maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
        (maxarg1) : (maxarg2))
	int iminarg1, iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
        (iminarg1) : (iminarg2))
	T sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)

	inline void nrerror(const char* msg)
	{
		fprintf(stderr, "svd run-time error...[%s]\n", msg);
		//exit(1);
	}


public:
	inline T *vector(long nl, long nh)
	{
		T *v;

		v = (T *)malloc((size_t)((nh - nl + 1 + NR_END) * sizeof(T)));
		if (!v) nrerror("memory allocation error");
		return v - nl + NR_END;
	}

	inline void free_vector(T *v, long nl, long nh)
	{
		free((FREE_ARG)(v + nl - NR_END));
	}
	inline T **matrix(long nrl, long nrh, long ncl, long nch)
		/* allocate a T matrix with subscript range m[nrl..nrh][ncl..nch] */
	{
		long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
		T **m;

		/* allocate pointers to rows */
		m = (T **)malloc((size_t)((nrow + NR_END) * sizeof(T*)));
		if (!m) nrerror("allocation error: matrix");
		m += NR_END;
		m -= nrl;

		/* allocate rows and set pointers to them */
		m[nrl] = (T *)malloc((size_t)((nrow*ncol + NR_END) * sizeof(T)));
		if (!m[nrl]) nrerror("allocation error: matrix");
		m[nrl] += NR_END;
		m[nrl] -= ncl;

		for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

		/* return pointer to array of pointers to rows */
		return m;
	}
	inline void free_matrix(T **m, long nrl, long nrh, long ncl, long nch)
		/* free a T matrix allocated by matrix() */
	{
		free((FREE_ARG)(m[nrl] + ncl - NR_END));
		free((FREE_ARG)(m + nrl - NR_END));
	}

	void svbksb(T **u, T w[], T **v, int m, int n, T b[], T x[])
	{
		int jj, j, i;
		T s, *tmp;

		tmp = vector(1, n);
		for (j = 1; j <= n; j++) {
			s = 0.0;
			if (w[j]) {
				for (i = 1; i <= m; i++) s += u[i][j] * b[i];
				s /= w[j];
			}
			tmp[j] = s;
		}
		for (j = 1; j <= n; j++) {
			s = 0.0;
			for (jj = 1; jj <= n; jj++) s += v[j][jj] * tmp[jj];
			x[j] = s;
		}
		free_vector(tmp, 1, n);
	}

	void svdcmp(T **a, int m, int n, T *w, T **v)
	{
		//double pythag(double a, double b);
		int flag, i, its, j, jj, k, l, nm;
		T anorm, c, f, g, h, s, scale, x, y, z, *rv1;

		rv1 = vector(1, n);
		g = scale = anorm = 0.0;
		for (i = 1; i <= n; i++) {
			l = i + 1;
			rv1[i] = scale*g;
			g = s = scale = 0.0;
			if (i <= m) {
				for (k = i; k <= m; k++) scale += fabs(a[k][i]);
				if (scale) {
					for (k = i; k <= m; k++) {
						a[k][i] /= scale;
						s += a[k][i] * a[k][i];
					}
					f = a[i][i];
					g = -SIGN(sqrt(s), f);
					h = f*g - s;
					a[i][i] = f - g;
					for (j = l; j <= n; j++) {
						for (s = 0.0, k = i; k <= m; k++) s += a[k][i] * a[k][j];
						f = s / h;
						for (k = i; k <= m; k++) a[k][j] += f*a[k][i];
					}
					for (k = i; k <= m; k++) a[k][i] *= scale;
				}
			}
			w[i] = scale *g;
			g = s = scale = 0.0;
			if (i <= m && i != n) {
				for (k = l; k <= n; k++) scale += fabs(a[i][k]);
				if (scale) {
					for (k = l; k <= n; k++) {
						a[i][k] /= scale;
						s += a[i][k] * a[i][k];
					}
					f = a[i][l];
					g = -SIGN(sqrt(s), f);
					h = f*g - s;
					a[i][l] = f - g;
					for (k = l; k <= n; k++) rv1[k] = a[i][k] / h;
					for (j = l; j <= m; j++) {
						for (s = 0.0, k = l; k <= n; k++) s += a[j][k] * a[i][k];
						for (k = l; k <= n; k++) a[j][k] += s*rv1[k];
					}
					for (k = l; k <= n; k++) a[i][k] *= scale;
				}
			}
			anorm = FMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
		}
		for (i = n; i >= 1; i--) {
			if (i < n) {
				if (g) {
					for (j = l; j <= n; j++)
						v[j][i] = (a[i][j] / a[i][l]) / g;
					for (j = l; j <= n; j++) {
						for (s = 0.0, k = l; k <= n; k++) s += a[i][k] * v[k][j];
						for (k = l; k <= n; k++) v[k][j] += s*v[k][i];
					}
				}
				for (j = l; j <= n; j++) v[i][j] = v[j][i] = 0.0;
			}
			v[i][i] = 1.0;
			g = rv1[i];
			l = i;
		}
		for (i = IMIN(m, n); i >= 1; i--) {
			l = i + 1;
			g = w[i];
			for (j = l; j <= n; j++) a[i][j] = 0.0;
			if (g) {
				g = 1.0 / g;
				for (j = l; j <= n; j++) {
					for (s = 0.0, k = l; k <= m; k++) s += a[k][i] * a[k][j];
					f = (s / a[i][i])*g;
					for (k = i; k <= m; k++) a[k][j] += f*a[k][i];
				}
				for (j = i; j <= m; j++) a[j][i] *= g;
			}
			else for (j = i; j <= m; j++) a[j][i] = 0.0;
			++a[i][i];
		}
		for (k = n; k >= 1; k--) {
			for (its = 1; its <= 300; its++) {
				flag = 1;
				for (l = k; l >= 1; l--) {
					nm = l - 1;
					if ((T)(fabs(rv1[l]) + anorm) == anorm) {
						flag = 0;
						break;
					}
					if ((T)(fabs(w[nm]) + anorm) == anorm) break;
				}
				if (flag) {
					c = 0.0;
					s = 1.0;
					for (i = l; i <= k; i++) {
						f = s*rv1[i];
						rv1[i] = c*rv1[i];
						if ((T)(fabs(f) + anorm) == anorm) break;
						g = w[i];
						h = pythag(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g*h;
						s = -f*h;
						for (j = 1; j <= m; j++) {
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = y*c + z*s;
							a[j][i] = z*c - y*s;
						}
					}
				}
				z = w[k];
				if (l == k) {
					if (z < 0.0) {
						w[k] = -z;
						for (j = 1; j <= n; j++) v[j][k] = -v[j][k];
					}
					break;
				}
				if (its == 300) nrerror("no convergence in 300 svdcmp iterations");
				x = w[l];
				nm = k - 1;
				y = w[nm];
				g = rv1[nm];
				h = rv1[k];
				f = ((y - z)*(y + z) + (g - h)*(g + h)) / (2.0*h*y);
				g = pythag(f, 1.0);
				f = ((x - z)*(x + z) + h*((y / (f + SIGN(g, f))) - h)) / x;
				c = s = 1.0;
				for (j = l; j <= nm; j++) {
					i = j + 1;
					g = rv1[i];
					y = w[i];
					h = s*g;
					g = c*g;
					z = pythag(f, h);
					rv1[j] = z;
					c = f / z;
					s = h / z;
					f = x*c + g*s;
					g = g*c - x*s;
					h = y*s;
					y *= c;
					for (jj = 1; jj <= n; jj++) {
						x = v[jj][j];
						z = v[jj][i];
						v[jj][j] = x*c + z*s;
						v[jj][i] = z*c - x*s;
					}
					z = pythag(f, h);
					w[j] = z;
					if (z) {
						z = 1.0 / z;
						c = f*z;
						s = h*z;
					}
					f = c*g + s*y;
					x = c*y - s*g;
					for (jj = 1; jj <= m; jj++) {
						y = a[jj][j];
						z = a[jj][i];
						a[jj][j] = y*c + z*s;
						a[jj][i] = z*c - y*s;
					}
				}
				rv1[l] = 0.0;
				rv1[k] = f;
				w[k] = x;
			}
		}
		free_vector(rv1, 1, n);
	}

	T pythag(T a, T b)
	{
		T absa, absb;
		absa = fabs(a);
		absb = fabs(b);
		if (absa > absb) return absa*sqrt(1.0 + SQR(absb / absa));
		else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0 + SQR(absa / absb)));
	}

#if 0
	static void nrerror(char error_text[])
		/* Numerical Recipes standard error handler */
	{
		fprintf(stderr, "êîílââéZ run-time error...\n");
		//fprintf(stderr,"%s\n",error_text);
		//fprintf(stderr,"...now exiting to system...\n");
		//exit(1);
	}

	static float *vector(long nl, long nh)
		/* allocate a float vector with subscript range v[nl..nh] */
	{
		float *v;

		v = (float *)malloc((size_t)((nh - nl + 1 + NR_END) * sizeof(float)));
		if (!v) nrerror("allocation error: vector");
		return v - nl + NR_END;
	}

	int *ivector(long nl, long nh)
		/* allocate an int vector with subscript range v[nl..nh] */
	{
		int *v;

		v = (int *)malloc((size_t)((nh - nl + 1 + NR_END) * sizeof(int)));
		if (!v) nrerror("allocation error: vector");
		return v - nl + NR_END;
	}

	unsigned char *cvector(long nl, long nh)
		/* allocate an unsigned char vector with subscript range v[nl..nh] */
	{
		unsigned char *v;

		v = (unsigned char *)malloc((size_t)((nh - nl + 1 + NR_END) * sizeof(unsigned char)));
		if (!v) nrerror("allocation error: vector");
		return v - nl + NR_END;
	}

	unsigned long *lvector(long nl, long nh)
		/* allocate an unsigned long vector with subscript range v[nl..nh] */
	{
		unsigned long *v;

		v = (unsigned long *)malloc((size_t)((nh - nl + 1 + NR_END) * sizeof(long)));
		if (!v) nrerror("allocation error: vector");
		return v - nl + NR_END;
	}

	double *dvector(long nl, long nh)
		/* allocate a double vector with subscript range v[nl..nh] */
	{
		double *v;

		v = (double *)malloc((size_t)((nh - nl + 1 + NR_END) * sizeof(double)));
		if (!v) nrerror("allocation error: vector");
		return v - nl + NR_END;
	}

	float **matrix(long nrl, long nrh, long ncl, long nch)
		/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
	{
		long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
		float **m;

		/* allocate pointers to rows */
		m = (float **)malloc((size_t)((nrow + NR_END) * sizeof(float*)));
		if (!m) nrerror("allocation error: matrix");
		m += NR_END;
		m -= nrl;

		/* allocate rows and set pointers to them */
		m[nrl] = (float *)malloc((size_t)((nrow*ncol + NR_END) * sizeof(float)));
		if (!m[nrl]) nrerror("allocation error: matrix");
		m[nrl] += NR_END;
		m[nrl] -= ncl;

		for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

		/* return pointer to array of pointers to rows */
		return m;
	}

	double **dmatrix(long nrl, long nrh, long ncl, long nch)
		/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
	{
		long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
		double **m;

		/* allocate pointers to rows */
		m = (double **)malloc((size_t)((nrow + NR_END) * sizeof(double*)));
		if (!m) nrerror("allocation error: matrix");
		m += NR_END;
		m -= nrl;

		/* allocate rows and set pointers to them */
		m[nrl] = (double *)malloc((size_t)((nrow*ncol + NR_END) * sizeof(double)));
		if (!m[nrl]) nrerror("allocation error: matrix");
		m[nrl] += NR_END;
		m[nrl] -= ncl;

		for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

		/* return pointer to array of pointers to rows */
		return m;
	}

	int **imatrix(long nrl, long nrh, long ncl, long nch)
		/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
	{
		long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
		int **m;

		/* allocate pointers to rows */
		m = (int **)malloc((size_t)((nrow + NR_END) * sizeof(int*)));
		if (!m) nrerror("allocation error: matrix");
		m += NR_END;
		m -= nrl;


		/* allocate rows and set pointers to them */
		m[nrl] = (int *)malloc((size_t)((nrow*ncol + NR_END) * sizeof(int)));
		if (!m[nrl]) nrerror("allocation error: matrix");
		m[nrl] += NR_END;
		m[nrl] -= ncl;

		for (i = nrl + 1; i <= nrh; i++) m[i] = m[i - 1] + ncol;

		/* return pointer to array of pointers to rows */
		return m;
	}

	float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
		long newrl, long newcl)
		/* point a submatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
	{
		long i, j, nrow = oldrh - oldrl + 1, ncol = oldcl - newcl;
		float **m;

		/* allocate array of pointers to rows */
		m = (float **)malloc((size_t)((nrow + NR_END) * sizeof(float*)));
		if (!m) nrerror("allocation error: matrix");
		m += NR_END;
		m -= newrl;

		/* set pointers to rows */
		for (i = oldrl, j = newrl; i <= oldrh; i++, j++) m[j] = a[i] + ncol;

		/* return pointer to array of pointers to rows */
		return m;
	}

	float **convert_matrix(float *a, long nrl, long nrh, long ncl, long nch)
		/* allocate a float matrix m[nrl..nrh][ncl..nch] that points to the matrix
		declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
		and ncol=nch-ncl+1. The routine should be called with the address
		&a[0][0] as the first argument. */
	{
		long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
		float **m;

		/* allocate pointers to rows */
		m = (float **)malloc((size_t)((nrow + NR_END) * sizeof(float*)));
		if (!m) nrerror("allocation error: matrix");
		m += NR_END;
		m -= nrl;

		/* set pointers to rows */
		m[nrl] = a - ncl;
		for (i = 1, j = nrl + 1; i < nrow; i++, j++) m[j] = m[j - 1] + ncol;
		/* return pointer to array of pointers to rows */
		return m;
	}

	float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
		/* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
	{
		long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1, ndep = ndh - ndl + 1;
		float ***t;

		/* allocate pointers to pointers to rows */
		t = (float ***)malloc((size_t)((nrow + NR_END) * sizeof(float**)));
		if (!t) nrerror("allocation error: tensor");
		t += NR_END;
		t -= nrl;

		/* allocate pointers to rows and set pointers to them */
		t[nrl] = (float **)malloc((size_t)((nrow*ncol + NR_END) * sizeof(float*)));
		if (!t[nrl]) nrerror("allocation error: tensor");
		t[nrl] += NR_END;
		t[nrl] -= ncl;

		/* allocate rows and set pointers to them */
		t[nrl][ncl] = (float *)malloc((size_t)((nrow*ncol*ndep + NR_END) * sizeof(float)));
		if (!t[nrl][ncl]) nrerror("allocation error: tensor");
		t[nrl][ncl] += NR_END;
		t[nrl][ncl] -= ndl;

		for (j = ncl + 1; j <= nch; j++) t[nrl][j] = t[nrl][j - 1] + ndep;
		for (i = nrl + 1; i <= nrh; i++) {
			t[i] = t[i - 1] + ncol;
			t[i][ncl] = t[i - 1][ncl] + ncol*ndep;
			for (j = ncl + 1; j <= nch; j++) t[i][j] = t[i][j - 1] + ndep;
		}

		/* return pointer to array of pointers to rows */
		return t;
	}

	static void free_vector(float *v, long nl, long nh)
		/* free a float vector allocated with vector() */
	{
		free((FREE_ARG)(v + nl - NR_END));
	}

	void free_ivector(int *v, long nl, long nh)
		/* free an int vector allocated with ivector() */
	{
		free((FREE_ARG)(v + nl - NR_END));
	}

	void free_cvector(unsigned char *v, long nl, long nh)
		/* free an unsigned char vector allocated with cvector() */
	{
		free((FREE_ARG)(v + nl - NR_END));
	}

	void free_lvector(unsigned long *v, long nl, long nh)
		/* free an unsigned long vector allocated with lvector() */
	{
		free((FREE_ARG)(v + nl - NR_END));
	}

	void free_dvector(double *v, long nl, long nh)
		/* free a double vector allocated with dvector() */
	{
		free((FREE_ARG)(v + nl - NR_END));
	}

	void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
		/* free a float matrix allocated by matrix() */
	{
		free((FREE_ARG)(m[nrl] + ncl - NR_END));
		free((FREE_ARG)(m + nrl - NR_END));
	}

	void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
		/* free a double matrix allocated by dmatrix() */
	{
		free((FREE_ARG)(m[nrl] + ncl - NR_END));
		free((FREE_ARG)(m + nrl - NR_END));
	}

	void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
		/* free an int matrix allocated by imatrix() */
	{
		free((FREE_ARG)(m[nrl] + ncl - NR_END));
		free((FREE_ARG)(m + nrl - NR_END));
	}

	void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch)
		/* free a submatrix allocated by submatrix() */
	{
		free((FREE_ARG)(b + nrl - NR_END));
	}

	void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch)
		/* free a matrix allocated by convert_matrix() */
	{
		free((FREE_ARG)(b + nrl - NR_END));
	}

	void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,
		long ndl, long ndh)
		/* free a float f3tensor allocated by f3tensor() */
	{
		free((FREE_ARG)(t[nrl][ncl] + ndl - NR_END));
		free((FREE_ARG)(t[nrl] + ncl - NR_END));
		free((FREE_ARG)(t + nrl - NR_END));
	}
	/* (C) Copr. 1986-92 Numerical Recipes Software 9z!+!1(t+%. */
#endif

};

#endif
