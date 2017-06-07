#ifndef _CLAPCK_UTIL_HPP

#define _CLAPCK_UTIL_HPP

#pragma comment(lib, "../third_party/clapack/lib/cLAPACK.lib")
#pragma comment(lib, "../third_party/clapack/lib/BLAS.lib")
#pragma comment(lib, "../third_party/clapack/lib/libI77.lib")
#pragma comment(lib, "../third_party/clapack/lib/libF77.lib")

extern "C"
{
	//FILE _iob[] = { *stdin, *stdout, *stderr };
	//extern "C" FILE * __cdecl __iob_func(void)
	//{
	//	return _iob;
	//}
	
	extern "C" int dgesvd_(char*jobu, char*jobvt, long int*m, long int*n, double*A, long int*lda, double*s, double*u, long int*ldu, double*vt, long int*ldvt, double*work, long int*lwork_tmp, long int*info);
	extern "C" void dgemm_(char* transa, char* transb, long int* m, long int* n, long int* k, double* alpha, const double* A, long int* lda, const double* B, long int* ldb, double* beta, double* C, long int* ldc);
};

#endif