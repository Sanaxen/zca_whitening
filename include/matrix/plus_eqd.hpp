#ifndef _PLUS_EQD_HPP
#define _PLUS_EQD_HPP


template <typename T>
inline void plus_eqd_( T* a, int am, int an, const T* b, int bm, int bn,  T* verify = NULL)
{
#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU || an > MIN_SIZE_APPLYING_GPGPU)
	{
		//printf("GPU plus_eqd_\n");
		plus_eq_gpu(a, am, an, b, bm, bn);
	}
	else
	{
		plus_eq_standerd(a, am, an, b, bm, bn);
	}
#else
	plus_eq_standerd(a, am, an, b, bm, bn);
#endif

	//verify
	if (verify)
	{
		T eps = 0.0;
		for (int i = 0; i < am*an; ++i)
		{
			eps += fabs(a[i] - verify[i]);
		}
		printf("eps=%f\n", eps);
	}
}

template <typename T>
inline void plus_eq_standerd( T* a, int am, int an, const T* b, int bm, int bn)
{
	const int mn = am*an;
#if 0
#pragma omp parallel for
	for (int i = 0; i < mn; ++i) a[i] += b[i];
#else
	int i = 0;
#pragma omp parallel for private(i)
	for (i = 0; i < mn - 4; i += 4)
	{
		a[i + 0] += b[i + 0];
		a[i + 1] += b[i + 1];
		a[i + 2] += b[i + 2];
		a[i + 3] += b[i + 3];
	}
	for (i; i < mn; ++i)  a[i] += b[i];

#endif

}

#if USE_GPU
template <typename T>
inline void plus_eq_gpu( T* a, int am, int an, const T* b, int bm, int bn)
{
	std::vector<float> va;
	std::vector<float> vb;
#pragma omp parallel
#pragma omp sections nowait
	{
#pragma omp section
		{
			copy_array(a, am*an, va);
		}
#pragma omp section
		{
			copy_array(b, bm*bn, vb);
		}
	}

	concurrency::extent<2> e_a(am, an);

	// Copy in
	array_view<float, 2> av_a(e_a, va);
	array_view<const float, 2> av_b(e_a, vb);

	parallel_for_each(av_a.extent, [=](index<2> idx) restrict(amp, cpu)
	{
		av_a[idx] += av_b[idx];
	});
	// explicitly about copying out data
	av_a.synchronize();


	const int mn = am*an;
#pragma omp parallel for
	for (int i = 0; i < mn; ++i)
	{
		a[i] = va[i];
	}
}
#endif

#endif
