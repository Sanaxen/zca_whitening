#ifndef _RECIPROCAL_HPP
#define _RECIPROCAL_HPP


template <typename T>
inline void reciprocal_( const T* a, int am, int an, T* ret, T* verify = NULL)
{
#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU && an > MIN_SIZE_APPLYING_GPGPU)
	{
		reciprocal_gpu(a, am, an, ret);
	}
	else
	{
		reciprocal_standerd(a, am, an, ret);
	}
#else
	reciprocal_standerd(a, am, an, ret);
#endif

	//verify
	if (verify)
	{
		T eps = 0.0;
		for (int i = 0; i < am*an; ++i)
		{
			eps += fabs(ret[i] - verify[i]);
		}
		printf("eps=%f\n", eps);
	}
}

template <typename T>
inline void reciprocal_standerd( const T* a, int am, int an, T *ret)
{
	const int mn = am*an;
#pragma omp parallel for
	for (int i = 0; i < mn; ++i) ret[i] = (fabs(a[i]) > 1.0e-16) ? 1.0/a[i] : 1.0;

}

#if USE_GPU
template <typename T>
inline void reciprocal_gpu( const T* a, int am, int an, T* ret)
{
	std::vector<float> va;
	std::vector<float> vresult(am*an);
	{
		copy_array(a, am*an, va);
	}

	concurrency::extent<2> e_a(am, an);

	// Copy in
	array_view<const float, 2> av_a(e_a, va);
	array_view<float, 2> av_c(e_a, vresult);
	av_c.discard_data();

	parallel_for_each(av_a.extent, [=](index<2> idx) restrict(amp, cpu)
	{
		av_c[idx] = (concurrency::fast_math::fabs(av_a[idx]) > 1.0e-10 ) ? 1.0/av_a[idx] : 1.0;
	});
	// explicitly about copying out data
	av_c.synchronize();


	const int mn = am*an;
#pragma omp parallel for
	for (int i = 0; i < mn; ++i)
	{
		ret[i] = vresult[i];
	}
}
#endif


#endif
