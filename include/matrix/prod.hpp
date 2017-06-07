#ifndef _PROD_HPP
#define _PROD_HPP


template <typename T>
inline void prod_( const T* a, int am, int an, const T c,  T* ret, T* verify = NULL)
{
#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU || an > MIN_SIZE_APPLYING_GPGPU)
	{
		//printf("GPU prod_\n");
		prod_gpu(a, am, an, c, ret);
	}
	else
	{
		prod_standerd(a, am, an, c, ret);
	}
#else
	prod_standerd(a, am, an, c, ret);
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
inline void prod_standerd( const T* a, int am, int an, const T c, T* ret)
{
	const int mn = am*an;
#pragma omp parallel for
	for (int i = 0; i < mn; ++i) ret[i] = c*a[i];

}

#if USE_GPU
template <typename T>
inline void prod_gpu( const T* a, int am, int an, const T c, T* ret)
{
	const float cc = static_cast<const float>(c);
	std::vector<float> va;
	std::vector<float> vresult(am*an);
	{
		copy_array(a, am*an, va);
	}

	concurrency::extent<2> e_a(am, an);

	// Copy in
	array_view<float, 2> av_a(e_a, va);
	array_view<float, 2> av_c(e_a, vresult);

	parallel_for_each(av_a.extent, [=](index<2> idx) restrict(amp, cpu)
	{
		av_c[idx] = av_a[idx] * cc;
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
