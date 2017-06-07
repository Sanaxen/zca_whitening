#ifndef _HADAMARD_HPP
#define _HADAMARD_HPP


template <typename T>
inline void hadamard_(const T* a, int am, int an, const T* b, int bm, int bn, T* ret, T* verify = NULL)
{
#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU || an > MIN_SIZE_APPLYING_GPGPU)
	{
		//printf("GPU hadamard_\n");
		hadamard_gpu(a, am, an, b, bm, bn, ret);
	}
	else
	{
		hadamard_standerd(a, am, an, b, bm, bn, ret);
	}
#else
	hadamard_standerd(a, am, an, b, bm, bn, ret);
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
inline void hadamard_standerd(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	const int mn = am*an;
#pragma omp parallel for
	for (int i = 0; i < mn; ++i) ret[i] = a[i] * b[i];

}

#if USE_GPU
template <typename T>
inline void hadamard_gpu(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	std::vector<float> va;
	std::vector<float> vb;
	std::vector<float> vresult(am*an);
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
	array_view<const float, 2> av_a(e_a, va);
	array_view<const float, 2> av_b(e_a, vb);
	array_view<float, 2> av_c(e_a, vresult);

	av_c.discard_data();
	parallel_for_each(av_a.extent, [=](index<2> idx) restrict(amp, cpu)
	{
		av_c[idx] = av_a[idx] * av_b[idx];
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
