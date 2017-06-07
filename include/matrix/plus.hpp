#ifndef _PLUS_HPP
#define _PLUS_HPP


template <typename T>
inline void plus_(const T* a, int am, int an, const T* b, int bm, int bn, T* ret, T* verify = NULL)
{
#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU || an > MIN_SIZE_APPLYING_GPGPU || bm > MIN_SIZE_APPLYING_GPGPU || bn > MIN_SIZE_APPLYING_GPGPU)
	{
		printf("GPU plus_\n");
		plus_gpu(a, am, an, b, bm, bn, ret);
	}
	else
	{
		plus_standerd(a, am, an, b, bm, bn, ret);
	}
#else
	plus_standerd(a, am, an, b, bm, bn, ret);
#endif
	//verify
	if ( verify )
	{
		T eps = 0.0;
		const int m = am, n = bn, l = an;
		for ( int i = 0; i < m*n; ++i )
		{
			eps += fabs(ret[i] - verify[i]);
		}
		printf("eps=%f\n", eps);
	}

}

template <typename T>
inline void plus_standerd(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	const int m = am, n = an;

	const int mn = m*n;
#pragma omp parallel for
		for (int i = 0; i < mn; ++i)
		{
			ret[i] = a[i] + b[i];;
		}
}


#if USE_GPU

template <typename T>
inline void plus_gpu(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	//printf("GPU ");
	const int m = am, n = an;
	std::vector<float> va;
	std::vector<float> vb;
	std::vector<float> vresult(am*bn);

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

	concurrency::extent<2> e_a(am, an), e_b(bm, bn), e_c(am, an);

	// Copy in
	array_view<const float, 2> av_a(e_a, va); 
	array_view<const float, 2> av_b(e_b, vb); 
	array_view<float, 2> av_c(e_c, vresult);
	av_c.discard_data();

	// Compute - outer 2 for loops of CPU is replaced by a parallel_for_each
	concurrency::parallel_for_each(av_c.extent, [=](index<2> idx) restrict(amp,cpu)
		{
			av_c[idx] = av_a[idx] * av_b[idx];;
		});
	// explicitly about copying out data
	av_c.synchronize();


	const int mn = am*bn;
#pragma omp parallel for
	for ( int i = 0; i < mn; ++i )
	{
		ret[i] = vresult[i];
	}
}
#endif


#endif
