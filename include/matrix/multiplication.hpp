#ifndef _MULTIPLICATION_HPP
#define _MULTIPLICATION_HPP


template <typename T>
inline void mull_(const T* a, int am, int an, const T* b, int bm, int bn, T* ret, T* verify = NULL)
{
#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU || an > MIN_SIZE_APPLYING_GPGPU || bm > MIN_SIZE_APPLYING_GPGPU || bn > MIN_SIZE_APPLYING_GPGPU)
	{
		//printf("GPU mull_\n");
		int stat = -1;
		stat = mull_gpu_tiled<double, 16>(a, am, an, b, bm, bn, ret);
		if (stat == -1)
		{
			stat = mull_gpu_tiled<double, 8>(a, am, an, b, bm, bn, ret);
		}
		//if (stat == -1)
		//{
		//	stat = mull_gpu_tiled<double, 4>(a, am, an, b, bm, bn, ret);
		//}
		//if (stat == -1)
		//{
		//	stat = mull_gpu_tiled<double, 2>(a, am, an, b, bm, bn, ret);
		//}
		if (stat == -1)
		{
			mull_gpu(a, am, an, b, bm, bn, ret);
		}
	}
	else
	{
		mull_Unrolling(a, am, an, b, bm, bn, ret);
	}
#else
	//mull_standerd(a, am, an, b, bm, bn, ret);
	mull_Unrolling(a, am, an, b, bm, bn, ret);
#endif
	//verify
	if ( verify )
	{
		T eps = 0.0;
		const int m = am, n = bn, l = an;
		for ( int i = 0; i < m*n; i++ )
		{
			eps += fabs(ret[i] - verify[i]);
		}
		printf("eps=%f\n", eps);
	}

}

template <typename T>
inline void mull_standerd(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	const int m = am, n = bn, l = an;

#pragma omp parallel for
		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j) {
				T sum = 0.0;
				for (int k = 0; k < l; ++k)
					sum += a[i*an + k]*b[k*bn + j];
				ret[n*i + j] = sum;
			}
}

template <typename T>
inline void mull_Unrolling(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	const int m = am, n = bn, l = an;
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < m*n; ++i)
		{
			ret[i] = 0.0;
		}

#pragma omp for
		for (int i = 0; i < m; ++i)
		{
			for (int k = 0; k < l; ++k)
			{
				const T mm = a[i*an + k];

				//ƒAƒ“ƒ[ƒŠƒ“ƒO
				int j = 0;
				for (j = 0; j < n - 4; j += 4)
				{
					ret[n*i + j] += mm*b[k*bn + j];
					ret[n*i + j + 1] += mm*b[k*bn + j + 1];
					ret[n*i + j + 2] += mm*b[k*bn + j + 2];
					ret[n*i + j + 3] += mm*b[k*bn + j + 3];
				}
#if 0
				for (j = 0; j < n - 8; j += 8)
				{
					ret[n*i + j] += mm*b[k*bn + j];
					ret[n*i + j + 1] += mm*b[k*bn + j + 1];
					ret[n*i + j + 2] += mm*b[k*bn + j + 2];
					ret[n*i + j + 3] += mm*b[k*bn + j + 3];
					ret[n*i + j + 4] += mm*b[k*bn + j + 4];
					ret[n*i + j + 5] += mm*b[k*bn + j + 5];
					ret[n*i + j + 6] += mm*b[k*bn + j + 6];
					ret[n*i + j + 7] += mm*b[k*bn + j + 7];
				}
#endif

#if 0
				for (j = 0; j < n - 16; j += 16)
				{
					ret[n*i + j] += mm*b[k*bn + j];
					ret[n*i + j + 1] += mm*b[k*bn + j + 1];
					ret[n*i + j + 2] += mm*b[k*bn + j + 2];
					ret[n*i + j + 3] += mm*b[k*bn + j + 3];
					ret[n*i + j + 4] += mm*b[k*bn + j + 4];
					ret[n*i + j + 5] += mm*b[k*bn + j + 5];
					ret[n*i + j + 6] += mm*b[k*bn + j + 6];
					ret[n*i + j + 7] += mm*b[k*bn + j + 7];
					ret[n*i + j + 8] += mm*b[k*bn + j + 8];
					ret[n*i + j + 9] += mm*b[k*bn + j + 9];
					ret[n*i + j + 10] += mm*b[k*bn + j + 10];
					ret[n*i + j + 11] += mm*b[k*bn + j + 11];
					ret[n*i + j + 12] += mm*b[k*bn + j + 12];
					ret[n*i + j + 13] += mm*b[k*bn + j + 13];
					ret[n*i + j + 14] += mm*b[k*bn + j + 14];
					ret[n*i + j + 15] += mm*b[k*bn + j + 15];
				}
#endif
				for ( j; j < n; j += 1)
				{
					ret[n*i + j] += mm*b[k*bn + j];
				}
			}
		}
	}
}

#if USE_GPU

template <typename T>
inline void mull_gpu(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	//printf("GPU ");
	const int m = am, n = bn, l = an;
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

	concurrency::extent<2> e_a(am, an), e_b(bm, bn), e_c(am, bn);

	// Copy in
	array_view<const float, 2> av_a(e_a, va); 
	array_view<const float, 2> av_b(e_b, vb); 
	array_view<float, 2> av_c(e_c, vresult);
	av_c.discard_data();

	// Compute - outer 2 for loops of CPU is replaced by a parallel_for_each
	concurrency::parallel_for_each(av_c.extent, [=](index<2> idx) restrict(amp,cpu)
		{
			float result = 0;

			for(int i = 0; i < av_a.extent[1]; ++i)
			{
				index<2> idx_a(idx[0], i);
				index<2> idx_b(i, idx[1]);

				result += av_a[idx_a] * av_b[idx_b];
			}

			av_c[idx] = result;
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

template <typename T, int tile_size>
inline int mull_gpu_tiled(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	if (!(am%tile_size == 0 && bn%tile_size == 0 && an%tile_size == 0))
	{
		//printf("mull_gpu_tiled size error.\n");
		return -1;
	}
	//printf("mull_gpu_tiled size OK.\n");
	
	const int m = am, n = bn, l = an;
	std::vector<float> va;
	std::vector<float> vb;
	std::vector<float> vresult;
	{
		copy_array(a, am*an, va);
		copy_array(b, bm*bn, vb);
		vresult.resize(am*bn);
	}

	concurrency::extent<2> e_a(am, an), e_b(bm, bn), e_c(am, bn);

	array_view<const float, 2> av_a(e_a, va);
	array_view<const float, 2> av_b(e_b, vb);
	array_view<float, 2> av_c(e_c, vresult);

	concurrency::extent<2> compute_domain(e_c);

	parallel_for_each(compute_domain.tile<tile_size, tile_size>(), [=](tiled_index<tile_size, tile_size> tidx) restrict(amp)
	{
		float temp_c = 0;

		index<2> localIdx = tidx.local;
		index<2> globalIdx = tidx.global;

		for (int i = 0; i < an; i += tile_size)
		{
			tile_static float localB[tile_size][tile_size];
			tile_static float localA[tile_size][tile_size];

			localA[localIdx[0]][localIdx[1]] = av_a(globalIdx[0], i + localIdx[1]);
			localB[localIdx[0]][localIdx[1]] = av_b(i + localIdx[0], globalIdx[1]);

			tidx.barrier.wait();

			for (unsigned k = 0; k < tile_size; k++)
			{
				temp_c += localA[localIdx[0]][k] * localB[k][localIdx[1]];
			}

			tidx.barrier.wait();
		}

		av_c[tidx] = temp_c;
	});
	// copying out data is implicit - when array_view goes out of scope data is synchronized	//av_c.synchronize();
	av_c.synchronize();

	const int mn = am*bn;
	//copy_array(vresult, mn, ret);

#pragma omp parallel for
	for (int i = 0; i < mn; i++)
	{
		ret[i] = vresult[i];
	}
	return 0;
}

#endif


#endif
