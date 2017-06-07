#ifndef _PCA_NormalizedData_HPP
#define _PCA_NormalizedData_HPP

//PCA on Normalized Data
template<class T>
inline T* pca_normalizedData(T* data, const int dataNum, double eps = 0.0)
{
	double av = 0.0;
	const int sz = dataNum;
	T* normalized = new T[dataNum];

#pragma omp parallel for reduction(+:av)
	for (int k = 0; k < sz; k++)
	{
		av += data[k];
	}
	av /= (double)( sz);

	double sd = 0.0;
#pragma omp parallel for reduction(+:sd)
	for (int k = 0; k < sz; k++)
	{
		sd += pow(data[k] - av, 2.0);
	}
	sd = sqrt(sd / (double)(sz)) + eps;

#pragma omp parallel for
	for (int k = 0; k < sz; k++)
	{
		normalized[k] = (data[k] - av) / sd;
	}

	return normalized;
}

#endif
