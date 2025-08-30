#include <iostream>
#include <math.h>

__global__ void copy(float *odata, float *idata, int N) {
  int index = threadIdx.x;
  // int stride = 1;

  for (int i = index; i < N; i++)
    for (int j = 0; j < N; j++)
      odata[i * N + j] = idata[i * N + j];
}

int main()
{

	int N = 2048;
	float *in, *out;

	cudaMallocManaged(&in, N*N*sizeof(float));
	cudaMallocManaged(&out, N*N*sizeof(float));

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			in[i * N + j] = j * 2.0;

	cudaMemPrefetchAsync(in, N*N*sizeof(float), 0, 0);

	copy<<<1, 1>>>(out, in, N);

	cudaDeviceSynchronize();

	float maxError = 0.0f;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			maxError = fmax(maxError, fabs(out[i * N + j] - j * 2.0));

	std::cout << "max error: " << maxError << std::endl;

	cudaFree(in);
	cudaFree(out);
	return 0;
}
