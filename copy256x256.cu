#include <iostream>
#include <math.h>

__global__ void copy(float *odata, float *idata, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// printf("index: %i\n", index);

	if (index >= N * N)
		return ;
	odata[index] = idata[index];
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

	int threadsPerBlock = 256;
	int blocks = (N * N + 255) / 256;

	copy<<<blocks, threadsPerBlock>>>(out, in, N);

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
