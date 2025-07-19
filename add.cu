#include <iostream>
#include <iomanip>
#include <math.h>

__global__
void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

int main(void)
{
	float *x, *y;
	int N = 1000000;

	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Prefetch the x and y arrays to the GPU
	cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
	cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

	// Run kernel on 1M elements on the CPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	add<<<numBlocks, blockSize>>>(N, x, y);

	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	std::cout << std::fixed << std::setprecision(6);
	std::cout << "Max error: " << maxError << std::endl;

	// Free memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}
