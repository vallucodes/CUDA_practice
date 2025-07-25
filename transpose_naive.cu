#include <iostream>
#include <math.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void copy(float *odata, float *idata, int N)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = N;


	if (x >= N || y >= N)
		return ;
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[x * width + (y + j)] = idata[(y + j) * width + x];
}

int main()
{
	int N = 2048;

	float *out, *in;

	cudaMallocManaged(&in, N*N*sizeof(float));
	cudaMallocManaged(&out, N*N*sizeof(float));

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			out[i * N + j] = 0;
			in[i * N + j] = sin(i) + cos(j) + (i*j*0.001f);
		}
	}

	cudaMemPrefetchAsync(in, N*N*sizeof(float), 0, 0);
	cudaMemPrefetchAsync(out, N*N*sizeof(float), 0, 0);

	int blocksX = (N + TILE_DIM - 1) / TILE_DIM;
	int blocksY = (N + TILE_DIM - 1) / TILE_DIM;
	// printf("blocksX: %i, blocksY %i\n", blocksX, blocksY);

	dim3 grid(blocksX, blocksY);
	dim3 threads(TILE_DIM, BLOCK_ROWS);

	copy<<<grid, threads>>>(out, in, N);

	cudaDeviceSynchronize();

	int errors = 0;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			float expected = sin(j) + cos(i) + (i*j*0.001f);
			float actual = out[i * N + j];
			float error = fabs(actual - expected);
			if (error > 1e-5) {
				errors++;
				if (errors < 10)
					std::cout << "Mismatch at (" << i << "," << j
							<< "): got " << actual << " expected " << expected << std::endl;
			}
		}

	std::cout << "Total mismatches: " << errors << std::endl;

	return 0;
}
