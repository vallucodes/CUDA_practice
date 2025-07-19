#include <iostream>
#include <math.h>

#define TILE_DIM 4
#define BLOCK_ROWS 2

__global__ void transpose(float *odata, float *idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j) * width + x] = odata[(y + j) * width + x];
}

int main()
{
	int N = 16;

	float *out, *in;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)

	transpose<<<>>>(out, in);

	float maxError = 0.0;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			maxError = max(maxError, fabs());

	std::cout << "max error: " << maxError << std::endl;

	return 0;
}
