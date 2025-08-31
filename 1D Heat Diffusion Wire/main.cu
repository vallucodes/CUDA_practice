#include <iostream>
#include <math.h>

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: build element matrices
__global__ void create_element(float *out, float length, float k, int Ne, int N) {
    int e = threadIdx.x;
    if (e >= N) return;

    float l = length / N;

    // Local 2x2 matrix
    out[e*Ne*Ne + 0*Ne + 0] = k / l;
    out[e*Ne*Ne + 0*Ne + 1] = -k / l;
    out[e*Ne*Ne + 1*Ne + 0] = -k / l;
    out[e*Ne*Ne + 1*Ne + 1] = k / l;
}

// Thomas algorithm solver for tridiagonal system
// c[1..n-1] subdiag, d[0..n-1] diag, e[0..n-2] superdiag, b rhs
void thomas(int n, float *c, float *d, float *e, float *b, float *x) {

	// forward sweep
	for (int i = 1; i < n; ++i)
	{
		float m = c[i] / d[i-1];
		d[i] -= m * e[i-1];
		b[i] -= m * b[i-1];
	}

	// back substitution
	x[n-1] = b[n-1] / d[n-1];

	for (int i = n-2; i >= 0; --i)
		x[i] = (b[i] - e[i]*x[i+1]) / d[i];
}

int main() {
	int N = 5;			// number of elements
	int Ne = 2;			// nodes per element
	int nodes = N + 1;
	float length = 1.0f;
	float k = 1.0f;
	float T_first = 60.0f, T_last = -54.0f;		//boundary conditions

	// --- Create element matrices on device
	float *out;
	cudaMallocManaged(&out, N * Ne * Ne * sizeof(float));

	create_element<<<1, N>>>(out, length, k, Ne, N);
	cudaDeviceSynchronize();

	float K[N + 1][N + 1] = {0}; // for N=5 → 6 nodes
	for (int e = 0; e < N; e++) {
		int n1 = e;
		int n2 = e + 1;
		float k11 = out[e*Ne*Ne + 0];
		float k12 = out[e*Ne*Ne + 1];
		float k21 = out[e*Ne*Ne + 2];
		float k22 = out[e*Ne*Ne + 3];

		K[n1][n1] += k11;
		K[n1][n2] += k12;
		K[n2][n1] += k21;
		K[n2][n2] += k22;
	}

	// Reduced system size = (nodes-2)
	int nRed = nodes - 2;
	float c[nRed] = {0},
			d[nRed] = {0},
			e[nRed] = {0},
			b[nRed] = {0},
			x[nRed] = {0};

	// Fill tridiagonal arrays from reduced system
	for (int i = 0; i < nRed; ++i)
	{
		d[i] = K[i + 1][i + 1];			// main diagonal
		if (i > 0)
			c[i] = K[i + 1][i];			// sub-diagonal
		if (i < nRed-1)
			e[i] = K[i + 1][i + 2];		// super-diagonal

		if (i == 0)
			b[i] = -K[1][0] * T_first;
		else if (i == nRed-1)
			b[i] = -K[nodes - 2][nodes - 1] * T_last;
		else
			b[i] = 0;
	}

	for (int i = 0; i < nRed; ++i)
	{
		printf("d[%i] = %f, ", i, d[i]);
	}
	printf("\n");
	for (int i = 0; i < nRed; ++i)
	{
		printf("c[%i] = %f, ", i, c[i]);
	}
	printf("\n");
	for (int i = 0; i < nRed; ++i)
	{
		printf("e[%i] = %f, ", i, e[i]);
	}
	printf("\n");

	thomas(nRed, c, d, e, b, x);

	// --- Collect results
	float T[N];
	T[0] = T_first;
	for (int i = 0; i < nRed; i++)
		T[i+1] = x[i];
	T[nodes-1] = T_last;

	// --- Print results
	printf("Nodal temperatures:\n");
	for (int i = 0; i < nodes; i++) {
		printf("T[%d] = %.4f\n", i+1, T[i]);
	}

	cudaFree(out);
	return 0;
}
