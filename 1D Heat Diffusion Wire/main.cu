#include <iostream>
#include <math.h>

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: build element matrices
__global__ void create_element(float *element_matrices, float length, float k, int Ne, int N) {
	int e = threadIdx.x;
	if (e >= N) return;

	float l = length / N;

	// Local 2x2 matrix
	element_matrices[e*Ne*Ne + 0*Ne + 0] = k / l;
	element_matrices[e*Ne*Ne + 0*Ne + 1] = -k / l;
	element_matrices[e*Ne*Ne + 1*Ne + 0] = -k / l;
	element_matrices[e*Ne*Ne + 1*Ne + 1] = k / l;
}

// Thomas algorithm solver for tridiagonal system
// c[1..n-1] subdiag, d[0..n-1] diag, e[0..n-2] superdiag, b rhs
void thomas(int n, float *c, float *d, float *e, float *b, float *x) {

	// forward sweep
	for (int i = 1; i < n; ++i)
	{
		float m = c[i] / d[i - 1];
		d[i] -= m * e[i - 1];
		b[i] -= m * b[i - 1];
	}

	// back substitution
	x[n - 1] = b[n - 1] / d[n - 1];

	for (int i = n - 2; i >= 0; --i)
		x[i] = (b[i] - e[i] * x[i +1 ]) / d[i];
}

int main() {
	int N = 11;			// number of elements
	int Ne = 2;			// nodes per element
	int nodes = N + 1;
	float length = 1.0f;
	float k = 1.0f;
	float T_first = 100.0f, T_last = 60.0f;		//boundary conditions

	// --- Create element matrices on device
	float *element_matrices;
	cudaMallocManaged(&element_matrices, N * Ne * Ne * sizeof(float));

	create_element<<<1, N>>>(element_matrices, length, k, Ne, N);
	cudaDeviceSynchronize();

	// Reduced system size = (nodes-2)
	int nRed = nodes - 2;
	float c[nRed] = {0},
			d[nRed] = {0},
			e[nRed] = {0},
			b[nRed] = {0},
			x[nRed] = {0};


	// Fill diagonals for Thomas algorithm //this task could be given to kernel
	for (int i = 0; i < nRed; ++i)
	{
		int node = i + 1;

		int eL = node - 1;
		int eR = node;

		int baseL = eL * Ne * Ne;
		int baseR = eR * Ne * Ne;

		d[i] = element_matrices[baseL + 3] + element_matrices[baseR];
		if (i > 0)
			c[i] = element_matrices[baseL + 2];
		if (i < nRed - 1)
			e[i] = element_matrices[baseL + 1];
		c[0] = 0;
		e[nRed - 1] = 0;

		if (i == 0)
			b[i] = -element_matrices[baseL + 2] * T_first;
		else if (i == nRed - 1)
			b[i] = -element_matrices[baseR + 1] * T_last;
		else
			b[i] = 0;
	}

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

	cudaFree(element_matrices);
	return 0;
}

	// printf("nRed = %i\n", nRed);

	// for (int i = 0; i < nRed; ++i)
	// 	printf("d[%i] = %f, ", i, d[i]);
	// printf("\n");
	// for (int i = 0; i < nRed; ++i)
	// 	printf("c[%i] = %f, ", i, c[i]);
	// printf("\n");
	// for (int i = 0; i < nRed; ++i)
	// 	printf("e[%i] = %f, ", i, e[i]);
	// printf("\n");
	// for (int i = 0; i < nRed; ++i)
	// 	printf("b[%i] = %f, ", i, b[i]);
	// printf("\n\n");
