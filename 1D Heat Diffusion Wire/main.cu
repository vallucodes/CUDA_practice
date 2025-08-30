#include <iostream>
#include <math.h>

// __global__ void create_element(float *out, float length, float k, int Ne, int N) {
// 	int e = threadIdx.x; // element index
// 	if (e >= N) return; // bounds check

// 	float Le = length / N; // element length

// 	// 1D linear element stiffness matrix
// 	// Using proper 2D matrix indexing: e*Ne*Ne + row*Ne + col
// 	out[e*Ne*Ne + 0*Ne + 0] = k / Le;  // row 0 col 0
// 	out[e*Ne*Ne + 0*Ne + 1] = -k / Le; // row 0 col 1
// 	out[e*Ne*Ne + 1*Ne + 0] = -k / Le; // row 1 col 0
// 	out[e*Ne*Ne + 1*Ne + 1] = k / Le;  // row 1 col 1
// }

// int main() {
// 	int N = 5;      // number of elements
// 	int Ne = 2;     // nodes per element
// 	float length = 1.0f;
// 	float k = 1.0f;

// 	float *out;
// 	cudaMallocManaged(&out, N * Ne * Ne * sizeof(float));

// 	create_element<<<1, N>>>(out, length, k, Ne, N);
// 	cudaError_t err = cudaGetLastError();
// 	if (err != cudaSuccess) {
//     	printf("Kernel launch error: %s\n", cudaGetErrorString(err));
// 	}
// 	cudaDeviceSynchronize();

// 	// print results
// 	for (int e = 0; e < N; e++) {
// 		printf("Element %d:\n", e);
// 		for (int i = 0; i < Ne; i++) {
// 			for (int j = 0; j < Ne; j++) {
// 				printf("%8.4f ", out[e * Ne * Ne + i * Ne + j]);
// 			}
// 			printf("\n");
// 		}
// 		printf("\n");
// 	}

// 	cudaFree(out);
// 	return 0;
// }

__global__ void create_element(float *out, float length, float k, int Ne) {
    int e = threadIdx.x; // element index
    if (e >= 2) return;  // just for safety

    // 1D linear element stiffness matrix
    out[e*Ne*Ne + 0] = k/length;      // row 0 col 0
    out[e*Ne*Ne + 1] = -k/length;     // row 0 col 1
    out[e*Ne*Ne + 2] = -k/length;     // row 1 col 0
    out[e*Ne*Ne + 3] = k/length;      // row 1 col 1
}

int main() {
    int N = 2;
    int Ne = 2; // nodes per element
    float length = 1;
    float k = 1;

    float *out;
    cudaMallocManaged(&out, N*Ne*Ne*sizeof(float));

    create_element<<<1, N>>>(out, length, k, Ne);
    cudaDeviceSynchronize();

    // print
    for(int e=0; e<N; e++){
        printf("Element %d:\n", e);
        for(int i=0; i<Ne; i++){
            for(int j=0; j<Ne; j++){
                printf("%f ", out[e*Ne*Ne + i*Ne + j]);
            }
            printf("\n");
        }
    }

    cudaFree(out);
    return 0;
}

