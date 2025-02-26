#ifndef RUN_KERNELS_CUH
#define RUN_KERNELS_CUH
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void cudaCheck(cudaError_t err, const char *file, int line);
void CudaDeviceInfo();

void randomize_matrix(float *mat, int N);
void run_kernel(int selected_kernel, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C);

#endif