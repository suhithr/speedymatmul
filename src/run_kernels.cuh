#ifndef RUN_KERNELS_CUH
#define RUN_KERNELS_CUH
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void cudaCheck(cudaError_t err, const char *file, int line);
void CudaDeviceInfo();

void randomize_matrix(float *mat, int N);
void print_matrix(float *mat, int M, int N, std::ofstream &fs);
bool verify_matrix(float *D, float *D_ref, int SIZE);
void run_kernel(int selected_kernel, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);

#endif