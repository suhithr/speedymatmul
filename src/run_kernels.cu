#include "kernels.cuh"
#include "run_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

void cudaCheck(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
  {
    printf("[CUDA ERROR] from file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void CudaDeviceInfo()
{
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
      Name: %s\n\
      Compute Capability: %d.%d\n\
      memoryBusWidth: %d\n\
      maxThreadsPerBlock: %d\n\
      maxThreadsPerMultiProcessor: %d\n\
      maxRegsPerBlock: %d\n\
      maxRegsPerMultiProcessor: %d\n\
      totalGlobalMem: %zuMB\n\
      sharedMemPerBlock: %zuKB\n\
      sharedMemPerMultiprocessor: %zuKB\n\
      totalConstMem: %zuKB\n\
      multiProcessorCount: %d\n\
      Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void print_matrix(float *mat, int M, int N, std::ofstream &fs)
{
  int i;
  fs << std::setprecision(2) << std::fixed;
  fs << "[";
  for (i = 0; i < M * N; i++)
  {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << mat[i];
    else
      fs << std::setw(5) << mat[i] << ", ";
    if ((i + 1) % N == 0)
    {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

void randomize_matrix(float *mat, int N)
{
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  // we call this function 3 times in quick succession to initialize the
  // matrices
  struct timeval time{};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++)
  {
    // creating small floats from -4.04 to +4.04
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.0);
    mat[i] = tmp;
  }
}

bool verify_matrix(float *D_ref, float *D, int SIZE)
{
  double diff = 0.0;
  for (int i = 0; i < SIZE; i++)
  {
    diff = std::fabs(D[i] - D_ref[i]);
    if (diff > 0.01)
    {
      printf("Value %6.2f is %6.2f :: diff %6.2f at %d \n", D_ref[i], D[i],
             diff, i);
      fflush(stdout);
      return false;
    }
  }
  return true;
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C)
{
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_global_memory_clsc(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C)
{
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_global_memory_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, A, B,
                                                        beta, C);
}

void run_sgemm_shared_memory_blocking(int M, int N, int K, float alpha,
                                      float *A, float *B, float beta,
                                      float *C)
{
  const uint BLOCKSIZE = 32;

  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

  sgemm_shared_memory_blocking<BLOCKSIZE>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_cublas_fp32(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C, cublasHandle_t handle)
{
  // cuBLAS uses column-major order. So we change the order of our row-major A &
  // B, since (B^T*A^T)^T = (A*B)
  cublasStatus_t gemm_stat;
  gemm_stat =
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                   CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void run_kernel(int selected_kernel, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle)
{
  switch (selected_kernel)
  {
  case 0:
    run_cublas_fp32(M, N, K, alpha, A, B, beta, C, handle);
    break;
  case 1:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_global_memory_clsc(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_global_memory_clsc(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}
