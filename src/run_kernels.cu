#include "kernels.cuh"
#include "run_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <sys/time.h>

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("[CUDA ERROR] from file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void CudaDeviceInfo() {
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

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  // we call this function 3 times in quick succession to initialize the
  // matrices
  struct timeval time{};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    // creating small floats from -4.04 to +4.04
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.0);
    mat[i] = tmp;
  }
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int selected_kernel, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C) {
  switch (selected_kernel) {
  case 0:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  }
}