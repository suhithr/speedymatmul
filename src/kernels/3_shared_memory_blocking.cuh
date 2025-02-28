#ifndef SHARED_MEMORY_BLOCKING_CUH
#define SHARED_MEMORY_BLOCKING_CUH

#include <cstdio>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_shared_memory_blocking(int M, int N, int K, float alpha,
                                             const float *A, const float *B,
                                             float beta, float *C) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
  __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

  float tmp = 0.0;

  // slides in chunks of 32
  for (int offset = 0; offset < K; offset += 32) {
    // load shared memory
    sA[threadIdx.x * 32 + threadIdx.y] = A[x * K + offset + threadIdx.y];
    sB[threadIdx.x * 32 + threadIdx.y] = B[(offset + threadIdx.x) * N + y];
    __syncthreads();

    for (int i = 0; i < 32; i++) {
      tmp += sA[threadIdx.x * 32 + i] * sB[(i * 32) + threadIdx.y];
    }
    __syncthreads();
    // C[x, y]
  }
  C[x * N + y] += alpha * tmp + beta * C[x * N + y];
}
#endif