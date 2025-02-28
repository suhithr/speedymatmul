#ifndef GLOBAL_MEMORY_COALESCING_CUH
#define GLOBAL_MEMORY_COALESCING_CUH

#include <cstdio>
#include <cuda_runtime.h>

/*
Naive Algorithm with an access pattern that supports global memory coalescing:
MxK * KxN = MxN
*/
__global__ void sgemm_global_memory_coalescing(int M, int N, int K, float alpha,
                                               const float *A, const float *B,
                                               float beta, float *C) {
  const uint BLOCKSIZE = 32;
  const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      // tmp += A[y * K + i] * B[i * N + x]; // Corrected multiplication
      tmp += A[x * K + i] * B[i * N + y]; // Corrected multiplication
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

#endif