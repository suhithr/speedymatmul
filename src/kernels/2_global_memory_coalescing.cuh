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
                                               float beta, float *C)
{
  const uint x = blockIdx.x * 32 + (threadIdx.x / 32);//threadIdx.y;// (threadIdx.x / 32);
  const uint y = blockIdx.y * 32 + (threadIdx.x % 32);// threadIdx.x;// (threadIdx.x % 32);

  if (x < M && y < N)
  {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += A[x * K + i] * B[i * N + y]; // Corrected multiplication
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

#endif