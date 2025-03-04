#ifndef NAIVE_CUH
#define NAIVE_CUH

#include <cstdio>
#include <cuda_runtime.h>

/*
Naive:
MxK * KxN = MxN
*/
// 128x128
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  //               0-3          32                 0-31
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y]; // Corrected multiplication
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

#endif