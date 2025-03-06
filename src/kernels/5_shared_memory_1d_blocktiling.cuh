#ifndef SHARED_MEMORY_1D_BLOCKTILING_CUH
#define SHARED_MEMORY_1D_BLOCKTILING_CUH

#include <cstdio>
#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint BK>
__global__ void sgemm_shared_memory_1d_blocktiling(int M, int N, int K, float alpha,
                                                   const float *A, const float *B,
                                                   float beta, float *C)
{
  // const uint threadId = threadIdx.x + blockDim.x * (threadIdx.y + (blockDim.y * threadIdx.z));

  //                                                64 / 2 * [0..1]
  const uint b_inner_col = threadIdx.x % BN; // 0..63
  const uint b_inner_row = threadIdx.x / BN; // 0..8
  const uint a_inner_col = threadIdx.x % BK;
  const uint a_inner_row = threadIdx.x / BK;
  // const uint c_row = blockIdx.y;
  // const uint c_col = blockIdx.x;

  __shared__ float sA[BM * BK];
  __shared__ float sB[BK * BN];

  // // we store multiple values in tmp since each thread calculates many items
  float tmp[8] = {0.0};

  // // slides in chunks of 8
  A += blockIdx.y * BN * K;
  B += blockIdx.x * BM;
  C += (blockIdx.y * BM * N) + (blockIdx.x * BN);
  for (int offset = 0; offset < K; offset += BK)
  {
    // 64, 8 threads loading
    sA[a_inner_row * BK + a_inner_col] = A[a_inner_row * K + a_inner_col]; // the same row (x) of the result element (x, y)
                                                                                       // with the column offset by (offset + tid.y)
    sB[b_inner_row * BN + b_inner_col] = B[(b_inner_row)*N + b_inner_col];         // the same column of the result element (x, y)
                                                                                       // with the row offset by (offset + tid.x) * N
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int resIdx = 0; resIdx < 8; resIdx++)
    {
      for (int dotIdx = 0; dotIdx < BK; dotIdx++)
      {
        // we need to add 0, 8, 16, ... to the a_in_block_r
        tmp[resIdx] += 
          sA[((b_inner_row * 8) + resIdx) * BK + dotIdx] * sB[dotIdx * BN + b_inner_col];
      }
    }
    // for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    //   // we make the dotproduct loop the outside loop, which facilitates
    //   // reuse of the Bs entry, which we can cache in a tmp var.
    //   float tmpB = sB[dotIdx * BN + b_inner_col];
    //   for (uint resIdx = 0; resIdx < 8; ++resIdx) {
    //     tmp[resIdx] +=
    //         sA[(b_inner_row * 8 + resIdx) * BK + dotIdx] * tmpB;
    //   }
    __syncthreads();
  }

  for (int resIdx = 0; resIdx < 8; resIdx++)
  {
    C[(b_inner_row * 8 + resIdx) * N + b_inner_col] = alpha * tmp[resIdx] + beta * C[(b_inner_row * 8 + resIdx) * N + b_inner_col];
  }
}
#endif
// printf("threadId %d, tidx, tidy: (%d, %d), C (%d, %d)\n",
//   threadId,
//   threadIdx.x, threadIdx.y,
//   cRow, cCol
// );