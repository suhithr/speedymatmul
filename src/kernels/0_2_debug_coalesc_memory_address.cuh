#ifndef DEBUG_GLOBAL_MEMORY_COALESCING_CUH
#define DEBUG_GLOBAL_MEMORY_COALESCING_CUH

#include <cstdio>
#include <cuda_runtime.h>

/*
Prints out the memory addresses that could be accessed in the
global memory coalescing kernel.

MxK * KxN = MxN
*/
template <const uint BLOCKSIZE>
__global__ void debug_sgemm_global_memory_coalescing(int M, int N, int K, float alpha,
                                                     const float *A, const float *B,
                                                     float beta, float *C)
{
    // threadId used to calculate which threads are next to each other
    // and may thus form a warp.
    const uint threadId = threadIdx.x + (blockDim.x * threadIdx.y);

    // method I attempted to use as an access pattern for coalescing loads and writes
    const uint x = blockIdx.x * BLOCKSIZE + threadIdx.y;
    const uint y = blockIdx.y * BLOCKSIZE + threadIdx.x;

    // the worklog's method
    // const uint BLOCKSIZE = 32;
    // const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    // const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    printf("threadId: %d, tIdx.x, tIdx.y: (%d, %d), C (%d, %d), A [%d, 0..31], B[0..31, %d] \n",
           threadId,
           threadIdx.x, threadIdx.y,
           x, y,
           x,
           y);

    //   if (x < M && y < N)
    //   {
    //     float tmp = 0.0;
    //     for (int i = 0; i < K; ++i)
    //     {
    //       tmp += A[x * K + i] * B[i * N + y];
    //     }
    //     C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    //   }
}

#endif