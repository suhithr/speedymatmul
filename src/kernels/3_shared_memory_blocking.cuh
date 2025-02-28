#ifndef SHARED_MEMORY_BLOCKING_CUH
#define SHARED_MEMORY_BLOCKING_CUH

#include <cstdio>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_shared_memory_blocking(int M, int N, int K, float alpha,
                                             const float *A, const float *B,
                                             float beta, float *C)
{

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // The row and column within the shared memory chunk
  const uint shmem_row = threadIdx.x;
  const uint shmem_col = threadIdx.y; // this is because the shmem size == block size

  __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
  __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

  float tmp = 0.0;

  // slides in chunks of 32
  for (int offset = 0; offset < K; offset += 32)
  {
    // load shared memory
    sA[threadIdx.x * BLOCKSIZE + threadIdx.y] = A[x * K + (offset + threadIdx.y)]; // the same row (x) of the result element (x, y)
                                                                          // with the column offset by (offset + tid.y) 
    sB[threadIdx.x * BLOCKSIZE + threadIdx.y] = B[(offset + threadIdx.x) * N + y]; // the same column of the result element (x, y)
                                                                          // with the row offset by (offset + tid.x) * N
    __syncthreads();

    for (int i = 0; i < BLOCKSIZE; i++)
    {
      tmp += sA[threadIdx.x * BLOCKSIZE + i] * sB[(i * BLOCKSIZE) + threadIdx.y];
    }
    __syncthreads();
    // C[x, y]
  }
  C[x * N + y] += alpha * tmp + beta * C[x * N + y];
}
#endif