#ifndef SHARED_MEMORY_BLOCKING_CUH
#define SHARED_MEMORY_BLOCKING_CUH

#include <cstdio>
#include <cuda_runtime.h>

template <const uint BLOCK_SIDE>
__global__ void sgemm_shared_memory_blocking(int M, int N, int K, float alpha,
                                             const float *A, const float *B,
                                             float beta, float *C)
{

  const uint cCol = blockIdx.x * blockDim.x + threadIdx.x;
  const uint cRow = blockIdx.y * blockDim.y + threadIdx.y;

  // The row and column within the shared memory chunk
  const uint shmem_col = threadIdx.x;
  const uint shmem_row = threadIdx.y; // this is because the shmem size == block size

  __shared__ float sA[BLOCK_SIDE * BLOCK_SIDE];
  __shared__ float sB[BLOCK_SIDE * BLOCK_SIDE];

  float tmp = 0.0;

  // slides in chunks of 32
  for (int offset = 0; offset < K; offset += 32)
  {
    // load shared memory
    sA[shmem_row * BLOCK_SIDE + shmem_col] = A[cRow * K + (offset + shmem_col)]; // the same row (x) of the result element (x, y)
                                                                                 // with the column offset by (offset + tid.y)
    sB[shmem_row * BLOCK_SIDE + shmem_col] = B[(offset + shmem_row) * N + cCol]; // the same column of the result element (x, y)
                                                                                 // with the row offset by (offset + tid.x) * N
    __syncthreads();

    for (int i = 0; i < BLOCK_SIDE; i++)
    {
      tmp += sA[shmem_row * BLOCK_SIDE + i] * sB[(i * BLOCK_SIDE) + shmem_col];
    }
    __syncthreads();
    // C[x, y]
  }
  C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
}
#endif
