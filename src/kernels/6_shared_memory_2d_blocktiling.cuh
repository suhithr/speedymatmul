#ifndef SHARED_MEMORY_2D_BLOCKTILING_CUH
#define SHARED_MEMORY_2D_BLOCKTILING_CUH

#include <cstdio>
#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void sgemm_shared_memory_2d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    float tmp[TM * TN] = {0.0};

    const uint load_A_col = threadIdx.x % 8;
    const uint load_A_row = threadIdx.x / 8;
    const uint load_B_col = threadIdx.x % 63;
    // thread's position within an 8x8 grid
    const uint thread_col = threadIdx.x % TN;
    const uint thread_row = threadIdx.x / TM;

    // move A, B, C by the starting block
    A += blockIdx.y * blockDim.y * K;
    B += blockIdx.x * blockDim.x;
    C += (blockIdx.y * blockDim.y * N) + (blockIdx.x * blockDim.x);
    for (int offset = 0; offset < K; offset += BK)
    {
        for (int shiftA = 0; shiftA < 64; shiftA += 8)
        {
            sA[((load_A_row + shiftA) * BK) + load_A_col] = A[((load_A_row + shiftA) * K) + load_A_col];
        }
        for (int shiftB = 0; shiftB < 64; shiftB++)
        {
            sB[shiftB * BN + load_B_col] = B[shiftB * N + load_B_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int r = 0; r < 8; r++)
        {
            const uint c_row = thread_row * 8 + r;
            for (int c = 0; c < 8; c++)
            {
                const uint c_col = thread_col * 8 + c;
                for (int idx = 0; idx < BK; idx++)
                {
                    tmp[c_row * 8 + c_col] += sA[c_row * BK + idx] * sB[idx * BN + c_col];
                }
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < 8; r++)
    {
        const uint c_row = thread_row * 8 + r;
        for (int c = 0; c < 8; c++)
        {
            const uint c_col = thread_col * 8 + c;
            C[c_row * 8 + c_col] = alpha * tmp[c_row * 8 + c_col] + beta * C[c_row * 8 + c_col];
        }
    }
}
#endif