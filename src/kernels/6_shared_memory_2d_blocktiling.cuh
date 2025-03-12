#ifndef SHARED_MEMORY_2D_BLOCKTILING_CUH
#define SHARED_MEMORY_2D_BLOCKTILING_CUH

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void sgemm_shared_memory_2d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    // This is the size of the block that we are computing
    const uint resultsFromBlocktile = BM * BN;
    // Each thread computes TM*TN results so, how many threads do we need in this blocktile
    // to compute all the elements?
    const uint numThreadsBlocktile = resultsFromBlocktile / (TM * TN);
    assert(numThreadsBlocktile == blockDim.x); // blockDim.x must be set up to be the right number of threads
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    float tmp[TM * TN] = {0.0}; // a thread's register cache
    float sharedA_cache[TM] = {0.0};
    float sharedB_cache[TN] = {0.0};

    const uint load_A_col = threadIdx.x % BK;
    const uint load_A_row = threadIdx.x / BK;
    const uint strideA = numThreadsBlocktile / BK;
    assert(((numThreadsBlocktile) % BK == 0)); // The total num of threads must be evenly divisible by BK
                                               // so we can skip strideA complete rows while loading a tile.

    const uint load_B_col = threadIdx.x % BN;
    const uint load_B_row = threadIdx.x / BN;
    const uint strideB = numThreadsBlocktile / BN;
    assert((numThreadsBlocktile) % BN == 0);

    // The threads position within a grid of (BM/TM) x (BN/TN)
    // as it's divied up in this way.
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // move A, B, C by the starting block
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (blockIdx.y * BM * N) + (blockIdx.x * BN);
    for (int offset = 0; offset < K; offset += BK)
    {
        for (int shiftA = 0; shiftA < BM; shiftA += strideA)
        {
            sA[((load_A_row + shiftA) * BK) + load_A_col] = A[((load_A_row + shiftA) * K) + load_A_col];
        }
        for (int shiftB = 0; shiftB < BK; shiftB += strideB)
        {
            sB[(load_B_row + shiftB) * BN + load_B_col] = B[(load_B_row + shiftB) * N + load_B_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        // Using the caches (registers) to store elements
        // to reduce shared memory accesses. (for example sB is hit just 1x per TN, not for every TM)
        for (int idx = 0; idx < BK; ++idx)
        {
            for (int r = 0; r < TM; ++r)
            {
                sharedA_cache[r] = sA[(thread_row * TM + r) * BK + idx];
            }
            for (int c = 0; c < TN; ++c)
            {
                sharedB_cache[c] = sB[(idx * BN) + (thread_col * TN + c)];
            }
            for (int r = 0; r < TM; ++r)
            {
                for (int c = 0; c < TN; ++c)
                {
                    tmp[r * TN + c] +=
                        sharedA_cache[r] * sharedB_cache[c];
                }
            }
        }
#if 0
        // Calculations without registers and with the dot.product on the inner loop
        for (int r = 0; r < TM; r++)
        {
            const uint c_row = thread_row * TM + r;
            for (int c = 0; c < TN; c++)
            {
                const uint c_col = thread_col * TN + c;
                for (int idx = 0; idx < BK; idx++)
                {
                    tmp[(r)*TN + c] += sA[c_row * BK + idx] * sB[idx * BN + c_col];
                }
            }
        }
        // Calculations without registers and with the dot.product on the outer loop
        for (int idx = 0; idx < BK; idx++)
        {
            for (int r = 0; r < TM; r++)
            {
                const uint c_row = (thread_row * TM + r) * BK + idx;
                for (int c = 0; c < TN; c++)
                {
                    const uint c_col = (idx * BN) + (thread_col * TN) + c;
                    tmp[r * TN + c] += sA[c_row] * sB[c_col];
                }
            }
        }
#endif
        __syncthreads();
    }
    for (int r = 0; r < TM; r++)
    {
        const uint c_row = thread_row * TM + r;
        for (int c = 0; c < TN; c++)
        {
            const uint c_col = thread_col * TN + c;
            C[c_row * N + c_col] = alpha * tmp[r * TN + c] + beta * C[c_row * N + c_col];
        }
    }
}
#endif