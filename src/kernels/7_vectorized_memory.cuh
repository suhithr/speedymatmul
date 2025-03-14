#ifndef VECTORIZED_SHARED_MEMORY_2D_BLOCKTILING_CUH
#define VECTORIZED_SHARED_MEMORY_2D_BLOCKTILING_CUH

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void sgemm_vectorized_shared_memory_2d_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // This is the size of the block that we are computing
    const uint resultsFromBlocktile = BM * BN;
    // Each thread computes TM*TN results so, how many threads do we need in this blocktile
    // to compute all the elements?
    const uint numThreadsBlocktile = resultsFromBlocktile / (TM * TN);
    assert(numThreadsBlocktile == blockDim.x); // blockDim.x must be set up to be the right number of threads
    __shared__ float sA[BK * BM];
    __shared__ float sB[BK * BN];
    float tmp[TM * TN] = {0.0}; // a thread's register cache
    float sharedA_cache[TM] = {0.0};
    float sharedB_cache[TN] = {0.0};

    const uint vector_size = 4;

    const uint load_A_col = threadIdx.x % (BK/4); // 0..1
    const uint load_A_row = threadIdx.x / (BK/4); // 0..31
    const uint strideArows = (numThreadsBlocktile / BK) * 4;
    assert(((numThreadsBlocktile) % BM == 0)); // The total num of threads must be evenly divisible by BM
                                               // so we can skip strideArows complete rows while loading a tile.
    assert(((numThreadsBlocktile) % BK == 0)); // The total num of threads must be evenly divisible by BK as well

    const uint load_B_col = threadIdx.x % (BN/4);
    const uint load_B_row = threadIdx.x / (BN/4);
    const uint strideB = (numThreadsBlocktile / BN) * 4;
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
        for (int shiftA = 0; shiftA < BM; shiftA += strideArows)
        {
            float4 tmp = reinterpret_cast<float4*>(&A[((load_A_row + shiftA) * K) + load_A_col * 4])[0];
            sA[(load_A_col * 4 + 0) * BM + load_A_row] = tmp.x;
            sA[(load_A_col * 4 + 1) * BM + load_A_row] = tmp.y;
            sA[(load_A_col * 4 + 2) * BM + load_A_row] = tmp.z;
            sA[(load_A_col * 4 + 3) * BM + load_A_row] = tmp.w;
        }
        for (int shiftB = 0; shiftB < BK; shiftB += strideB)
        {
            *reinterpret_cast<float4*>(&sB[(load_B_row + shiftB) * BN + load_B_col * 4]) = *reinterpret_cast<float4*>(&B[(load_B_row + shiftB) * N + load_B_col * 4]);
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
                sharedA_cache[r] = sA[(idx * BM) + (thread_row * TM + r)];
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