#ifndef WARPTILING_CUH 
#define WARPTILING_CUH 

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>


template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN, const uint warp_rows, const uint warp_cols>
__global__ void sgemm_vectorized_shared_memory_2d_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{

    const uint tid = threadIdx.x + (threadDim.x*(threadIdx.y + threadDim.y*threadIdx.z));
    const uint warp_id = tid / warpSize;
    const uint warp_row = warp_id / warp_cols, warp_col = warp_id % warp_cols;
    const uint thread_lane = tid % warpSize;

    const uint t_row_in_warp = (thread_lane / warp_rows) * TM;
    const uint t_col_in_warp = (thread_lane % warp_rows) * TN;

    // This is the size of the block that we are computing
    const uint resultsFromBlocktile = BM * BN;
    // Each thread computes TM*TN results so, how many threads do we need in this blocktile
    // to compute all the elements?
    const uint numThreadsBlocktile = resultsFromBlocktile / (TM * TN);
    assert(numThreadsBlocktile == blockDim.x); // blockDim.x must be set up to be the right number of threads
    assert (numThreadsBlockTile % warpSize == 0);
    __shared__ float sA[BK * BM];
    __shared__ float sB[BK * BN];

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

    float registerA[TM] = {0.0}, registerB[TN] = 0.0, acc_cache[TM*TN] = {0.0}; // register caches

    // move A, B, C by the starting block
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (blockIdx.y * BM * N) + (blockIdx.x * BN);
    for (int offset = 0; offset < K; offset += BK)
    {
        // Load from GMEM to SMEM
        // these loops are for the thread-level shifting to encourage coalescing of GMEM loads
        for (int shiftA = 0; shiftA < BM; shiftA += strideArows)
        {
            float4 tmp = reinterpret_cast<float4*>(&A[((load_A_row + shiftA) * K) + load_A_col * 4])[0];
            // sA[(load_A_col * 4 + 0) * BM + load_A_row] = tmp.x;
            // sA[(load_A_col * 4 + 1) * BM + load_A_row] = tmp.y;
            // sA[(load_A_col * 4 + 2) * BM + load_A_row] = tmp.z;
            // sA[(load_A_col * 4 + 3) * BM + load_A_row] = tmp.w;
            sA[(load_A_row * 4 + 0) * BK + load_A_col] = tmp.x;
            sA[(load_A_row * 4 + 1) * BK + load_A_col] = tmp.y;
            sA[(load_A_row * 4 + 2) * BK + load_A_col] = tmp.z;
            sA[(load_A_row * 4 + 3) * BK + load_A_col] = tmp.w;
        }
        for (int shiftB = 0; shiftB < BK; shiftB += strideB)
        {
            *reinterpret_cast<float4*>(&sB[(load_B_row + shiftB) * BN + load_B_col * 4]) = *reinterpret_cast<float4*>(&B[(load_B_row + shiftB) * N + load_B_col * 4]);
        }

        // shift A & B pointers
        __syncthreads();
        A += BK;
        B += BK * N;

        // Using the caches (registers) to store elements
        // to reduce shared memory accesses. (for example sB is hit just 1x per TN, not for every TM)
        for (int frgmtOffset = 0; frgmtOffset < BK; ++frgmtOffset)
        {
            // load to registers
            for (int i = 0; i < TM; i++) {
                registerA[i] = sA[(t_row_in_warp + i) * BK + frgmtOffset];
            }
            for (int i = 0; i < TN; i++) {
                registerB[i] = sB[frgmtOffset * BK + t_col_in_warp + i];
            }
            

        }
        __syncthreads();
    }
}
#endif