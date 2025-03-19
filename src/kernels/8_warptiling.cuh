#ifndef WARPTILING_CUH
#define WARPTILING_CUH

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN, const uint thread_tile_rows, const uint thread_tile_cols>
__global__ void sgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)//  float* sA, float* sB)
{
    const uint tid = threadIdx.x + (blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
    const uint warp_id = tid / warpSize;
    const uint warp_tile_cols = BN / (TN * thread_tile_cols);
    const uint warp_tile_rows = BM / (TM * thread_tile_rows);
    assert(BN % (TN * thread_tile_cols) == 0);
    assert(BM % (TM * thread_tile_rows) == 0);
    // warp's x, y
    const uint warp_row = warp_id / warp_tile_cols;
    const uint warp_col = warp_id % warp_tile_cols;
    assert(warp_row <= warp_tile_rows);
    const uint thread_lane = tid % warpSize;

    // thread's x, y relative to warp
    const uint thread_tile_row = thread_lane / thread_tile_cols;
    const uint thread_tile_col = thread_lane % thread_tile_cols;
    // This is the size of the block that we are computing
    const uint resultsFromBlocktile = BM * BN;
    const uint numThreadsBlocktile = resultsFromBlocktile / (TM * TN);
    assert(numThreadsBlocktile == blockDim.x); // blockDim.x must be set up to be the right number of threads
    assert(numThreadsBlocktile % warpSize == 0);

    const uint load_A_col = threadIdx.x % (BK/4); // 0..1
    const uint load_A_row = threadIdx.x / (BK/4); // 0..31
    const uint strideArows = (numThreadsBlocktile / BK) * 4;
    assert(((numThreadsBlocktile) % BM == 0)); // The total num of threads must be evenly divisible by BM
    assert(((numThreadsBlocktile) % BK == 0)); // The total num of threads must be evenly divisible by BK

    const uint load_B_col = threadIdx.x % (BN / 4);
    const uint load_B_row = threadIdx.x / (BN / 4);
    const uint strideB = (numThreadsBlocktile / BN) * 4;
    assert((numThreadsBlocktile) % BN == 0);

    // The threads position within a grid of (BM/TM) x (BN/TN)
    // as it's divied up in this way.

    __shared__ float sA[BK * BM];
    __shared__ float sB[BK * BN];
    float registerA[TM] = {0.0}, registerB[TN] = {0.0}, acc_cache[TM * TN] = {0.0}; // register caches

    // move A, B, C by the starting block
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (blockIdx.y * BM * N) + (blockIdx.x * BN);

    const uint thread_row_in_block = ((warp_row * thread_tile_rows + thread_tile_row) * TM);
    const uint thread_col_in_block = ((warp_col * thread_tile_cols + thread_tile_col) * TN);
    for (int offset = 0; offset < K; offset += BK)
    {
        // Load from GMEM to SMEM
        // these loops are for the thread-level shifting to encourage coalescing of GMEM loads
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
            *reinterpret_cast<float4 *>(&sB[(load_B_row + shiftB) * BN + load_B_col * 4]) = *reinterpret_cast<float4 *>(&B[(load_B_row + shiftB) * N + load_B_col * 4]);
        }

        // // shift A & B pointers
        __syncthreads();
        A += BK;
        B += BK * N;

        // // Using the caches (registers) to store elements
        // // to reduce shared memory accesses. (for example sB is hit just 1x per TN, not for every TM)
        for (int frgmtOffset = 0; frgmtOffset < BK; ++frgmtOffset)
        {
            // load to registers
            for (int i = 0; i < TM; i++)
            {
                registerA[i] = sA[frgmtOffset * BM + thread_row_in_block + i];
            }
            for (int i = 0; i < TN; i++)
            {
                registerB[i] = sB[frgmtOffset * BN + thread_col_in_block + i];
            }

            for (int rid = 0; rid < TM; ++rid)
            {
                for (int cid = 0; cid < TN; ++cid)
                {
                    acc_cache[rid * TN + cid] += registerA[rid] * registerB[cid];
                }
            }
        }
        __syncthreads();
    }
    for (int rinc = 0; rinc < TM; ++rinc)
    {
        const uint crow = (thread_row_in_block + rinc) * N;
        for (int cinc = 0; cinc < TN; ++cinc)
        {
            const uint ccol = thread_col_in_block + cinc;
            C[crow + ccol] = alpha * acc_cache[rinc * TN + cinc] + beta * C[crow + ccol];
        }
    }
}
#endif