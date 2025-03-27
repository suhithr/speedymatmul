#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "kernels/2_naive.cuh"
#include "kernels/3_global_memory_coalescing.cuh"
#include "kernels/0_print_coalesc_memory_address.cuh"
#include "kernels/4_shared_memory_blocking.cuh"
#include "kernels/5_shared_memory_1d_blocktiling.cuh"
#include "kernels/6_shared_memory_2d_blocktiling.cuh"
#include "kernels/7_vectorized_memory.cuh"
#include "kernels/8_warptiling.cuh"

#endif