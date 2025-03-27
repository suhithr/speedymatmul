#ifndef MEMORY_INSTRUMENT_CUH
#define MEMORY_INSTRUMENT_CUH

#include <cuda_runtime.h>
const uint MAX_NUMBER_OF_ACCESSES = 2000;

void allocate_instrumentation_buffers(const dim3 dimGrid, const dim3 dimBlock, uint *access_locations, long long int *access_times)
{
    uint num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
    uint num_threads = dimBlock.x * dimBlock.y * dimBlock.z;
    uint num_elements_to_allocate = num_blocks * num_threads * MAX_NUMBER_OF_ACCESSES;
    cudaMalloc((void **)&access_locations, sizeof(uint) * num_elements_to_allocate);
    cudaMalloc((void **)&access_times, sizeof(long long int) * num_elements_to_allocate);
}

void free_instrumentation_buffers(uint *access_locations, float *access_times) {
    cudaFree(access_locations);
    cudaFree(access_times);
}

/*Wrap an ELEMENT_DTYPE pointer in the kernel and log all the accesses & clocks in the kernel*/
template <typename Element_Dtype>
class MemoryInstrument
{
    // The pointer data that we are wrapping with instrumentation
    const Element_Dtype *data;
    const uint tid;
    const uint thread_idx_in_block;
    const uint block_idx;
    // location in GMEM
    // location in SMEM
    long long int start_clock; // the SM start clock (this is populated when the kernel is launched)
    const uint SM_id;
    long long int *access_times;
    uint *access_locations;
    uint buffer_idx;
    uint entry_count;

    __host__ MemoryInstrument(const Element_Dtype *device_data, uint *access_locations, float *access_times) : data(device_data),
                                                                                                               tid(0),
                                                                                                               thread_idx_in_block(0),
                                                                                                               block_idx(0),
                                                                                                               buffer_idx(0),
                                                                                                               access_locations(access_locations),
                                                                                                               access_times(access_times),
                                                                                                               entry_count(0)
    {
    }

    // run this inside the kernel, before doing anything else
    // initializing the threads, blocks, and buffer_idx as this is local
    __device__ void kernel_entrypoint()
    {
        tid = threadIdx.x  + blockDim.x*(threadIdx.y + blockDim.y * threadIdx.z);

    }

    __device__ Element_Dtype& operator[](int original_index)
    {
        buffer_idx = (block_idx * block_size + tid) * MAX_NUMBER_OF_ACCESSES + entry_count;
        long long int current_clock = clock64();

        access_times[buffer_idx] = current_clock;
        access_locations[buffer_idx] = &data[original_index];

        entry_count = std::min(entry_count+1, MAX_NUMBER_OF_ACCESSES);
        return data[original_index];
    }
};

#endif