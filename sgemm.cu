// #include "kernels.cuh"
#include "run_kernels.cuh"
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixMultiplicationMistake.txt";

/*
CLI:
./sgemm {kernel_num} {--profile}
*/
int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr << "Select a kernel (range 0 - 8)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // read kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 8)
  {
    std::cerr << "Please enter a valid kernel number (0-8)" << std::endl;
    exit(EXIT_FAILURE);
  }
  bool profile_mode = false;
  if (argc >= 3 && std::strcmp(argv[2], "--profile") == 0)
  {
    profile_mode = true;
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL)
  {
    deviceIdx = std::atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  CudaDeviceInfo();

  // we use cudaEvent to push event tasks into our execution stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  std::vector<int> SIZE = {128, 256,
                           512, 1024, 2048}; //4096};
                           // at size 4096 we seem to run into floating point accuracy issues
                           // and are not matching cublas

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.6, beta = 4.0; // GEMM parameters C=alpha*AB + beta*C

  float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr;

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "CUBLAS initialization failed";
    exit(EXIT_FAILURE);
  }

  int repeat_times = 500;
  // Debugging kernels where we don't want to print much
  if (kernel_num == 0)
  {
    SIZE = {32};
    repeat_times = 50;
  }
  // In profile mode we want to warm up the GPU. So we run 50 times
  // pass in the --launch-skip 49 flag to NCU to skip the first 49 launches
  if (profile_mode)
  {

    SIZE = {2048};
    repeat_times = 50;
  }
  // SIZE = {64};
  for (int size : SIZE)
  {
    m = n = k = size;
    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;

    // Verify correctness once, executing it once
    // also avoids cold-start issues with the timing
    // eg: starting from an idle clock speed, JIT compilation/kernel caching
    // which happens on the first run, or even memory page allocation

    if (kernel_num > 1 && profile_mode == false)
    {

      run_kernel(1, m, n, k, alpha, dA, dB, beta, dC_ref, handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);

      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());

      cudaCheck(
          cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n,
                           cudaMemcpyDeviceToHost));
      if (!verify_matrix(C_ref, C, m * n))
      {
        std::cerr << "Failed to pass correctness verification against cuBLAS";
        if (m <= 128)
        {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }
    cudaCheck(cudaEventRecord(beg));
    for (int i = 0; i < repeat_times; i++)
    {
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));

    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.0; // convert ms to seconds

    long flops = 2 * k * m * n;
    printf("Average elapsed time: (%8.6f) s, performance: (%4.1f) GFLOPS. "
           "Size: (%ld).\n",
           elapsed_time / repeat_times,
           (flops * repeat_times) / (1e9 * elapsed_time), m);
    fflush(stdout);

    // reset dC to the dC_ref value
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  free(A);
  free(B);
  free(C);
  cudaCheck(cudaFree(dA));
  cudaCheck(cudaFree(dB));
  cudaCheck(cudaFree(dC));
  cudaCheck(cudaFree(dC_ref));
  cublasDestroy(handle);

  return 0;
}