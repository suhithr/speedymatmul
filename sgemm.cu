#include "run_kernels.cuh"
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Select a kernel (range 0 - 0)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // read kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 0) {
    std::cerr << "Please enter a valid kernel number (0-0)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = std::atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  CudaDeviceInfo();

  // we use cudaEvent to push event tasks into our execution stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  std::vector<int> SIZE = {128, 256}; // , 512, 1024, 2048, 4096, 8192, 16384};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.6, beta = 4.0; // GEMM parameters C=alpha*AB + beta*C

  float *A = nullptr, *B = nullptr, *C = nullptr;
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = k = size;
    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;

    run_kernel(0, m, n, k, alpha, dA, dB, beta, dC);
  }
}