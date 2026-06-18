// Tensor-core ceiling microbenchmark for the LIO density-as-GEMM reform.
// Measures cuBLAS FP32 (SGEMM) vs TF32 tensor vs FP16 tensor at the matrix
// shapes the density build would produce: C[M_g, ncols] = P[M_g,M_g] * Phi[M_g,ncols]
// i.e. M=N_basis_group, K=N_basis_group (SMALL), N=npoints*components (large).
// Small K is the crux: tensor cores need K to amortize the MMA pipeline.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static double bench(cublasHandle_t h, cublasComputeType_t ct, cublasGemmAlgo_t algo,
                    int M, int N, int K, const float* dA, const float* dB, float* dC) {
  const float alpha = 1.f, beta = 0.f;
  // warmup
  for (int i = 0; i < 5; i++)
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                 dA, CUDA_R_32F, K, dB, CUDA_R_32F, K, &beta,
                 dC, CUDA_R_32F, M, ct, algo);
  cudaDeviceSynchronize();
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  const int reps = 50;
  cudaEventRecord(s);
  for (int i = 0; i < reps; i++)
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                 dA, CUDA_R_32F, K, dB, CUDA_R_32F, K, &beta,
                 dC, CUDA_R_32F, M, ct, algo);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms / reps;
}

int main() {
  cublasHandle_t h; cublasCreate(&h);
  struct Shape { int M, K, N; const char* tag; };
  std::vector<Shape> shapes = {
    {128, 128, 4096,  "small group, ncols=4096"},
    {192, 192, 8192,  "mid group,   ncols=8192"},
    {256, 256, 16384, "large group, ncols=16384"},
    {512, 512, 16384, "big basis,   ncols=16384"},
  };
  printf("%-28s %10s %10s %10s   %6s %6s\n",
         "shape", "FP32(ms)", "TF32(ms)", "FP16(ms)", "TF32x", "FP16x");
  for (auto& s : shapes) {
    size_t szA = (size_t)s.M * s.K, szB = (size_t)s.K * s.N, szC = (size_t)s.M * s.N;
    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA * 4); cudaMalloc(&dB, szB * 4); cudaMalloc(&dC, szC * 4);
    cudaMemset(dA, 1, szA * 4); cudaMemset(dB, 1, szB * 4);
    double f32 = bench(h, CUBLAS_COMPUTE_32F,            CUBLAS_GEMM_DEFAULT,        s.M, s.N, s.K, dA, dB, dC);
    double tf32= bench(h, CUBLAS_COMPUTE_32F_FAST_TF32,  CUBLAS_GEMM_DEFAULT_TENSOR_OP, s.M, s.N, s.K, dA, dB, dC);
    double f16 = bench(h, CUBLAS_COMPUTE_32F_FAST_16F,   CUBLAS_GEMM_DEFAULT_TENSOR_OP, s.M, s.N, s.K, dA, dB, dC);
    printf("%-28s %10.4f %10.4f %10.4f   %5.2fx %5.2fx\n",
           s.tag, f32, tf32, f16, f32/tf32, f32/f16);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }
  cublasDestroy(h);
  return 0;
}
