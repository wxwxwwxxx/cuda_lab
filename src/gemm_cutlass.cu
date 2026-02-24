#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                            \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

static void gemm_cpu_ref(const cutlass::half_t* A,
                         const cutlass::half_t* B,
                         float* C,
                         int M,
                         int N,
                         int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float af = __half2float(reinterpret_cast<const __half*>(A)[i * K + k]);
        float bf = __half2float(reinterpret_cast<const __half*>(B)[k * N + j]);
        acc += af * bf;
      }
      C[i * N + j] = acc;
    }
  }
}

static void fill_random(std::vector<cutlass::half_t>& x,
                        float lo = -1.0f,
                        float hi = 1.0f,
                        uint32_t seed = 123) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& v : x) {
    v = cutlass::half_t(dist(rng));
  }
}

static void compare(const std::vector<float>& ref,
                    const std::vector<float>& out,
                    float atol = 1e-2f,
                    float rtol = 1e-2f) {
  float max_abs = 0.0f;
  int bad = 0;
  for (int i = 0; i < static_cast<int>(ref.size()); ++i) {
    float abs_err = std::fabs(ref[i] - out[i]);
    max_abs = std::max(max_abs, abs_err);
    float tol = atol + rtol * std::fabs(ref[i]);
    if (abs_err > tol) {
      ++bad;
    }
  }
  std::cout << "max_abs_err=" << max_abs << " bad=" << bad << "/" << ref.size() << "\\n";
  std::cout << (bad == 0 ? "✅ PASS" : "❌ FAIL") << "\\n";
}

int main() {
  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<ElementInputA,
                                                   LayoutInputA,
                                                   ElementInputB,
                                                   LayoutInputB,
                                                   ElementOutput,
                                                   LayoutOutput,
                                                   ElementAccumulator>;

  constexpr int M = 2048;
  constexpr int N = 2048;
  constexpr int K = 2048;

  const size_t bytesA = static_cast<size_t>(M) * K * sizeof(ElementInputA);
  const size_t bytesB = static_cast<size_t>(K) * N * sizeof(ElementInputB);
  const size_t bytesC = static_cast<size_t>(M) * N * sizeof(ElementOutput);

  std::vector<ElementInputA> hA(static_cast<size_t>(M) * K);
  std::vector<ElementInputB> hB(static_cast<size_t>(K) * N);
  std::vector<ElementOutput> hC(static_cast<size_t>(M) * N, 0);
  std::vector<ElementOutput> hCRef(static_cast<size_t>(M) * N, 0);

  fill_random(hA, -1.0f, 1.0f, 49);
  fill_random(hB, -1.0f, 1.0f, 59);

  auto t0 = std::chrono::high_resolution_clock::now();
  gemm_cpu_ref(hA.data(), hB.data(), hCRef.data(), M, N, K);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "CPU reference: "
            << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\\n";

  ElementInputA* dA = nullptr;
  ElementInputB* dB = nullptr;
  ElementOutput* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, bytesC));

  CutlassGemm gemm_op;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  typename CutlassGemm::Arguments arguments{
      problem_size,
      {dA, K},
      {dB, N},
      {dC, N},
      {dC, N},
      {1.0f, 0.0f}};

  cutlass::Status status = gemm_op(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM launch failed: " << cutlassGetStatusString(status) << "\\n";
    return 1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  status = gemm_op(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM run failed: " << cutlassGetStatusString(status) << "\\n";
    return 1;
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  std::cout << "CUTLASS kernel time: " << kernel_ms << " ms\\n";

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
  compare(hCRef, hC);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
