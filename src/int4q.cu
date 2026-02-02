#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

// -------------------------------
// INT4 helpers (signed int4)
// range: [-8, 7]
// pack two int4 into one byte:
//   low  nibble: element 2i
//   high nibble: element 2i+1
// -------------------------------
__device__ __forceinline__ int clamp_int4(int v) {
  v = (v < -8) ? -8 : v;
  v = (v >  7) ?  7 : v;
  return v;
}

__device__ __forceinline__ uint8_t pack_int4(int lo, int hi) {
  // store as 4-bit two's complement in each nibble
  uint8_t ulo = static_cast<uint8_t>(lo) & 0x0F;
  uint8_t uhi = (static_cast<uint8_t>(hi) & 0x0F) << 4;
  return static_cast<uint8_t>(ulo | uhi);
}

__host__ __device__ __forceinline__ int unpack_int4_lo(uint8_t b) {
  int v = b & 0x0F;
  // sign-extend from 4-bit two's complement
  v = (v & 0x08) ? (v | ~0x0F) : v;
  return v;
}

__host__ __device__ __forceinline__ int unpack_int4_hi(uint8_t b) {
  int v = (b >> 4) & 0x0F;
  v = (v & 0x08) ? (v | ~0x0F) : v;
  return v;
}

// -------------------------------
// Kernel: quantize float -> packed int4
// symmetric quant: q = round(x/scale), clamp to [-8,7]
// -------------------------------
__global__ void quantize_to_int4_packed(
    const float* __restrict__ x,
    uint8_t* __restrict__ q_packed,
    int n,
    float scale)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pack_idx = tid;              // each thread handles one packed byte
  int i0 = pack_idx * 2;
  int i1 = i0 + 1;
  if (i0 >= n) return;

  float inv = 1.0f / scale;

  int q0 = __float2int_rn(x[i0] * inv);
  q0 = clamp_int4(q0);

  int q1 = 0;
  if (i1 < n) {
    q1 = __float2int_rn(x[i1] * inv);
    q1 = clamp_int4(q1);
  }

  q_packed[pack_idx] = pack_int4(q0, q1);
}

// -------------------------------
// Kernel: dequant packed int4 -> float
// x_hat = q * scale
// -------------------------------
__global__ void dequantize_from_int4_packed(
    const uint8_t* __restrict__ q_packed,
    float* __restrict__ x_hat,
    int n,
    float scale)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pack_idx = tid;
  int i0 = pack_idx * 2;
  int i1 = i0 + 1;
  if (i0 >= n) return;

  uint8_t b = q_packed[pack_idx];
  int q0 = unpack_int4_lo(b);
  int q1 = unpack_int4_hi(b);

  x_hat[i0] = static_cast<float>(q0) * scale;
  if (i1 < n) x_hat[i1] = static_cast<float>(q1) * scale;
}

// CPU reference: compute symmetric scale from max abs
static float compute_symmetric_scale(const std::vector<float>& x) {
  float max_abs = 0.0f;
  for (float v : x) max_abs = std::max(max_abs, std::fabs(v));
  // int4 signed max magnitude is 7 (because range [-8,7], but symmetric usually uses 7)
  // If max_abs==0, scale can be 1 to avoid div0.
  if (max_abs == 0.0f) return 1.0f;
  return max_abs / 7.0f;
}

int main() {
  // small demo
  const int N = 33; // intentionally odd to show tail handling
  std::vector<float> h_x(N);

  // Make some values (mix small/large, positive/negative)
  for (int i = 0; i < N; ++i) {
    float v = 0.1f * (i - 16);
    // if (i % 7 == 0) v *= 5.0f;
    // if (i % 11 == 0) v *= -3.0f;
    h_x[i] = v;
  }

  float scale = compute_symmetric_scale(h_x);

  // device buffers
  float* d_x = nullptr;
  float* d_xhat = nullptr;
  uint8_t* d_qpack = nullptr;

  int packed_bytes = (N + 1) / 2;

  ck(cudaMalloc(&d_x,    N * sizeof(float)), "malloc d_x");
  ck(cudaMalloc(&d_xhat, N * sizeof(float)), "malloc d_xhat");
  ck(cudaMalloc(&d_qpack, packed_bytes * sizeof(uint8_t)), "malloc d_qpack");

  ck(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D x");

  int threads = 128;
  int blocks_pack = (packed_bytes + threads - 1) / threads;

  quantize_to_int4_packed<<<blocks_pack, threads>>>(d_x, d_qpack, N, scale);
  ck(cudaGetLastError(), "launch quantize");
  dequantize_from_int4_packed<<<blocks_pack, threads>>>(d_qpack, d_xhat, N, scale);
  ck(cudaGetLastError(), "launch dequantize");
  ck(cudaDeviceSynchronize(), "sync");

  std::vector<float> h_xhat(N);
  std::vector<uint8_t> h_qpack(packed_bytes);

  ck(cudaMemcpy(h_xhat.data(), d_xhat, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H xhat");
  ck(cudaMemcpy(h_qpack.data(), d_qpack, packed_bytes * sizeof(uint8_t), cudaMemcpyDeviceToHost), "D2H qpack");

  // Print results
  std::printf("=== Ampere INT4 Quant Demo (symmetric, zp=0) ===\n");
  std::printf("N=%d, packed_bytes=%d, scale=%.8f\n\n", N, packed_bytes, scale);

  std::printf("idx | x(float)      | q(int4) | x_hat(float)  | abs_err\n");
  std::printf("----+---------------+---------+--------------+--------\n");

  auto get_q = [&](int i)->int {
    uint8_t b = h_qpack[i/2];
    return (i % 2 == 0) ? unpack_int4_lo(b) : unpack_int4_hi(b);
  };

  double max_err = 0.0, mse = 0.0;
  for (int i = 0; i < N; ++i) {
    int q = get_q(i);
    float err = std::fabs(h_x[i] - h_xhat[i]);
    max_err = std::max<double>(max_err, err);
    mse += double(h_x[i] - h_xhat[i]) * double(h_x[i] - h_xhat[i]);
    std::printf("%3d | %13.6f | %7d | %12.6f | %7.6f\n",
                i, h_x[i], q, h_xhat[i], err);
  }
  mse /= N;
  std::printf("\nmax_abs_err=%.8f, mse=%.10f\n", (float)max_err, (float)mse);

  // Show packed bytes
  std::printf("\nPacked bytes (each contains 2 int4, low then high nibble):\n");
  for (int i = 0; i < packed_bytes; ++i) {
    std::printf("byte[%2d] = 0x%02X\n", i, (unsigned)h_qpack[i]);
  }

  cudaFree(d_x);
  cudaFree(d_xhat);
  cudaFree(d_qpack);
  return 0;
}