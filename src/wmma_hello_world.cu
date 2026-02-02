#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(err));                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// One warp computes one 16x16x16 GEMM: C = A * B + C
__global__ void wmma_gemm_16x16x16_f16f16f32(const half* A, const half* B, float* C) {
    // WMMA tile sizes
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    // We use one warp only.
    // blockDim.x should be 32; threadIdx.x in [0,31]
    // No need for shared memory in the minimal example (load from global).
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // Initialize accumulator to 0
    wmma::fill_fragment(c_frag, 0.0f);

    // Leading dimensions
    // A is row-major 16x16 => lda = 16
    // B is col-major 16x16 => ldb = 16
    const int lda = K;
    const int ldb = K;

    // Load A and B fragments (each warp cooperatively loads)
    wmma::load_matrix_sync(a_frag, A, lda);
    wmma::load_matrix_sync(b_frag, B, ldb);

    // Tensor Core MMA: c_frag = a_frag * b_frag + c_frag
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store back to C (row-major)
    const int ldc = N;
    wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
}

static void host_ref_gemm(const half* A, const half* B, float* C) {
    // A: row-major, B: col-major, C: row-major
    // C = A * B
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float acc = 0.f;
            for (int k = 0; k < 16; ++k) {
                // A(i,k) in row-major: A[i*16 + k]
                float a = __half2float(A[i * 16 + k]);

                // B(k,j) but B is col-major:
                // element (k,j) stored at B[j*16 + k]
                float b = __half2float(B[j * 16 + k]);

                acc += a * b;
            }
            C[i * 16 + j] = acc;
        }
    }
}

int main() {
    // Host buffers
    std::vector<half> hA(16 * 16);
    std::vector<half> hB(16 * 16);
    std::vector<float> hC(16 * 16, 0.f);
    std::vector<float> hC_ref(16 * 16, 0.f);

    // Initialize A (row-major) and B (col-major) with simple values
    // Keep values small to reduce FP16 rounding differences.
    for (int i = 0; i < 16; ++i) {
        for (int k = 0; k < 16; ++k) {
            float v = (i + k) % 7 - 3; // [-3,3]
            hA[i * 16 + k] = __float2half(v);
        }
    }

    // B is col-major, so fill by (col, row)
    for (int j = 0; j < 16; ++j) {
        for (int k = 0; k < 16; ++k) {
            float v = (j * 2 + k) % 9 - 4; // [-4,4]
            hB[j * 16 + k] = __float2half(v);
        }
    }

    // Device buffers
    half* dA = nullptr;
    half* dB = nullptr;
    float* dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Launch: 1 block, 1 warp
    dim3 block(32, 1, 1);
    dim3 grid(1, 1, 1);
    wmma_gemm_16x16x16_f16f16f32<<<grid, block>>>(dA, dB, dC);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Reference
    host_ref_gemm(hA.data(), hB.data(), hC_ref.data());

    // Compare
    float max_abs_err = 0.f;
    float max_rel_err = 0.f;
    for (int idx = 0; idx < 256; ++idx) {
        float got = hC[idx];
        float ref = hC_ref[idx];
        float abs_err = std::fabs(got - ref);
        float rel_err = abs_err / (std::fabs(ref) + 1e-6f);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    printf("WMMA 16x16x16 (A f16 row-major, B f16 col-major, C f32)\n");
    printf("max_abs_err = %.6f, max_rel_err = %.6f\n", max_abs_err, max_rel_err);

    // Print a small corner of C
    printf("C[0:4,0:4]:\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%8.2f ", hC[i * 16 + j]);
        }
        printf("\n");
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}