#include <cuda_runtime.h>
#include <cuda/barrier> // 必须保留，这是 libcu++ 的核心

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <mma.h>

using namespace nvcuda;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
using token_t   = barrier_t::arrival_token;

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
              cudaGetErrorString(_e));                                            \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

constexpr int tile = 16;
constexpr int BM = tile * 8; // 256
constexpr int BN = tile * 8; // 256
constexpr int BK = 32;
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int lda = BK+APAD;
constexpr int ldb = BN+BPAD;
constexpr int lda_vec = BK>>3; //8 half at once
constexpr int ldb_vec = BN>>3;


// ---------------------------
// 6) Kernel signature only (no implementation)
// ---------------------------
__global__ void gemm_kernel(const half* __restrict__ A,
                            const half* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    // TODO: implement C = A * B
    // A: [M, K] row-major
    // B: [K, N] row-major
    // C: [M, N] row-major
    const half* __restrict__ A_ = (const half*)__builtin_assume_aligned(A, 32);
    const half* __restrict__ B_ = (const half*)__builtin_assume_aligned(B, 32);
    const uint4* A4 = reinterpret_cast<const uint4*>(A_);
    const uint4* B4 = reinterpret_cast<const uint4*>(B_);

    int n_block_num = N/BN;
    // int m_block_num = M/BM;

    // __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    // -----------------------------------------------------------
    // [修改 1]：不要直接声明对象，而是声明对齐的裸内存
    // -----------------------------------------------------------
    // barrier_t 的大小和对齐要求
            // __shared__ alignas(barrier_t) char smem_bar_copy_raw[2 * sizeof(barrier_t)];
            // __shared__ alignas(barrier_t) char smem_bar_consume_raw[2 * sizeof(barrier_t)];

    // 将裸内存强转为对象指针
    __shared__ barrier_t bar_copy[2];
    __shared__ barrier_t bar_consume[2];
    if (threadIdx.x == 0) {
        // 这里的语法是：new (地址) 类型(参数);
        // 这会在指定的内存地址上调用构造函数，而不分配新内存
        init(&bar_copy[0],blockDim.x);
        init(&bar_copy[1],blockDim.x);
        init(&bar_consume[0],blockDim.x);
        init(&bar_consume[1],blockDim.x);
    }
    __syncthreads();

    __shared__ alignas(16) half smemA[2][BM][BK+APAD];
    __shared__ alignas(16) half smemB[2][BK][BN+BPAD];
    // int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
//    int c_tile_y = warp_id >> 2;
//    int c_tile_x = warp_id & 3;
    int cur_buf=0, next_buf=1;
    int m_pos = (blockIdx.x/n_block_num)*BM;
    int n_pos = (blockIdx.x%n_block_num)*BN;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[8];
    #pragma unroll
    for(int i=0;i<8;i++)
    {
        wmma::fill_fragment(c_frag[i],0.0f);
    }


    // move to smem
    token_t tok_copy[2];
    token_t tok_done[2];
    tok_done[next_buf] = bar_consume[next_buf].arrive();
    auto move_data = [&](int k_pos,int buf){
        // move A
        // BK and BN must be less than 256 (lane num*8)
        uint4* smemA_vec = reinterpret_cast<uint4*>(&smemA[buf][0][0]);
        for(int t=threadIdx.x;t<((BM*BK)>>3);t+=blockDim.x)
        {
            int vec_ay=t/lda_vec;
            int vec_ax=t%lda_vec;
            cuda::memcpy_async(smemA_vec+vec_ay*(lda>>3)+vec_ax,A4 + (m_pos+vec_ay)*(K>>3) + (k_pos>>3) + vec_ax,cuda::aligned_size_t<16>(sizeof(uint4)),bar_copy[buf]);
        }
        // move B
        uint4* smemB_vec = reinterpret_cast<uint4*>(&smemB[buf][0][0]);
        for(int t=threadIdx.x;t<((BN*BK)>>3);t+=blockDim.x)
        {
            int vec_by=t/ldb_vec;
            int vec_bx=t%ldb_vec;
            cuda::memcpy_async(smemB_vec+vec_by*(ldb>>3)+vec_bx,B4 + (k_pos+vec_by)*(N>>3) + (n_pos>>3) + vec_bx,cuda::aligned_size_t<16>(sizeof(uint4)),bar_copy[buf]);
        }
    };
    move_data(0,cur_buf);
    tok_copy[cur_buf]=bar_copy[cur_buf].arrive();
    for(int k_pos=0;k_pos<K;k_pos+=BK)
    {
        if (k_pos+BK<K) {
            bar_consume[next_buf].wait(std::move(tok_done[next_buf]));
            move_data(k_pos+BK,next_buf);
            tok_copy[next_buf]=bar_copy[next_buf].arrive();
        }
        bar_copy[cur_buf].wait(std::move(tok_copy[cur_buf]));

        // warp id as y, tiled n as x
        for(int k_step=0;k_step<BK;k_step+=16){
            wmma::load_matrix_sync(a_frag, &smemA[cur_buf][tile*warp_id][k_step], lda);
            for(int n_step=0;n_step<8;++n_step)
            {
                wmma::load_matrix_sync(b_frag, &smemB[cur_buf][k_step][tile*n_step], ldb);
                wmma::mma_sync(c_frag[n_step], a_frag, b_frag, c_frag[n_step]);
            }
        }

        tok_done[cur_buf]=bar_consume[cur_buf].arrive();
        cur_buf ^= 1;
        next_buf ^= 1;
    }
    #pragma unroll
    for(int n=0;n<8;n++)
        wmma::store_matrix_sync(&C[(m_pos+warp_id*tile)*N+n_pos+n*tile], c_frag[n], N, wmma::mem_row_major);
}

// CPU reference GEMM: C_ref = A * B
static void gemm_cpu_ref(const half* A, const half* B, float* C, int M, int N, int K) {
    // Simple triple loop (correctness reference)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const half* a_row = A + i * K;
            for (int k = 0; k < K; ++k) {
                float af = __half2float(a_row[k]);
                float bf = __half2float(B[k*N+j]);
                acc += af*bf;
            }
            C[i * N + j] = acc;
        }
    }
}

static void fill_random(std::vector<half>& x, float lo = -1.0f, float hi = 1.0f, uint32_t seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : x) v = __float2half_rn(dist(rng));
}

static void compare(const std::vector<float>& ref,
                    const std::vector<float>& out,
                    float atol = 1e-3f,
                    float rtol = 1e-3f) {
    if (ref.size() != out.size()) {
        std::cerr << "Size mismatch: ref=" << ref.size() << " out=" << out.size() << "\n";
        std::exit(1);
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int max_i = -1;

    int bad = 0;
    for (int i = 0; i < (int)ref.size(); ++i) {
        float a = ref[i];
        float b = out[i];
        float abs_err = std::fabs(a - b);
        float rel_err = abs_err / (std::fabs(a) + 1e-8f);
        if (abs_err > max_abs) { max_abs = abs_err; max_i = i; }
        if (rel_err > max_rel) { max_rel = rel_err; }

        float tol = atol + rtol * std::fabs(a);
        if (abs_err > tol) bad++;
    }

    std::cout << "Compare:\n";
    std::cout << "  max_abs_err = " << max_abs << "\n";
    std::cout << "  max_rel_err = " << max_rel << "\n";
    std::cout << "  bad_count   = " << bad << " / " << ref.size() << "\n";

    if (max_i >= 0) {
        std::cout << "  worst_idx   = " << max_i
                  << " ref=" << ref[max_i]
                  << " out=" << out[max_i] << "\n";
    }

    if (bad == 0) std::cout << "  ✅ PASS\n";
    else          std::cout << "  ❌ FAIL\n";
}

int main() {
    // ---------------------------
    // 3) Problem size (fixed)
    // ---------------------------
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    const size_t bytesA = (size_t)M * K * sizeof(half);
    const size_t bytesB = (size_t)K * N * sizeof(half);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    std::cout << "GEMM: C[M,N] = A[M,K] * B[K,N]\n";
    std::cout << "M=" << M << " N=" << N << " K=" << K << " (half)\n";

    // ---------------------------
    // 2) Host allocations in main
    // ---------------------------
    std::vector<half> hA((size_t)M * K);
    std::vector<half> hB((size_t)K * N);
    std::vector<float> hC((size_t)M * N, 0.0f);      // output from your kernel
    std::vector<float> hC_ref((size_t)M * N, 0.0f);  // correct answer

    // Random init
    fill_random(hA, -1.0f, 1.0f, 49);
    fill_random(hB, -1.0f, 1.0f, 59);

    // ---------------------------
    // 5) CPU reference in main
    // ---------------------------
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        // gemm_cpu_ref(hA.data(), hB.data(), hC_ref.data(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "CPU reference time: " << ms << " ms\n";
    }

    // ---------------------------
    // 4) Move to GPU in main
    // ---------------------------
    half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC)); // make output deterministic even before you implement

    // ---------------------------
    // 7) Timing (GPU)
    // ---------------------------
    // You can choose your own tiling; this is a common baseline.
    dim3 block(256);//8 warps
    dim3 grid((N*M/(BN*BM)));

    // Warmup (optional)
    gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    std::cout << "Kernel time (1 run): " << kernel_ms << " ms\n";

    // (Optional) compute achieved GFLOPs
    // GEMM FLOPs ~= 2*M*N*K
    double gflops = (2.0 * M * N * K) / (kernel_ms * 1e6);
    std::cout << "Throughput: " << gflops << " GFLOP/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Copy back
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // ---------------------------
    // 5) Compare your output with correct answer
    // ---------------------------
    compare(hC_ref, hC, /*atol=*/1e-3f, /*rtol=*/1e-3f);

    // Cleanup
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}