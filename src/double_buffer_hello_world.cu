#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda/barrier> // 必须保留，这是 libcu++ 的核心

// 这是一个纯 CUDA C++ 的例子，没有任何 Cooperative Groups 的影子

// Kernel: 演示 cuda::barrier 和 cuda::memcpy_async (无 CG)
__global__ void async_copy_kernel_no_cg(const float* __restrict__ input, float* __restrict__ output, int n) {
    // 1. 定义 Shared Memory
    // 假设 Block 大小是 128
    __shared__ alignas(16) float s_buffer[128];

    // 2. 定义 Barrier
    // thread_scope_block: 作用域为当前 Block
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // 3. 初始化 Barrier (替换掉了 CG)
    // 使用原始的 threadIdx.x 判断主线程
    // 使用原始的 blockDim.x 获取线程总数
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }

    // 4. 必须同步！(替换掉了 block.sync())
    // 确保 bar 初始化完成，所有线程才能往下走
    __syncthreads();

    // 计算全局索引
    int tid = threadIdx.x; // 替换 block.thread_rank()
    int global_idx = blockIdx.x * blockDim.x + tid;

    if (global_idx < n) {
        // -----------------------------------------------------------
        // 步骤 A: 异步拷贝
        // -----------------------------------------------------------
        // cuda::memcpy_async 不需要 CG 对象也能工作
        // 它只需要：目标地址、源地址、大小、以及一个 barrier 对象
        cuda::memcpy_async(
                &s_buffer[tid],       // shared memory 地址
                &input[global_idx],   // global memory 地址
                sizeof(float),        // 搬运字节数
                bar                   // 绑定 barrier
        );

        // -----------------------------------------------------------
        // 步骤 B: 重叠计算 (Overlap)
        // -----------------------------------------------------------
        // 这里依然可以做不依赖 s_buffer 的计算
        float independent_val = global_idx * 0.1f;

        // -----------------------------------------------------------
        // 步骤 C: 等待 (Arrive and Wait)
        // -----------------------------------------------------------
        // 这一步没有任何变化，barrier 是个独立对象
        // 它会阻塞当前线程，直到所有线程都到达且拷贝完成
        bar.arrive_and_wait();

        // -----------------------------------------------------------
        // 步骤 D: 读取 Shared Memory 并计算
        // -----------------------------------------------------------
        float cached_val = s_buffer[tid];
        float result = cached_val * 2.0f + independent_val;

        // -----------------------------------------------------------
        // 步骤 E: 写回
        // -----------------------------------------------------------
        output[global_idx] = result;
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    // Host 内存分配与初始化
    std::vector<float> h_in(N, 1.0f);
    std::vector<float> h_out(N);

    // Device 内存分配
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // 启动配置
    int threads = 128;
    int blocks = (N + threads - 1) / threads;

    std::cout << "Launching NO-CG kernel..." << std::endl;

    // 启动 Kernel
    async_copy_kernel_no_cg<<<blocks, threads>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();

    // 结果验证
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    if (abs(h_out[0] - 2.0f) < 1e-4) { // 简单检查第一个元素: 1.0*2.0 + 0 = 2.0
        std::cout << "Test Passed: Logic works without Cooperative Groups!" << std::endl;
    } else {
        std::cout << "Test Failed: Value is " << h_out[0] << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}