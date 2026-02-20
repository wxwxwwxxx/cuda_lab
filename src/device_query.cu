#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("获取设备数量失败，错误码: %d\n-> %s\n", 
               (int)error_id, cudaGetErrorString(error_id));
        return -1;
    }

    if (deviceCount == 0) {
        printf("当前系统中没有检测到支持 CUDA 的设备。\n");
        return 0;
    } else {
        printf("检测到 %d 个支持 CUDA 的设备。\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\n==================================================\n");
        printf("设备 %d: \"%s\"\n", dev, deviceProp.name);
        printf("==================================================\n");
        
        // 计算能力 (Compute Capability)
        printf("  计算能力 (Compute Capability):                 %d.%d\n", deviceProp.major, deviceProp.minor);
        
        // 显存信息
        printf("  全局内存总量 (Global Memory):                  %.2f MB (%llu bytes)\n", 
               (float)deviceProp.totalGlobalMem / (1024 * 1024), (unsigned long long)deviceProp.totalGlobalMem);
        printf("  常量内存总量 (Constant Memory):                %zu bytes\n", deviceProp.totalConstMem);
        printf("  每个 Block 最大共享内存 (Shared Mem per Block): %zu bytes\n", deviceProp.sharedMemPerBlock);
        
        // 处理器与线程信息
        printf("  流多处理器数量 (SM Count):                     %d\n", deviceProp.multiProcessorCount);
        printf("  每个 Block 可用最大寄存器数量 (Regs per Block): %d\n", deviceProp.regsPerBlock);
        printf("  线程束大小 (Warp Size):                        %d\n", deviceProp.warpSize);
        printf("  每个 SM 最大线程数 (Max Threads per SM):       %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  每个 Block 最大线程数 (Max Threads per Block): %d\n", deviceProp.maxThreadsPerBlock);
        
        // 维度限制
        printf("  Block 的最大维度 (x, y, z):                    (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Grid 的最大维度 (x, y, z):                     (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
               
        // 频率信息
        printf("  GPU 时钟频率 (Clock Rate):                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        printf("  显存时钟频率 (Memory Clock Rate):              %.2f GHz\n", deviceProp.memoryClockRate * 1e-6f);
        printf("  显存总线宽度 (Memory Bus Width):               %d bits\n", deviceProp.memoryBusWidth);
    }

    return 0;
}