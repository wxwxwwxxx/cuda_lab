#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>

namespace {

const char* yesNo(int v) {
    return v ? "Yes" : "No";
}

float bytesToMiB(size_t bytes) {
    return static_cast<float>(bytes) / (1024.0f * 1024.0f);
}

int getAttrOrNA(int dev, cudaDeviceAttr attr) {
    int value = -1;
    cudaError_t err = cudaDeviceGetAttribute(&value, attr, dev);
    if (err != cudaSuccess) {
        return -1;
    }
    return value;
}

// Rough mapping used for quick visibility. New architectures default to unknown.
int coresPerSM(int major, int minor) {
    struct SMToCores {
        int sm;
        int cores;
    };
    static const SMToCores map[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
        {0x50, 128}, {0x52, 128}, {0x53, 128},
        {0x60, 64},  {0x61, 128}, {0x62, 128},
        {0x70, 64},  {0x72, 64},  {0x75, 64},
        {0x80, 64},  {0x86, 128}, {0x87, 128}, {0x89, 128},
        {0x90, 128}
    };
    const int sm = (major << 4) + minor;
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); ++i) {
        if (map[i].sm == sm) {
            return map[i].cores;
        }
    }
    return -1;
}

void printUuid(const cudaUUID_t& uuid) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(uuid.bytes);
    // UUID format: 8-4-4-4-12
    printf(
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x%02x%02x%02x%02x",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

void printClockInfo(int dev) {
    const int coreClockKHz = getAttrOrNA(dev, cudaDevAttrClockRate);
    const int memClockKHz = getAttrOrNA(dev, cudaDevAttrMemoryClockRate);
    const int busWidthBits = getAttrOrNA(dev, cudaDevAttrGlobalMemoryBusWidth);

    if (coreClockKHz >= 0) {
        printf("  GPU 时钟频率 (Core Clock):                     %.3f GHz (%d KHz)\n",
               coreClockKHz * 1e-6f, coreClockKHz);
    } else {
        printf("  GPU 时钟频率 (Core Clock):                     N/A\n");
    }

    if (memClockKHz >= 0) {
        printf("  显存时钟频率 (Memory Clock):                   %.3f GHz (%d KHz)\n",
               memClockKHz * 1e-6f, memClockKHz);
    } else {
        printf("  显存时钟频率 (Memory Clock):                   N/A\n");
    }

    if (busWidthBits >= 0) {
        printf("  显存总线宽度 (Memory Bus Width):               %d bits\n", busWidthBits);
    } else {
        printf("  显存总线宽度 (Memory Bus Width):               N/A\n");
    }
}

}  // namespace

int main() {
    int runtimeVersion = 0;
    int driverVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);

    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    printf("CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);

    int deviceCount = 0;
    const cudaError_t errorId = cudaGetDeviceCount(&deviceCount);
    if (errorId != cudaSuccess) {
        printf("获取设备数量失败，错误码: %d\n-> %s\n",
               static_cast<int>(errorId), cudaGetErrorString(errorId));
        return -1;
    }

    if (deviceCount == 0) {
        printf("当前系统中没有检测到支持 CUDA 的设备。\n");
        return 0;
    }

    printf("检测到 %d 个支持 CUDA 的设备。\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop{};
        cudaError_t propErr = cudaGetDeviceProperties(&prop, dev);
        if (propErr != cudaSuccess) {
            printf("读取设备 %d 信息失败: %s\n", dev, cudaGetErrorString(propErr));
            continue;
        }

        printf("\n============================================================\n");
        printf("设备 %d: \"%s\"\n", dev, prop.name);
        printf("============================================================\n");

        printf("  UUID:                                       ");
        printUuid(prop.uuid);
        printf("\n");

        const int ccMajor = prop.major;
        const int ccMinor = prop.minor;
        const int smCount = prop.multiProcessorCount;
        const int cpsm = coresPerSM(ccMajor, ccMinor);
        printf("  计算能力 (Compute Capability):               %d.%d\n", ccMajor, ccMinor);
        printf("  流多处理器数量 (SM Count):                   %d\n", smCount);
        if (cpsm > 0) {
            printf("  估算 CUDA Cores/SM:                          %d\n", cpsm);
            printf("  估算总 CUDA Cores:                           %d\n", cpsm * smCount);
        } else {
            printf("  估算 CUDA Cores/SM:                          Unknown (新架构或未映射)\n");
        }

        printf("  全局内存总量 (Global Memory):                %.2f MiB (%llu bytes)\n",
               bytesToMiB(prop.totalGlobalMem),
               static_cast<unsigned long long>(prop.totalGlobalMem));
        printf("  常量内存总量 (Constant Memory):              %zu bytes\n", prop.totalConstMem);
        printf("  L2 Cache 大小:                               %d bytes\n", prop.l2CacheSize);
        printf("  每个 Block 共享内存上限:                     %zu bytes\n", prop.sharedMemPerBlock);
        printf("  每个 Block 可 opt-in 共享内存上限:           %zu bytes\n", prop.sharedMemPerBlockOptin);
        printf("  每个 SM 共享内存上限:                        %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  每个 Block 寄存器上限:                       %d\n", prop.regsPerBlock);
        printf("  每个 SM 寄存器上限:                          %d\n", prop.regsPerMultiprocessor);

        printf("  Warp Size:                                   %d\n", prop.warpSize);
        printf("  每个 Block 最大线程数:                       %d\n", prop.maxThreadsPerBlock);
        printf("  每个 SM 最大线程数:                          %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  每个 SM 最大 Block 数:                       %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Block 最大维度 (x, y, z):                    (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Grid 最大维度 (x, y, z):                     (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        printClockInfo(dev);

        printf("  并发 Kernel 支持:                            %s\n", yesNo(prop.concurrentKernels));
        printf("  ECC 支持:                                    %s\n", yesNo(prop.ECCEnabled));
        printf("  Unified Addressing:                          %s\n", yesNo(prop.unifiedAddressing));
        printf("  Managed Memory:                              %s\n", yesNo(prop.managedMemory));
        printf("  主机原生原子支持 (Host Native Atomic):       %s\n", yesNo(prop.hostNativeAtomicSupported));
        printf("  Cooperative Launch:                          %s\n", yesNo(prop.cooperativeLaunch));
        printf("  Stream Priorities:                           %s\n", yesNo(prop.streamPrioritiesSupported));
        printf("  全局内存 L1 Cache:                           %s\n", yesNo(prop.globalL1CacheSupported));
        printf("  本地内存 L1 Cache:                           %s\n", yesNo(prop.localL1CacheSupported));
        printf("  集成 GPU (Integrated):                       %s\n", yesNo(prop.integrated));

        printf("  PCI: domain:bus:device                       %d:%d:%d\n",
               prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        printf("  Async Engine 数量:                           %d\n", prop.asyncEngineCount);
        printf("  是否多 GPU 板卡:                             %s\n", yesNo(prop.isMultiGpuBoard));
        printf("  多 GPU 板卡组 ID:                            %d\n", prop.multiGpuBoardGroupID);
    }

    return 0;
}
