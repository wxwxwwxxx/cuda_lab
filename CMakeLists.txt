cmake_minimum_required(VERSION 3.20)

project(cuda_lab LANGUAGES CXX CUDA)

# =========================
# 基本配置
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 自动匹配本机 GPU（要求 CMake + NVCC 版本支持；若踩坑可改成写死 86）
set(CMAKE_CUDA_ARCHITECTURES native)
# 也可以写死：set(CMAKE_CUDA_ARCHITECTURES 80 86 90)

# 你项目里很多 .cu 都是单文件小实验，全局开 RDC 没问题；也可改成对特定 target 开。
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Debug/RelWithDebInfo flags
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-lineinfo -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-fno-omit-frame-pointer")

# =========================
# 查找 CUDA Toolkit（必需）
# =========================
find_package(CUDAToolkit REQUIRED)

# =========================
# CUTLASS (optional)
# =========================
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/thirdparty/cutlass)
set(CUTLASS_AVAILABLE OFF)
if (EXISTS ${CUTLASS_DIR}/include/cutlass/cutlass.h)
    set(CUTLASS_AVAILABLE ON)
else()
    message(STATUS "CUTLASS not found at ${CUTLASS_DIR}; run: git submodule update --init --recursive")
endif()

# =========================
# NVSHMEM (optional)
# =========================
# QUIET: 没装 nvshmem 也能正常配置，只是跳过 nvshmem 示例
find_package(NVSHMEM QUIET)

set(NVSHMEM_AVAILABLE OFF)
if (NVSHMEM_FOUND)
    set(NVSHMEM_AVAILABLE ON)
    message(STATUS "NVSHMEM found: enabling nvSHMEM examples")
else()
    message(STATUS "NVSHMEM not found: nvSHMEM examples will be skipped")
endif()

# =========================
# 搜索所有 cu 文件
# =========================
file(GLOB CUDA_SOURCES
        ${PROJECT_SOURCE_DIR}/src/*.cu
)

# 如果没有 CUTLASS，就移除依赖 CUTLASS 的示例
if (NOT CUTLASS_AVAILABLE)
    list(REMOVE_ITEM CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/gemm_cutlass.cu)
endif()

# 如果没有 NVSHMEM，就移除 nvshmem 示例
if (NOT NVSHMEM_AVAILABLE)
    list(REMOVE_ITEM CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/nvshmem_ring.cu)
endif()

# =========================
# 为每个 cu 生成一个可执行文件
# =========================
foreach(src ${CUDA_SOURCES})
    get_filename_component(exec_name ${src} NAME_WE)
    add_executable(${exec_name} ${src})

    target_link_libraries(${exec_name}
            PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
    )

    # 你的实验代码统一打开 fast math（如果你不希望全局生效，可以挪到某些 target）
    target_compile_options(${exec_name}
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )

    # -------------------------
    # CUTLASS demo
    # -------------------------
    if (exec_name STREQUAL "gemm_cutlass")
        target_include_directories(${exec_name}
                PRIVATE
                ${CUTLASS_DIR}/include
                ${CUTLASS_DIR}/tools/util/include
        )
    endif()

    # -------------------------
    # cuBLAS demo
    # -------------------------
    if (exec_name STREQUAL "gemm_cublas")
        target_link_libraries(${exec_name}
                PRIVATE
                CUDA::cublas
        )
    endif()

    # -------------------------
    # nvSHMEM demo
    # -------------------------
    if (exec_name STREQUAL "nvshmem_ring")
        if (NVSHMEM_AVAILABLE)
            target_link_libraries(${exec_name}
                    PRIVATE
                    nvshmem::nvshmem_host
                    nvshmem::nvshmem_device
            )

            # 强烈建议对 nvshmem target 显式打开这些属性（即便你已全局打开 RDC）
            set_target_properties(${exec_name} PROPERTIES
                    CUDA_SEPARABLE_COMPILATION ON
                    CUDA_RESOLVE_DEVICE_SYMBOLS ON
            )
        else()
            message(FATAL_ERROR "nvshmem_ring is in sources but NVSHMEM is not available. This should not happen.")
        endif()
    endif()

endforeach()