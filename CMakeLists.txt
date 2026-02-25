cmake_minimum_required(VERSION 3.20)

project(cuda_lab LANGUAGES CXX CUDA)

# =========================
# 基本配置
# =========================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)   # 自动匹配本机GPU
# 也可以写死：set(CMAKE_CUDA_ARCHITECTURES 80 86 90)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Debug 时打开 device 调试
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-lineinfo -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-fno-omit-frame-pointer")
# =========================
# 查找 CUDA
# =========================
find_package(CUDAToolkit REQUIRED)


set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/thirdparty/cutlass)
set(CUTLASS_AVAILABLE OFF)
if (EXISTS ${CUTLASS_DIR}/include/cutlass/cutlass.h)
    set(CUTLASS_AVAILABLE ON)
endif()


# =========================
# 搜索所有 cu 文件
# =========================
file(GLOB CUDA_SOURCES
        ${PROJECT_SOURCE_DIR}/src/*.cu
)

if (NOT CUTLASS_AVAILABLE)
    list(REMOVE_ITEM CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/gemm_cutlass.cu)
    message(STATUS "CUTLASS not found at ${CUTLASS_DIR}; run: git submodule update --init --recursive")
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

    target_compile_options(${exec_name}
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )

    if (exec_name STREQUAL "gemm_cutlass")
        target_include_directories(${exec_name}
                PRIVATE
                ${CUTLASS_DIR}/include
                ${CUTLASS_DIR}/tools/util/include
        )
    endif()

    if (exec_name STREQUAL "gemm_cublas")
        target_link_libraries(${exec_name}
                PRIVATE
                CUDA::cublas
        )
    endif()

endforeach()
