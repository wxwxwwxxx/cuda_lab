#include <cuda_runtime.h>

#include <cutlass/conv/kernel/default_conv3d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/tensor.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
              cudaGetErrorString(_e));                                            \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutInputA = cutlass::layout::TensorNDHWC;
using LayoutInputB = cutlass::layout::TensorNDHWC;  // KTRSC packed like NDHWC
using LayoutOutput = cutlass::layout::TensorNDHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 4;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;
static cutlass::conv::StrideSupport const StrideSupport = cutlass::conv::StrideSupport::kStrided;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute>;

using Conv3dFpropKernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm,
    StrideSupport>::Kernel;

using Conv3dOperation = cutlass::conv::device::ImplicitGemmConvolution<Conv3dFpropKernel>;

struct Options {
  int n = 1;
  int d = 8;
  int h = 16;
  int w = 16;
  int c = 32;

  int k = 32;
  int t = 3;
  int r = 3;
  int s = 3;

  int pad_d = 1;
  int pad_h = 1;
  int pad_w = 1;

  int stride_d = 1;
  int stride_h = 1;
  int stride_w = 1;

  int dilation_d = 1;
  int dilation_h = 1;
  int dilation_w = 1;

  float alpha = 1.0f;
  float beta = 0.0f;
  int warmup = 5;
  int iters = 100;
  bool verify = true;
};

static void print_usage(char const* exe) {
  std::cout
      << "Usage: " << exe << " [options]\n"
      << "Options:\n"
      << "  --n=INT --d=INT --h=INT --w=INT --c=INT      input NDHWC\n"
      << "  --k=INT --t=INT --r=INT --s=INT              filter KTRSC\n"
      << "  --pad_d=INT --pad_h=INT --pad_w=INT\n"
      << "  --stride_d=INT --stride_h=INT --stride_w=INT\n"
      << "  --dilation_d=INT --dilation_h=INT --dilation_w=INT\n"
      << "  --alpha=FLOAT --beta=FLOAT\n"
      << "  --warmup=INT --iters=INT\n"
      << "  --verify=0|1\n"
      << "  --help\n";
}

static bool parse_int(char const* s, int& out) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!s || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX) {
    return false;
  }
  out = static_cast<int>(v);
  return true;
}

static bool parse_float(char const* s, float& out) {
  char* end = nullptr;
  float v = std::strtof(s, &end);
  if (!s || end == s || *end != '\0') {
    return false;
  }
  out = v;
  return true;
}

static bool parse_kv_int(char const* arg, char const* name, int& out) {
  std::string prefix = std::string("--") + name + "=";
  size_t n = prefix.size();
  if (std::strncmp(arg, prefix.c_str(), n) != 0) {
    return false;
  }
  if (!parse_int(arg + n, out)) {
    std::cerr << "Invalid integer for --" << name << ": " << (arg + n) << "\n";
    std::exit(1);
  }
  return true;
}

static bool parse_kv_float(char const* arg, char const* name, float& out) {
  std::string prefix = std::string("--") + name + "=";
  size_t n = prefix.size();
  if (std::strncmp(arg, prefix.c_str(), n) != 0) {
    return false;
  }
  if (!parse_float(arg + n, out)) {
    std::cerr << "Invalid float for --" << name << ": " << (arg + n) << "\n";
    std::exit(1);
  }
  return true;
}

static bool parse_options(int argc, char const** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    char const* arg = argv[i];

    if (std::strcmp(arg, "--help") == 0) {
      print_usage(argv[0]);
      return false;
    }
    if (std::strcmp(arg, "--no-verify") == 0) {
      opt.verify = false;
      continue;
    }

    int verify_i = 0;
    if (parse_kv_int(arg, "verify", verify_i)) {
      opt.verify = (verify_i != 0);
      continue;
    }

    if (parse_kv_int(arg, "n", opt.n) || parse_kv_int(arg, "d", opt.d) ||
        parse_kv_int(arg, "h", opt.h) || parse_kv_int(arg, "w", opt.w) ||
        parse_kv_int(arg, "c", opt.c) || parse_kv_int(arg, "k", opt.k) ||
        parse_kv_int(arg, "t", opt.t) || parse_kv_int(arg, "r", opt.r) ||
        parse_kv_int(arg, "s", opt.s) || parse_kv_int(arg, "pad_d", opt.pad_d) ||
        parse_kv_int(arg, "pad_h", opt.pad_h) || parse_kv_int(arg, "pad_w", opt.pad_w) ||
        parse_kv_int(arg, "stride_d", opt.stride_d) || parse_kv_int(arg, "stride_h", opt.stride_h) ||
        parse_kv_int(arg, "stride_w", opt.stride_w) || parse_kv_int(arg, "dilation_d", opt.dilation_d) ||
        parse_kv_int(arg, "dilation_h", opt.dilation_h) || parse_kv_int(arg, "dilation_w", opt.dilation_w) ||
        parse_kv_int(arg, "warmup", opt.warmup) || parse_kv_int(arg, "iters", opt.iters)) {
      continue;
    }

    if (parse_kv_float(arg, "alpha", opt.alpha) || parse_kv_float(arg, "beta", opt.beta)) {
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    print_usage(argv[0]);
    return false;
  }
  return true;
}

static int output_dim(int in, int pad, int kernel, int stride, int dilation) {
  int effective = (kernel - 1) * dilation + 1;
  return (in + 2 * pad - effective) / stride + 1;
}

static int64_t idx_ndhwc(int n, int d, int h, int w, int c, int D, int H, int W, int C) {
  return (((((int64_t)n * D + d) * H + h) * W + w) * C + c);
}

static int64_t idx_ktrsc(int k, int t, int r, int s, int c, int T, int R, int S, int C) {
  return (((((int64_t)k * T + t) * R + r) * S + s) * C + c);
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

static void conv3d_cpu_ref(
    std::vector<cutlass::half_t> const& activation,
    std::vector<cutlass::half_t> const& filter,
    std::vector<cutlass::half_t> const& tensor_c,
    std::vector<cutlass::half_t>& output,
    Options const& opt,
    int z_out,
    int p_out,
    int q_out) {
  for (int n = 0; n < opt.n; ++n) {
    for (int z = 0; z < z_out; ++z) {
      for (int p = 0; p < p_out; ++p) {
        for (int q = 0; q < q_out; ++q) {
          for (int k = 0; k < opt.k; ++k) {
            float acc = 0.0f;
            for (int t = 0; t < opt.t; ++t) {
              int in_d = z * opt.stride_d - opt.pad_d + t * opt.dilation_d;
              if (in_d < 0 || in_d >= opt.d) {
                continue;
              }
              for (int r = 0; r < opt.r; ++r) {
                int in_h = p * opt.stride_h - opt.pad_h + r * opt.dilation_h;
                if (in_h < 0 || in_h >= opt.h) {
                  continue;
                }
                for (int s = 0; s < opt.s; ++s) {
                  int in_w = q * opt.stride_w - opt.pad_w + s * opt.dilation_w;
                  if (in_w < 0 || in_w >= opt.w) {
                    continue;
                  }
                  for (int c = 0; c < opt.c; ++c) {
                    int64_t a_idx = idx_ndhwc(n, in_d, in_h, in_w, c, opt.d, opt.h, opt.w, opt.c);
                    int64_t b_idx = idx_ktrsc(k, t, r, s, c, opt.t, opt.r, opt.s, opt.c);
                    acc += float(activation[a_idx]) * float(filter[b_idx]);
                  }
                }
              }
            }

            int64_t out_idx = idx_ndhwc(n, z, p, q, k, z_out, p_out, q_out, opt.k);
            float c_val = float(tensor_c[out_idx]);
            output[out_idx] = cutlass::half_t(opt.alpha * acc + opt.beta * c_val);
          }
        }
      }
    }
  }
}

static bool compare_half_tensors(std::vector<cutlass::half_t> const& ref,
                                 std::vector<cutlass::half_t> const& out,
                                 float atol = 0.1f,
                                 float rtol = 0.1f) {
  if (ref.size() != out.size()) {
    std::cerr << "Size mismatch: ref=" << ref.size() << " out=" << out.size() << "\n";
    return false;
  }

  float max_abs = 0.0f;
  float max_rel = 0.0f;
  int64_t max_i = -1;
  int64_t bad = 0;

  for (int64_t i = 0; i < static_cast<int64_t>(ref.size()); ++i) {
    float a = float(ref[i]);
    float b = float(out[i]);
    float abs_err = std::fabs(a - b);
    float rel_err = abs_err / (std::fabs(a) + 1e-8f);

    if (abs_err > max_abs) {
      max_abs = abs_err;
      max_i = i;
    }
    if (rel_err > max_rel) {
      max_rel = rel_err;
    }

    float tol = atol + rtol * std::fabs(a);
    if (abs_err > tol) {
      ++bad;
    }
  }

  std::cout << "Compare:\n";
  std::cout << "  max_abs_err = " << max_abs << "\n";
  std::cout << "  max_rel_err = " << max_rel << "\n";
  std::cout << "  bad_count   = " << bad << " / " << ref.size() << "\n";
  if (max_i >= 0) {
    std::cout << "  worst_idx   = " << max_i << " ref=" << float(ref[max_i]) << " out=" << float(out[max_i]) << "\n";
  }
  std::cout << (bad == 0 ? "  PASS\n" : "  FAIL\n");

  return bad == 0;
}

int main(int argc, char const** argv) {
  Options opt;
  if (!parse_options(argc, argv, opt)) {
    return 0;
  }

  int device_count = 0;
  cudaError_t device_query = cudaGetDeviceCount(&device_count);
  if (device_query != cudaSuccess || device_count <= 0) {
    std::cerr << "No usable CUDA device found: " << cudaGetErrorString(device_query) << "\n";
    return 0;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major < 8) {
    std::cerr << "This sample requires SM80+ (Ampere or newer).\n";
    return 0;
  }

  if (opt.n <= 0 || opt.d <= 0 || opt.h <= 0 || opt.w <= 0 || opt.c <= 0 || opt.k <= 0 ||
      opt.t <= 0 || opt.r <= 0 || opt.s <= 0 || opt.stride_d <= 0 || opt.stride_h <= 0 ||
      opt.stride_w <= 0 || opt.dilation_d <= 0 || opt.dilation_h <= 0 || opt.dilation_w <= 0 ||
      opt.warmup < 0 || opt.iters <= 0) {
    std::cerr << "Invalid non-positive option.\n";
    return 1;
  }

  // CUTLASS Tensor Core half kernels generally require channel dims aligned to 8.
  if ((opt.c % 8) != 0 || (opt.k % 8) != 0) {
    std::cerr << "Channel alignment error: require C % 8 == 0 and K % 8 == 0.\n";
    return 1;
  }

  int z_out = output_dim(opt.d, opt.pad_d, opt.t, opt.stride_d, opt.dilation_d);
  int p_out = output_dim(opt.h, opt.pad_h, opt.r, opt.stride_h, opt.dilation_h);
  int q_out = output_dim(opt.w, opt.pad_w, opt.s, opt.stride_w, opt.dilation_w);
  if (z_out <= 0 || p_out <= 0 || q_out <= 0) {
    std::cerr << "Invalid output size: Z=" << z_out << " P=" << p_out << " Q=" << q_out << "\n";
    return 1;
  }

  cutlass::Tensor5DCoord input_size(opt.n, opt.d, opt.h, opt.w, opt.c);      // NDHWC
  cutlass::Tensor5DCoord filter_size(opt.k, opt.t, opt.r, opt.s, opt.c);      // KTRSC
  cutlass::Tensor5DCoord output_size(opt.n, z_out, p_out, q_out, opt.k);      // NZPQK
  cutlass::Coord<3> padding = cutlass::make_Coord(opt.pad_d, opt.pad_h, opt.pad_w);
  cutlass::Coord<3> stride = cutlass::make_Coord(opt.stride_d, opt.stride_h, opt.stride_w);
  cutlass::Coord<3> dilation = cutlass::make_Coord(opt.dilation_d, opt.dilation_h, opt.dilation_w);

  cutlass::conv::Conv3dProblemSize problem_size(
      input_size,
      filter_size,
      padding,
      stride,
      dilation,
      output_size,
      cutlass::conv::Mode::kCrossCorrelation,
      1);

  int64_t act_elems = static_cast<int64_t>(opt.n) * opt.d * opt.h * opt.w * opt.c;
  int64_t flt_elems = static_cast<int64_t>(opt.k) * opt.t * opt.r * opt.s * opt.c;
  int64_t out_elems = static_cast<int64_t>(opt.n) * z_out * p_out * q_out * opt.k;

  size_t bytes_a = static_cast<size_t>(act_elems) * sizeof(ElementInputA);
  size_t bytes_b = static_cast<size_t>(flt_elems) * sizeof(ElementInputB);
  size_t bytes_c = static_cast<size_t>(out_elems) * sizeof(ElementOutput);

  std::cout << "Conv3D Fprop (CUTLASS, half)\n";
  std::cout << "Input  NDHWC: (" << opt.n << ", " << opt.d << ", " << opt.h << ", " << opt.w << ", " << opt.c << ")\n";
  std::cout << "Filter KTRSC: (" << opt.k << ", " << opt.t << ", " << opt.r << ", " << opt.s << ", " << opt.c << ")\n";
  std::cout << "Output NZPQK: (" << opt.n << ", " << z_out << ", " << p_out << ", " << q_out << ", " << opt.k << ")\n";
  std::cout << "alpha=" << opt.alpha << " beta=" << opt.beta
            << " warmup=" << opt.warmup << " iters=" << opt.iters
            << " verify=" << (opt.verify ? "true" : "false") << "\n";

  std::vector<ElementInputA> h_a(static_cast<size_t>(act_elems));
  std::vector<ElementInputB> h_b(static_cast<size_t>(flt_elems));
  std::vector<ElementOutput> h_c(static_cast<size_t>(out_elems));
  std::vector<ElementOutput> h_d(static_cast<size_t>(out_elems));
  std::vector<ElementOutput> h_ref(static_cast<size_t>(out_elems));

  fill_random(h_a, -1.0f, 1.0f, 2026);
  fill_random(h_b, -1.0f, 1.0f, 2027);
  fill_random(h_c, -1.0f, 1.0f, 2028);

  ElementInputA* d_a = nullptr;
  ElementInputB* d_b = nullptr;
  ElementOutput* d_c = nullptr;
  ElementOutput* d_d = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
  CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
  CUDA_CHECK(cudaMalloc(&d_c, bytes_c));
  CUDA_CHECK(cudaMalloc(&d_d, bytes_c));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), bytes_c, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_d, 0, bytes_c));

  LayoutInputA layout_a = LayoutInputA::packed(input_size);
  LayoutInputB layout_b = LayoutInputB::packed(filter_size);
  LayoutOutput layout_c = LayoutOutput::packed(output_size);

  cutlass::TensorRef<ElementInputA, LayoutInputA> ref_a(d_a, layout_a);
  cutlass::TensorRef<ElementInputB, LayoutInputB> ref_b(d_b, layout_b);
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_c(d_c, layout_c);
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_d(d_d, layout_c);

  typename Conv3dOperation::Arguments arguments{
      problem_size,
      ref_a,
      ref_b,
      ref_c,
      ref_d,
      {opt.alpha, opt.beta}};

  Conv3dOperation conv3d_op;

  cutlass::Status status = conv3d_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "can_implement failed: " << cutlassGetStatusString(status) << "\n";
    return 1;
  }

  size_t workspace_size = conv3d_op.get_workspace_size(arguments);
  uint8_t* workspace = nullptr;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  status = conv3d_op.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "initialize failed: " << cutlassGetStatusString(status) << "\n";
    return 1;
  }

  for (int i = 0; i < opt.warmup; ++i) {
    status = conv3d_op();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "warmup failed: " << cutlassGetStatusString(status) << "\n";
      return 1;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < opt.iters; ++i) {
    status = conv3d_op();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "run failed: " << cutlassGetStatusString(status) << "\n";
      return 1;
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / static_cast<float>(opt.iters);

  double flops = 2.0 * static_cast<double>(opt.n) * z_out * p_out * q_out *
                 opt.k * opt.t * opt.r * opt.s * opt.c;
  double gflops = flops / (static_cast<double>(avg_ms) * 1.0e6);

  std::cout << "Kernel avg time: " << avg_ms << " ms\n";
  std::cout << "Throughput: " << gflops << " GFLOP/s\n";

  CUDA_CHECK(cudaMemcpy(h_d.data(), d_d, bytes_c, cudaMemcpyDeviceToHost));

  bool pass = true;
  if (opt.verify) {
    auto t0 = std::chrono::high_resolution_clock::now();
    conv3d_cpu_ref(h_a, h_b, h_c, h_ref, opt, z_out, p_out, q_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU reference time: " << cpu_ms << " ms\n";
    pass = compare_half_tensors(h_ref, h_d);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_d));
  if (workspace) {
    CUDA_CHECK(cudaFree(workspace));
  }

  return pass ? 0 : 1;
}
