/**
 * @file   GpuRenderer.cc
 *
 * @brief  CUDA / HIP implementation of the ray tracer.
 *
 * Compiled as CUDA (when MUEYE_ENABLE_CUDA) or HIP (when MUEYE_ENABLE_HIP); the
 * CMake target marks this file with LANGUAGE CUDA/HIP. It reuses the exact
 * ray-march kernel from render_core.hh (whose trace_ray() is __host__ __device__)
 * and muGrid's portability macros from memory/gpu_runtime.hh — so the GPU image
 * matches the CPU/Metal backends.
 *
 * NOTE: this file is only built on a CUDA/HIP toolchain; it is not compiled on
 * the (CPU-only macOS) development machine.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "render/GpuRenderer.hh"

// muGrid's gpu_runtime.hh keys off MUGRID_ENABLE_CUDA / MUGRID_ENABLE_HIP and
// pulls in <cuda_runtime.h> / <hip/hip_runtime.h>. The muEye CMake defines the
// matching MUGRID_* macro alongside MUEYE_* when a GPU backend is enabled.
#include "memory/gpu_runtime.hh"
#include "render/render_core.hh"

namespace mueye {

namespace {

__global__ void render_kernel(const float *vol, const Vec4 *lut, RenderParams p,
                              Camera cam, unsigned char *out, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float u = (x + 0.5f) / w;
  float v = (y + 0.5f) / h;
  Vec4 c = trace_ray(vol, lut, p, cam, u, v);
  std::size_t idx = (static_cast<std::size_t>(y) * w + x) * 4;
  out[idx + 0] = static_cast<unsigned char>(clampf(c.x, 0.f, 1.f) * 255.f + 0.5f);
  out[idx + 1] = static_cast<unsigned char>(clampf(c.y, 0.f, 1.f) * 255.f + 0.5f);
  out[idx + 2] = static_cast<unsigned char>(clampf(c.z, 0.f, 1.f) * 255.f + 0.5f);
  out[idx + 3] = static_cast<unsigned char>(clampf(c.w, 0.f, 1.f) * 255.f + 0.5f);
}

}  // namespace

bool GpuRenderer::available() {
  int n = 0;
#if defined(MUGRID_ENABLE_CUDA)
  if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
#elif defined(MUGRID_ENABLE_HIP)
  if (hipGetDeviceCount(&n) != hipSuccess) return false;
#endif
  return n > 0;
}

GpuRenderer::GpuRenderer() = default;

GpuRenderer::~GpuRenderer() {
  if (d_volume_) GPU_FREE(d_volume_);
  if (d_lut_) GPU_FREE(d_lut_);
  if (d_output_) GPU_FREE(d_output_);
}

const char *GpuRenderer::name() const {
#if defined(MUGRID_ENABLE_HIP)
  return "HIP (GPU)";
#else
  return "CUDA (GPU)";
#endif
}

Backend GpuRenderer::backend() const {
#if defined(MUGRID_ENABLE_HIP)
  return Backend::HIP;
#else
  return Backend::CUDA;
#endif
}

void GpuRenderer::set_volume(const float *data, int nx, int ny, int nz) {
  if (data == nullptr) return;
  if (d_volume_) {
    GPU_FREE(d_volume_);
    d_volume_ = nullptr;
  }
  nx_ = nx;
  ny_ = ny;
  nz_ = nz;
  std::size_t bytes = static_cast<std::size_t>(nx) * ny * nz * sizeof(float);
  GPU_MALLOC(reinterpret_cast<void **>(&d_volume_), bytes);
  GPU_MEMCPY_H2D(d_volume_, data, bytes);
}

void GpuRenderer::set_transfer_function(const Vec4 *lut, int n) {
  if (lut == nullptr || n <= 0) return;
  if (d_lut_) {
    GPU_FREE(d_lut_);
    d_lut_ = nullptr;
  }
  std::size_t bytes = static_cast<std::size_t>(n) * sizeof(Vec4);
  GPU_MALLOC(reinterpret_cast<void **>(&d_lut_), bytes);
  GPU_MEMCPY_H2D(d_lut_, lut, bytes);
}

void GpuRenderer::render(const RenderParams &params, const Camera &camera,
                         Framebuffer &fb) {
  const int w = fb.width, h = fb.height;
  if (w <= 0 || h <= 0 || d_volume_ == nullptr || d_lut_ == nullptr) return;

  std::size_t bytes = static_cast<std::size_t>(w) * h * 4;
  if (d_output_ == nullptr || out_w_ != w || out_h_ != h) {
    if (d_output_) GPU_FREE(d_output_);
    GPU_MALLOC(reinterpret_cast<void **>(&d_output_), bytes);
    out_w_ = w;
    out_h_ = h;
    out_bytes_ = bytes;
  }

  dim3 block(16, 16);
  dim3 grid((w + 15) / 16, (h + 15) / 16);
  GPU_LAUNCH_KERNEL(render_kernel, grid, block, d_volume_, d_lut_, params,
                    camera, d_output_, w, h);
  GPU_DEVICE_SYNCHRONIZE();

  GPU_MEMCPY_D2H(fb.rgba.data(), d_output_, bytes);
}

}  // namespace mueye
