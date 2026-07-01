/**
 * @file   GpuRenderer.cc
 *
 * @brief  CUDA / HIP implementation of the ray tracer.
 *
 * Compiled as CUDA (when MUEYE_ENABLE_CUDA) or HIP (when MUEYE_ENABLE_HIP); the
 * CMake target marks this file with LANGUAGE CUDA/HIP. It reuses the exact
 * ray-march kernel from render_core.hh (whose trace_ray() is __host__ __device__
 * and templated on a volume sampler) and muGrid's portability macros from
 * memory/gpu_runtime.hh — so the GPU image matches the CPU/Metal backends.
 *
 * The volume is uploaded into a hardware 3-D texture and sampled through a
 * TextureSampler: the texture unit performs the trilinear interpolation
 * (replacing 8 global loads + 7 lerps per sample with one filtered fetch) and
 * the read is served by the texture cache, which is far better suited to the
 * 3-D-local, mildly divergent access pattern of ray marching than plain global
 * memory. Clamp addressing reproduces ArraySampler's clamp-to-edge exactly.
 *
 * NOTE: this file is only built on a CUDA/HIP toolchain; it is not compiled on
 * the (CPU-only macOS) development machine, nor in CI. Treat changes here as
 * unverified until run on real hardware.
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

// ---------------------------------------------------------------------------
// Backend-API spelling shim. CUDA and HIP expose an identical texture/array
// runtime API up to the `cuda`/`hip` prefix, so we paste the prefix onto a
// common spelling and only differ where a name is not a simple prefixing
// (make_*Extent / make_*PitchedPtr are wrapped below).
// ---------------------------------------------------------------------------
#if defined(MUGRID_ENABLE_CUDA)
#define GPU_(name) cuda##name
inline cudaExtent gpu_make_extent(std::size_t w, std::size_t h, std::size_t d) {
  return make_cudaExtent(w, h, d);
}
inline cudaPitchedPtr gpu_make_pitched(void *p, std::size_t pitch,
                                       std::size_t xsize, std::size_t ysize) {
  return make_cudaPitchedPtr(p, pitch, xsize, ysize);
}
#elif defined(MUGRID_ENABLE_HIP)
#define GPU_(name) hip##name
inline hipExtent gpu_make_extent(std::size_t w, std::size_t h, std::size_t d) {
  return make_hipExtent(w, h, d);
}
inline hipPitchedPtr gpu_make_pitched(void *p, std::size_t pitch,
                                      std::size_t xsize, std::size_t ysize) {
  return make_hipPitchedPtr(p, pitch, xsize, ysize);
}
#endif

using gpuTextureObject_t = GPU_(TextureObject_t);
using gpuArray_t = GPU_(Array_t);

namespace {

/** Hardware-texture volume sampler. Same value_at() interface as
 *  render_core.hh's ArraySampler, so trace_ray() is shared verbatim. The
 *  texture uses normalized coordinates with linear filtering and clamp
 *  addressing, matching ArraySampler's voxel-centre convention and clamp. */
struct TextureSampler {
  gpuTextureObject_t tex;

  MUEYE_HD float value_at(const RenderParams & /*p*/, const Vec3 &uvw) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // tex3D with normalized coords places texel centres at (i+0.5)/dim — the
    // same continuous-voxel mapping ArraySampler computes as uvw*dim - 0.5.
    return tex3D<float>(tex, uvw.x, uvw.y, uvw.z);
#else
    // Host stub: the texture sampler is only ever invoked from device code.
    (void)uvw;
    return 0.0f;
#endif
  }
};

__global__ void render_kernel(gpuTextureObject_t vol,
                              const Vec4 *MUEYE_RESTRICT lut, RenderParams p,
                              Camera cam, unsigned char *MUEYE_RESTRICT out,
                              int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float u = (x + 0.5f) / w;
  float v = (y + 0.5f) / h;
  TextureSampler s{vol};
  Vec4 c = trace_ray(s, lut, p, cam, u, v);
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
  if (d_tex_) GPU_(DestroyTextureObject)(static_cast<gpuTextureObject_t>(d_tex_));
  if (d_array_) GPU_(FreeArray)(static_cast<gpuArray_t>(d_array_));
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

  // Release any previous texture + backing array.
  if (d_tex_) {
    GPU_(DestroyTextureObject)(static_cast<gpuTextureObject_t>(d_tex_));
    d_tex_ = 0;
  }
  if (d_array_) {
    GPU_(FreeArray)(static_cast<gpuArray_t>(d_array_));
    d_array_ = nullptr;
  }
  nx_ = nx;
  ny_ = ny;
  nz_ = nz;

  // Allocate a float 3-D array (extent is in elements) and copy the volume in.
  GPU_(ChannelFormatDesc) channel = GPU_(CreateChannelDesc)<float>();
  gpuArray_t array = nullptr;
  GPU_(Malloc3DArray)(&array, &channel, gpu_make_extent(nx, ny, nz));

  GPU_(Memcpy3DParms) copy = {};
  copy.srcPtr = gpu_make_pitched(const_cast<float *>(data),
                                 static_cast<std::size_t>(nx) * sizeof(float),
                                 nx, ny);
  copy.dstArray = array;
  copy.extent = gpu_make_extent(nx, ny, nz);
  copy.kind = GPU_(MemcpyHostToDevice);
  GPU_(Memcpy3D)(&copy);

  // A texture object over that array: linear filtering + clamp addressing +
  // normalized coordinates so tex3D matches ArraySampler bit-for-bit up to the
  // texture unit's fixed-point interpolation weights.
  GPU_(ResourceDesc) res = {};
  res.resType = GPU_(ResourceTypeArray);
  res.res.array.array = array;

  GPU_(TextureDesc) tex_desc = {};
  tex_desc.addressMode[0] = GPU_(AddressModeClamp);
  tex_desc.addressMode[1] = GPU_(AddressModeClamp);
  tex_desc.addressMode[2] = GPU_(AddressModeClamp);
  tex_desc.filterMode = GPU_(FilterModeLinear);
  tex_desc.readMode = GPU_(ReadModeElementType);
  tex_desc.normalizedCoords = 1;

  gpuTextureObject_t tex = 0;
  GPU_(CreateTextureObject)(&tex, &res, &tex_desc, nullptr);

  d_array_ = array;
  d_tex_ = tex;
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
  if (w <= 0 || h <= 0 || d_tex_ == 0 || d_lut_ == nullptr) return;

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
  GPU_LAUNCH_KERNEL(render_kernel, grid, block,
                    static_cast<gpuTextureObject_t>(d_tex_), d_lut_, params,
                    camera, d_output_, w, h);
  GPU_DEVICE_SYNCHRONIZE();

  GPU_MEMCPY_D2H(fb.rgba.data(), d_output_, bytes);
}

}  // namespace mueye
