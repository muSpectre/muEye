/**
 * @file   GpuRenderer.cc
 *
 * @brief  CUDA / HIP implementation of the ray tracer.
 *
 * Compiled as CUDA (when MUEYE_ENABLE_CUDA) or HIP (when MUEYE_ENABLE_HIP); the
 * CMake target marks this file with LANGUAGE CUDA/HIP. It reuses the exact
 * ray-march kernel from render_core.hh (whose trace_ray_dvr()/trace_ray_iso()
 * are __host__ __device__ and templated on a volume sampler) and muGrid's
 * portability macros from memory/gpu_runtime.hh — so the GPU image matches the
 * CPU/Metal backends.
 *
 * The volume is uploaded into a hardware 3-D texture and sampled through a
 * TextureSampler (hardware trilinear filtering + texture cache). DVR and
 * isosurface are launched as separate single-path kernels (see render_core.hh
 * for why). render_to_gl() writes straight into a GL pixel buffer object shared
 * with CUDA/HIP, so the frame reaches the screen without a device->host copy.
 *
 * NOTE: this file is only built on a CUDA/HIP toolchain, not on the (CPU-only
 * macOS) development machine. CI (.github/workflows/gpu-build.yml) does a
 * compile + link check of both backends, but nothing here is *run* without a
 * GPU — treat runtime behaviour as unverified until exercised on real hardware.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "render/GpuRenderer.hh"

#include <string>

// muGrid's gpu_runtime.hh keys off MUGRID_ENABLE_CUDA / MUGRID_ENABLE_HIP and
// pulls in <cuda_runtime.h> / <hip/hip_runtime.h>. The muEye CMake defines the
// matching MUGRID_* macro alongside MUEYE_* when a GPU backend is enabled.
#include "memory/gpu_runtime.hh"
#include "render/render_core.hh"

// OpenGL entry points for the PBO upload path. GL_GLEXT_PROTOTYPES exposes the
// GL 1.5+ buffer functions (glGenBuffers/glBindBuffer/...) declared in glext.h;
// on Linux (the usual CUDA/HIP target) these are exported by libGL. Plus the
// CUDA/HIP <-> GL graphics-interop API.
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_gl_interop.h>
#elif defined(MUGRID_ENABLE_HIP)
#include <hip/hip_gl_interop.h>
#endif

namespace mueye {

// ---------------------------------------------------------------------------
// Backend-API spelling shim. CUDA and HIP expose an identical texture/array/
// graphics-interop runtime API up to the `cuda`/`hip` prefix, so we paste the
// prefix onto a common spelling and only differ where a name is not a simple
// prefixing (make_*Extent / make_*PitchedPtr are wrapped below).
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
using gpuGraphicsResource_t = GPU_(GraphicsResource_t);

// The texture object is stored backend-neutrally as an unsigned long long in the
// header (which must not name CUDA/HIP types). cudaTextureObject_t is already an
// integer handle, but hipTextureObject_t is a pointer — so pack/unpack it with
// the cast each backend actually permits (static for CUDA's integer, reinterpret
// for HIP's pointer).
inline unsigned long long tex_to_handle(gpuTextureObject_t t) {
#if defined(MUGRID_ENABLE_CUDA)
  return static_cast<unsigned long long>(t);
#else
  return reinterpret_cast<unsigned long long>(t);
#endif
}
inline gpuTextureObject_t handle_to_tex(unsigned long long h) {
#if defined(MUGRID_ENABLE_CUDA)
  return static_cast<gpuTextureObject_t>(h);
#else
  return reinterpret_cast<gpuTextureObject_t>(h);
#endif
}

namespace {

/** Hardware-texture volume sampler. Same value_at() interface as
 *  render_core.hh's ArraySampler, so the trace_ray_* templates are shared
 *  verbatim. Normalized coords + linear filtering + clamp addressing match
 *  ArraySampler's voxel-centre convention and clamp. */
struct TextureSampler {
  gpuTextureObject_t tex;

  MUEYE_HD float value_at(const RenderParams & /*p*/, const Vec3 &uvw) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return tex3D<float>(tex, uvw.x, uvw.y, uvw.z);
#else
    (void)uvw;
    return 0.0f;  // host stub; the texture sampler is only used on device
#endif
  }
};

/** Hardware-texture displacement sampler (float4 3-D texture), mirroring
 *  render_core.hh's ArrayDispSampler::disp_at for the warp path. */
struct TextureDispSampler {
  gpuTextureObject_t tex;  // 0 when no displacement is uploaded

  MUEYE_HD Vec3 disp_at(const RenderParams & /*p*/, const Vec3 &uvw) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    float4 d = tex3D<float4>(tex, uvw.x, uvw.y, uvw.z);
    return Vec3{d.x, d.y, d.z};
#else
    (void)uvw;
    return Vec3{0.0f, 0.0f, 0.0f};  // host stub
#endif
  }
};

MUEYE_HD inline void write_rgba(unsigned char *out, std::size_t idx, Vec4 c) {
  out[idx + 0] = static_cast<unsigned char>(clampf(c.x, 0.f, 1.f) * 255.f + 0.5f);
  out[idx + 1] = static_cast<unsigned char>(clampf(c.y, 0.f, 1.f) * 255.f + 0.5f);
  out[idx + 2] = static_cast<unsigned char>(clampf(c.z, 0.f, 1.f) * 255.f + 0.5f);
  out[idx + 3] = static_cast<unsigned char>(clampf(c.w, 0.f, 1.f) * 255.f + 0.5f);
}

// DVR and isosurface are separate kernels so each compiles a single path (see
// render_core.hh): p.mode is grid-uniform — never divergent — but a combined
// kernel inflates register/instruction footprint and can cap occupancy.
__global__ void render_kernel_dvr(gpuTextureObject_t vol,
                                  gpuTextureObject_t disp,
                                  const Vec4 *MUEYE_RESTRICT lut, RenderParams p,
                                  Camera cam, unsigned char *MUEYE_RESTRICT out,
                                  int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float u = (x + 0.5f) / w;
  float v = (y + 0.5f) / h;
  // NoBricks: no brick summary is uploaded to the device yet, so the GPU
  // path renders without empty-space skipping (same image, more samples).
  Vec4 c = trace_ray_dvr(TextureSampler{vol}, TextureDispSampler{disp},
                         NoBricks{}, lut, p, cam, u, v);
  write_rgba(out, (static_cast<std::size_t>(y) * w + x) * 4, c);
}

__global__ void render_kernel_iso(gpuTextureObject_t vol,
                                  gpuTextureObject_t disp, RenderParams p,
                                  Camera cam, unsigned char *MUEYE_RESTRICT out,
                                  int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float u = (x + 0.5f) / w;
  float v = (y + 0.5f) / h;
  Vec4 c = trace_ray_iso(TextureSampler{vol}, TextureDispSampler{disp},
                         NoBricks{}, p, cam, u, v);
  write_rgba(out, (static_cast<std::size_t>(y) * w + x) * 4, c);
}

/** Launch the kernel matching p.mode, writing RGBA8 into @p out (device ptr). */
void launch(gpuTextureObject_t tex, gpuTextureObject_t disp, const Vec4 *lut,
            RenderParams p, const Camera &cam, unsigned char *out, int w,
            int h) {
  // Never warp without a resident displacement texture (mirrors the Metal
  // backend's guard); the sampler would otherwise fetch an invalid texture.
  if (p.warp_enabled && disp == 0) p.warp_enabled = 0;
  dim3 block(16, 16);
  dim3 grid((w + 15) / 16, (h + 15) / 16);
  if (p.mode == RenderMode::Isosurface) {
    GPU_LAUNCH_KERNEL(render_kernel_iso, grid, block, tex, disp, p, cam, out, w,
                      h);
  } else {
    GPU_LAUNCH_KERNEL(render_kernel_dvr, grid, block, tex, disp, lut, p, cam,
                      out, w, h);
  }
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
  if (pbo_res_)
    GPU_(GraphicsUnregisterResource)(
        static_cast<gpuGraphicsResource_t>(pbo_res_));
  if (pbo_) glDeleteBuffers(1, &pbo_);
  if (d_tex_) GPU_(DestroyTextureObject)(handle_to_tex(d_tex_));
  if (d_array_) GPU_(FreeArray)(static_cast<gpuArray_t>(d_array_));
  if (d_disp_tex_) GPU_(DestroyTextureObject)(handle_to_tex(d_disp_tex_));
  if (d_disp_array_) GPU_(FreeArray)(static_cast<gpuArray_t>(d_disp_array_));
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
  error_.clear();

  // Release any previous texture + backing array. On any failure below we
  // return with d_tex_ == 0, which makes render()/render_to_gl() no-ops until
  // a later upload succeeds.
  if (d_tex_) {
    GPU_(DestroyTextureObject)(handle_to_tex(d_tex_));
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
  // The trailing flags arg is required by HIP's hipMalloc3DArray (CUDA defaults
  // it to 0); pass 0 explicitly so the call compiles under both.
  GPU_(Error_t) err =
      GPU_(Malloc3DArray)(&array, &channel, gpu_make_extent(nx, ny, nz), 0);
  if (err != GPU_(Success) || array == nullptr) {
    error_ = std::string("volume allocation failed (is the volume too large "
                         "for device memory?): ") +
             GPU_(GetErrorString)(err);
    return;
  }

  GPU_(Memcpy3DParms) copy = {};
  copy.srcPtr = gpu_make_pitched(const_cast<float *>(data),
                                 static_cast<std::size_t>(nx) * sizeof(float),
                                 nx, ny);
  copy.dstArray = array;
  copy.extent = gpu_make_extent(nx, ny, nz);
  copy.kind = GPU_(MemcpyHostToDevice);
  err = GPU_(Memcpy3D)(&copy);
  if (err != GPU_(Success)) {
    GPU_(FreeArray)(array);
    error_ = std::string("volume upload failed: ") + GPU_(GetErrorString)(err);
    return;
  }

  // A texture object over that array: linear filtering + clamp addressing +
  // normalized coordinates so tex3D matches ArraySampler up to the texture
  // unit's fixed-point interpolation weights.
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
  err = GPU_(CreateTextureObject)(&tex, &res, &tex_desc, nullptr);
  if (err != GPU_(Success)) {
    GPU_(FreeArray)(array);
    error_ = std::string("volume texture creation failed: ") +
             GPU_(GetErrorString)(err);
    return;
  }

  d_array_ = array;
  d_tex_ = tex_to_handle(tex);
}

void GpuRenderer::set_transfer_function(const Vec4 *lut, int n) {
  if (lut == nullptr || n <= 0) return;
  error_.clear();
  if (d_lut_) {
    GPU_FREE(d_lut_);
    d_lut_ = nullptr;
  }
  std::size_t bytes = static_cast<std::size_t>(n) * sizeof(Vec4);
  GPU_(Error_t) err = GPU_(Malloc)(reinterpret_cast<void **>(&d_lut_), bytes);
  if (err != GPU_(Success)) {
    d_lut_ = nullptr;
    error_ = std::string("transfer-function allocation failed: ") +
             GPU_(GetErrorString)(err);
    return;
  }
  err = GPU_(Memcpy)(d_lut_, lut, bytes, GPU_(MemcpyHostToDevice));
  if (err != GPU_(Success)) {
    GPU_FREE(d_lut_);
    d_lut_ = nullptr;
    error_ = std::string("transfer-function upload failed: ") +
             GPU_(GetErrorString)(err);
  }
}

void GpuRenderer::set_displacement(const float *data, int nx, int ny, int nz) {
  error_.clear();
  // Always release the previous displacement texture/array first.
  if (d_disp_tex_) {
    GPU_(DestroyTextureObject)(handle_to_tex(d_disp_tex_));
    d_disp_tex_ = 0;
  }
  if (d_disp_array_) {
    GPU_(FreeArray)(static_cast<gpuArray_t>(d_disp_array_));
    d_disp_array_ = nullptr;
  }
  if (data == nullptr) return;  // clear only

  // A float4 3-D array: the host buffer already stores 4 floats per voxel.
  GPU_(ChannelFormatDesc) channel = GPU_(CreateChannelDesc)<float4>();
  gpuArray_t array = nullptr;
  GPU_(Error_t) err =
      GPU_(Malloc3DArray)(&array, &channel, gpu_make_extent(nx, ny, nz), 0);
  if (err != GPU_(Success) || array == nullptr) {
    error_ = std::string("displacement allocation failed: ") +
             GPU_(GetErrorString)(err);
    return;
  }

  GPU_(Memcpy3DParms) copy = {};
  copy.srcPtr = gpu_make_pitched(const_cast<float *>(data),
                                 static_cast<std::size_t>(nx) * 4 * sizeof(float),
                                 nx, ny);
  copy.dstArray = array;
  copy.extent = gpu_make_extent(nx, ny, nz);
  copy.kind = GPU_(MemcpyHostToDevice);
  err = GPU_(Memcpy3D)(&copy);
  if (err != GPU_(Success)) {
    GPU_(FreeArray)(array);
    error_ =
        std::string("displacement upload failed: ") + GPU_(GetErrorString)(err);
    return;
  }

  // Same sampling convention as the volume (linear + clamp + normalized coords)
  // so the warp's fractional lookups match ArrayDispSampler.
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
  err = GPU_(CreateTextureObject)(&tex, &res, &tex_desc, nullptr);
  if (err != GPU_(Success)) {
    GPU_(FreeArray)(array);
    error_ = std::string("displacement texture creation failed: ") +
             GPU_(GetErrorString)(err);
    return;
  }
  d_disp_array_ = array;
  d_disp_tex_ = tex_to_handle(tex);
}

void GpuRenderer::render(const RenderParams &params, const Camera &camera,
                         Framebuffer &fb) {
  const int w = fb.width, h = fb.height;
  if (w <= 0 || h <= 0 || d_tex_ == 0 || d_lut_ == nullptr) return;

  std::size_t bytes = static_cast<std::size_t>(w) * h * 4;
  if (d_output_ == nullptr || out_w_ != w || out_h_ != h) {
    if (d_output_) {
      GPU_FREE(d_output_);
      d_output_ = nullptr;
    }
    GPU_(Error_t) err =
        GPU_(Malloc)(reinterpret_cast<void **>(&d_output_), bytes);
    if (err != GPU_(Success)) {
      d_output_ = nullptr;
      out_w_ = 0;
      out_h_ = 0;
      error_ = std::string("output framebuffer allocation failed: ") +
               GPU_(GetErrorString)(err);
      return;
    }
    out_w_ = w;
    out_h_ = h;
  }

  launch(handle_to_tex(d_tex_), handle_to_tex(d_disp_tex_), d_lut_, params,
         camera, d_output_, w, h);
  // Use the non-throwing runtime calls and report through last_error():
  // muGrid's GPU_CHECK-based macros throw, and an exception escaping render()
  // would take the whole viewer down for a recoverable device fault.
  GPU_(Error_t) err = GPU_(GetLastError)();
  if (err == GPU_(Success)) err = GPU_(DeviceSynchronize)();
  if (err != GPU_(Success)) {
    error_ = std::string("render kernel failed: ") + GPU_(GetErrorString)(err);
    return;
  }
  err = GPU_(Memcpy)(fb.rgba.data(), d_output_, bytes, GPU_(MemcpyDeviceToHost));
  if (err != GPU_(Success)) {
    error_ = std::string("framebuffer download failed: ") +
             GPU_(GetErrorString)(err);
  }
}

bool GpuRenderer::render_to_gl(const RenderParams &params, const Camera &camera,
                               unsigned int gl_tex, int width, int height) {
  if (interop_failed_ || width <= 0 || height <= 0 || d_tex_ == 0 ||
      d_lut_ == nullptr || gl_tex == 0)
    return false;

  // (Re)create the shared pixel buffer object when the size changes.
  if (pbo_ == 0 || pbo_w_ != width || pbo_h_ != height) {
    if (pbo_res_) {
      GPU_(GraphicsUnregisterResource)(
          static_cast<gpuGraphicsResource_t>(pbo_res_));
      pbo_res_ = nullptr;
    }
    if (pbo_ == 0) glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 static_cast<GLsizeiptr>(width) * height * 4, nullptr,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    gpuGraphicsResource_t res = nullptr;
    if (GPU_(GraphicsGLRegisterBuffer)(
            &res, pbo_, GPU_(GraphicsRegisterFlagsWriteDiscard)) !=
        GPU_(Success)) {
      // Registration failed (e.g. CUDA/GL not on the same device). It will
      // keep failing, so remember that and fall back to the host copy path
      // for good — without the flag, every frame would recreate the buffer
      // and retry the registration.
      glDeleteBuffers(1, &pbo_);
      pbo_ = 0;
      interop_failed_ = true;
      return false;
    }
    pbo_res_ = res;
    pbo_w_ = width;
    pbo_h_ = height;
  }

  auto res = static_cast<gpuGraphicsResource_t>(pbo_res_);
  unsigned char *dev_ptr = nullptr;
  std::size_t mapped_bytes = 0;
  const std::size_t needed = static_cast<std::size_t>(width) * height * 4;
  GPU_(Error_t) err = GPU_(GraphicsMapResources)(1, &res, 0);
  if (err != GPU_(Success)) {
    // Never launch into an unmapped buffer. Give up on interop for good (the
    // caller falls back to render() + host upload) rather than retrying and
    // failing every frame.
    error_ = std::string("GL interop map failed: ") + GPU_(GetErrorString)(err);
    interop_failed_ = true;
    return false;
  }
  err = GPU_(GraphicsResourceGetMappedPointer)(
      reinterpret_cast<void **>(&dev_ptr), &mapped_bytes, res);
  if (err != GPU_(Success) || dev_ptr == nullptr || mapped_bytes < needed) {
    GPU_(GraphicsUnmapResources)(1, &res, 0);
    error_ = err != GPU_(Success)
                 ? std::string("GL interop pointer failed: ") +
                       GPU_(GetErrorString)(err)
                 : std::string("GL interop buffer smaller than the frame");
    interop_failed_ = true;
    return false;
  }

  launch(handle_to_tex(d_tex_), handle_to_tex(d_disp_tex_), d_lut_, params,
         camera, dev_ptr, width, height);
  err = GPU_(GetLastError)();

  // Unmapping synchronizes the render with subsequent GL use of the buffer.
  GPU_(Error_t) unmap_err = GPU_(GraphicsUnmapResources)(1, &res, 0);
  if (err == GPU_(Success)) err = unmap_err;
  if (err != GPU_(Success)) {
    // A launch or sync fault: report it and let the caller fall back to the
    // host path for this frame (which surfaces its own error if the device
    // is wedged).
    error_ = std::string("render kernel failed: ") + GPU_(GetErrorString)(err);
    return false;
  }

  // Device-to-device copy PBO -> texture (no host round trip).
  glBindTexture(GL_TEXTURE_2D, gl_tex);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                  GL_UNSIGNED_BYTE, nullptr);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  return true;
}

}  // namespace mueye
