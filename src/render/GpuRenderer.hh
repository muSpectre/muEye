/**
 * @file   GpuRenderer.hh
 *
 * @brief  CUDA / HIP compute ray tracer.
 *
 * Pure-C++ interface so it can be included from ordinary translation units. The
 * implementation (GpuRenderer.cc) is compiled as CUDA or HIP and shares the
 * ray-march kernel with the CPU backend through render_core.hh's
 * __host__ __device__ trace_ray(). Device memory / launch / portability come
 * from muGrid's memory/gpu_runtime.hh.
 *
 * Only built when MUEYE_ENABLE_CUDA or MUEYE_ENABLE_HIP is set.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_GPU_RENDERER_HH_
#define MUEYE_GPU_RENDERER_HH_

#include <cstddef>

#include "render/Renderer.hh"

namespace mueye {

class GpuRenderer : public Renderer {
 public:
  /** True if at least one CUDA/HIP device is present at runtime. */
  static bool available();

  GpuRenderer();
  ~GpuRenderer() override;

  GpuRenderer(const GpuRenderer &) = delete;
  GpuRenderer &operator=(const GpuRenderer &) = delete;

  void set_volume(const float *data, int nx, int ny, int nz) override;
  void set_transfer_function(const Vec4 *lut, int n) override;
  void render(const RenderParams &params, const Camera &camera,
              Framebuffer &fb) override;
  /** Zero-copy path: the kernel writes into a GL pixel buffer object shared
   *  with CUDA/HIP, which is then copied device-to-device into @p gl_tex — no
   *  device->host round trip. */
  bool render_to_gl(const RenderParams &params, const Camera &camera,
                    unsigned int gl_tex, int width, int height) override;
  const char *name() const override;
  Backend backend() const override;

 private:
  // The volume lives in a hardware 3-D texture (hardware trilinear filtering +
  // texture cache). The array backing and the texture object are stored as
  // opaque handles so this header stays free of CUDA/HIP types (it is included
  // by plain C++ translation units): d_array_ is a cudaArray_t / hipArray_t and
  // d_tex_ a cudaTextureObject_t / hipTextureObject_t (an unsigned long long).
  void *d_array_{nullptr};
  unsigned long long d_tex_{0};
  Vec4 *d_lut_{nullptr};
  unsigned char *d_output_{nullptr};
  int nx_{0}, ny_{0}, nz_{0};
  int out_w_{0}, out_h_{0};
  std::size_t out_bytes_{0};

  // GL-interop scratch for render_to_gl(): a GL pixel-unpack buffer registered
  // with CUDA/HIP. pbo_res_ is an opaque cudaGraphicsResource_t / hipGraphics-
  // Resource_t (kept as void* so this header stays free of CUDA/HIP + GL types).
  unsigned int pbo_{0};
  void *pbo_res_{nullptr};
  int pbo_w_{0}, pbo_h_{0};
};

}  // namespace mueye

#endif  // MUEYE_GPU_RENDERER_HH_
