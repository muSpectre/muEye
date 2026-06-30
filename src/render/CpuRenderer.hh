/**
 * @file   CpuRenderer.hh
 *
 * @brief  Multi-threaded CPU ray tracer (OpenMP or std::thread).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_CPU_RENDERER_HH_
#define MUEYE_CPU_RENDERER_HH_

#include "render/Renderer.hh"

namespace mueye {

/** Renders by running render_core::trace_ray() over every output pixel,
 *  parallelised across scanlines. */
class CpuRenderer : public Renderer {
 public:
  void set_volume(const float *data, int nx, int ny, int nz) override;
  void set_transfer_function(const Vec4 *lut, int n) override;
  void render(const RenderParams &params, const Camera &camera,
              Framebuffer &fb) override;
  const char *name() const override { return "CPU"; }
  Backend backend() const override { return Backend::CPU; }

  /** Number of worker threads; <= 0 means auto. */
  void set_num_threads(int n) override { nb_threads_ = n; }

 private:
  const float *volume_{nullptr};
  const Vec4 *lut_{nullptr};
  int nb_threads_{0};
};

}  // namespace mueye

#endif  // MUEYE_CPU_RENDERER_HH_
