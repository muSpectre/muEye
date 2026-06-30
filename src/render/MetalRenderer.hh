/**
 * @file   MetalRenderer.hh
 *
 * @brief  Metal compute-shader ray tracer (Apple GPUs).
 *
 * Pure-C++ interface; all Metal/Objective-C types are hidden behind a pImpl so
 * this header can be included from ordinary C++ translation units. The
 * implementation lives in MetalRenderer.mm.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_METAL_RENDERER_HH_
#define MUEYE_METAL_RENDERER_HH_

#include "render/Renderer.hh"

namespace mueye {

class MetalRenderer : public Renderer {
 public:
  /** True if a Metal device is present and the compute pipeline can be built. */
  static bool available();

  MetalRenderer();
  ~MetalRenderer() override;

  MetalRenderer(const MetalRenderer &) = delete;
  MetalRenderer &operator=(const MetalRenderer &) = delete;

  void set_volume(const float *data, int nx, int ny, int nz) override;
  void set_transfer_function(const Vec4 *lut, int n) override;
  void render(const RenderParams &params, const Camera &camera,
              Framebuffer &fb) override;
  const char *name() const override;
  Backend backend() const override { return Backend::Metal; }

  /** True if construction succeeded (device + pipeline ready). */
  bool ok() const;

 private:
  struct Impl;
  Impl *impl_;
};

}  // namespace mueye

#endif  // MUEYE_METAL_RENDERER_HH_
