/**
 * @file   CpuRenderer.hh
 *
 * @brief  Multi-threaded CPU ray tracer for muEye.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_CPU_RENDERER_HH_
#define MUEYE_CPU_RENDERER_HH_

#include "render/Renderer.hh"

namespace mueye {

/** Renders by running trace_ray() over every output pixel, parallelised with
 *  OpenMP when available. */
class CpuRenderer : public Renderer {
 public:
  void render(const RenderRequest &req, Framebuffer &fb) override;
  const char *name() const override { return "CPU"; }
};

}  // namespace mueye

#endif  // MUEYE_CPU_RENDERER_HH_
