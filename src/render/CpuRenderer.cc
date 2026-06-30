/**
 * @file   CpuRenderer.cc
 *
 * @brief  Implementation of the multi-threaded CPU ray tracer.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "render/CpuRenderer.hh"

#include <cstdint>

namespace mueye {

void CpuRenderer::render(const RenderRequest &req, Framebuffer &fb) {
  const int w = fb.width;
  const int h = fb.height;
  if (w <= 0 || h <= 0 || req.volume == nullptr) return;

  std::uint8_t *out = fb.rgba.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // Pixel-centre sampling in normalized image coordinates.
      float u = (x + 0.5f) / w;
      float v = (y + 0.5f) / h;
      Vec4 c = trace_ray(req.volume, req.lut, req.params, req.camera, u, v);
      std::size_t idx = (static_cast<std::size_t>(y) * w + x) * 4;
      out[idx + 0] = static_cast<std::uint8_t>(clampf(c.x, 0.f, 1.f) * 255.f + 0.5f);
      out[idx + 1] = static_cast<std::uint8_t>(clampf(c.y, 0.f, 1.f) * 255.f + 0.5f);
      out[idx + 2] = static_cast<std::uint8_t>(clampf(c.z, 0.f, 1.f) * 255.f + 0.5f);
      out[idx + 3] = static_cast<std::uint8_t>(clampf(c.w, 0.f, 1.f) * 255.f + 0.5f);
    }
  }
}

}  // namespace mueye
