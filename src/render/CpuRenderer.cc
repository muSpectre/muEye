/**
 * @file   CpuRenderer.cc
 *
 * @brief  Implementation of the multi-threaded CPU ray tracer.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "render/CpuRenderer.hh"

#include <cstdint>

#include "render/parallel_for.hh"

namespace mueye {

void CpuRenderer::set_volume(const float *data, int, int, int) {
  // The CPU backend renders straight from the host buffer; no copy needed.
  volume_ = data;
}

void CpuRenderer::set_transfer_function(const Vec4 *lut, int) { lut_ = lut; }

void CpuRenderer::render(const RenderParams &params, const Camera &camera,
                         Framebuffer &fb) {
  const int w = fb.width;
  const int h = fb.height;
  if (w <= 0 || h <= 0 || volume_ == nullptr) return;

  std::uint8_t *out = fb.rgba.data();
  const ArraySampler sampler{volume_};

  // Parallelise across scanlines.
  parallel_for(h, nb_threads_, [&](int y) {
    for (int x = 0; x < w; ++x) {
      float u = (x + 0.5f) / w;
      float v = (y + 0.5f) / h;
      Vec4 c = trace_ray(sampler, lut_, params, camera, u, v);
      std::size_t idx = (static_cast<std::size_t>(y) * w + x) * 4;
      out[idx + 0] = static_cast<std::uint8_t>(clampf(c.x, 0.f, 1.f) * 255.f + 0.5f);
      out[idx + 1] = static_cast<std::uint8_t>(clampf(c.y, 0.f, 1.f) * 255.f + 0.5f);
      out[idx + 2] = static_cast<std::uint8_t>(clampf(c.z, 0.f, 1.f) * 255.f + 0.5f);
      out[idx + 3] = static_cast<std::uint8_t>(clampf(c.w, 0.f, 1.f) * 255.f + 0.5f);
    }
  });
}

}  // namespace mueye
