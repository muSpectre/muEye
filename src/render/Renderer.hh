/**
 * @file   Renderer.hh
 *
 * @brief  Abstract renderer interface and RGBA8 framebuffer for muEye.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_RENDERER_HH_
#define MUEYE_RENDERER_HH_

#include <cstdint>
#include <vector>

#include "render/render_core.hh"

namespace mueye {

/** Host-side RGBA8 image the renderer writes into and the GUI uploads to a GL
 *  texture. */
struct Framebuffer {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgba;  //!< width*height*4, row-major, top-left origin

  void resize(int w, int h) {
    if (w == width && h == height) return;
    width = w;
    height = h;
    rgba.assign(static_cast<std::size_t>(w) * h * 4, 0);
  }
};

/** Everything needed to render one frame. The volume / LUT pointers are owned by
 *  the caller and must outlive the call. */
struct RenderRequest {
  const float *volume{nullptr};  //!< contiguous float buffer, column-major
  const Vec4 *lut{nullptr};      //!< transfer-function LUT (lut_size entries)
  RenderParams params{};
  Camera camera{};
};

/** Common interface for the CPU and (future) GPU renderers. */
class Renderer {
 public:
  virtual ~Renderer() = default;

  /** Render @p req into @p fb (which is assumed already sized). */
  virtual void render(const RenderRequest &req, Framebuffer &fb) = 0;

  /** Human-readable backend name for the UI. */
  virtual const char *name() const = 0;
};

}  // namespace mueye

#endif  // MUEYE_RENDERER_HH_
