/**
 * @file   Renderer.hh
 *
 * @brief  Rendering backend abstraction for muEye.
 *
 * A `Renderer` is a pluggable ray-tracing backend. The interface deliberately
 * separates *data upload* (set_volume / set_transfer_function — potentially
 * expensive, done only when the data changes) from *per-frame rendering*
 * (render — called on every camera/parameter change). This lets GPU backends
 * keep the volume resident in device memory across frames.
 *
 * Concrete backends:
 *   - CpuRenderer   (always available; OpenMP or std::thread parallel)
 *   - MetalRenderer (Apple GPUs; compute shader)
 *   - GpuRenderer   (CUDA / HIP; single-source kernel sharing render_core.hh)
 *
 * Backends are discovered and instantiated through RendererFactory.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_RENDERER_HH_
#define MUEYE_RENDERER_HH_

#include <cstdint>
#include <vector>

#include "render/render_core.hh"

namespace mueye {

/** Identifies a rendering backend. */
enum class Backend : int { CPU = 0, Metal = 1, CUDA = 2, HIP = 3 };

const char *to_string(Backend b);

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

/**
 * A ray-tracing backend.
 *
 * Lifecycle: construct → set_volume() / set_transfer_function() whenever the
 * data changes → render() once per frame. set_volume() and
 * set_transfer_function() must be called at least once before render().
 */
class Renderer {
 public:
  virtual ~Renderer() = default;

  /** Upload (and take ownership of a device copy of) the scalar volume. For the
   *  CPU backend this just stores the pointer (the caller keeps the data alive
   *  until the next set_volume); GPU backends copy to device memory. The data
   *  is a contiguous float buffer in column-major order (idx = i+nx*(j+ny*k)). */
  virtual void set_volume(const float *data, int nx, int ny, int nz) = 0;

  /** Upload the transfer-function LUT (n RGBA entries). */
  virtual void set_transfer_function(const Vec4 *lut, int n) = 0;

  /** Render one frame into @p fb (assumed already sized). */
  virtual void render(const RenderParams &params, const Camera &camera,
                      Framebuffer &fb) = 0;

  /** Human-readable backend name for the UI. */
  virtual const char *name() const = 0;

  /** Which backend this is. */
  virtual Backend backend() const = 0;

  /** Set the worker-thread count (CPU backend only; <= 0 means auto). No-op for
   *  GPU backends. */
  virtual void set_num_threads(int /*n*/) {}
};

}  // namespace mueye

#endif  // MUEYE_RENDERER_HH_
