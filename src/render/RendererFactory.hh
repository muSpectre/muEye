/**
 * @file   RendererFactory.hh
 *
 * @brief  Discovery and construction of rendering backends.
 *
 * Backends are gated at compile time (CPU always; Metal on Apple when
 * MUEYE_ENABLE_METAL; CUDA/HIP when MUEYE_ENABLE_CUDA/HIP) and at run time
 * (a backend is only "available" if its device is actually present).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_RENDERER_FACTORY_HH_
#define MUEYE_RENDERER_FACTORY_HH_

#include <memory>
#include <vector>

#include "render/Renderer.hh"

namespace mueye {

struct BackendInfo {
  Backend backend;
  const char *name;   //!< display name
  bool available;     //!< compiled in AND a device is present
  const char *note;   //!< why unavailable, or ""
};

/** List every backend muEye knows about, with availability. CPU is always
 *  first and always available. */
std::vector<BackendInfo> enumerate_backends();

/** Construct a renderer for @p backend, or nullptr if it cannot be created
 *  (not compiled in, no device, or initialisation failed). Falls back to
 *  nullptr — callers should keep their previous renderer in that case. */
std::unique_ptr<Renderer> create_renderer(Backend backend);

}  // namespace mueye

#endif  // MUEYE_RENDERER_FACTORY_HH_
