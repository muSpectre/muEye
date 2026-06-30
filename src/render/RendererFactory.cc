/**
 * @file   RendererFactory.cc
 *
 * @brief  Backend discovery and construction.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "render/RendererFactory.hh"

#include "render/CpuRenderer.hh"

#if defined(MUEYE_ENABLE_METAL)
#include "render/MetalRenderer.hh"
#endif

#if defined(MUEYE_ENABLE_CUDA) || defined(MUEYE_ENABLE_HIP)
#include "render/GpuRenderer.hh"
#endif

namespace mueye {

const char *to_string(Backend b) {
  switch (b) {
    case Backend::CPU:
      return "CPU";
    case Backend::Metal:
      return "Metal (GPU)";
    case Backend::CUDA:
      return "CUDA (GPU)";
    case Backend::HIP:
      return "HIP (GPU)";
  }
  return "?";
}

std::vector<BackendInfo> enumerate_backends() {
  std::vector<BackendInfo> out;

  // CPU — always present.
  out.push_back({Backend::CPU, "CPU", true, ""});

  // Metal.
#if defined(MUEYE_ENABLE_METAL)
  {
    bool dev = MetalRenderer::available();
    out.push_back({Backend::Metal, "Metal (GPU)", dev,
                   dev ? "" : "no Metal device"});
  }
#else
  out.push_back({Backend::Metal, "Metal (GPU)", false, "not built"});
#endif

  // CUDA / HIP.
#if defined(MUEYE_ENABLE_CUDA)
  {
    bool dev = GpuRenderer::available();
    out.push_back({Backend::CUDA, "CUDA (GPU)", dev, dev ? "" : "no CUDA device"});
  }
#elif defined(MUEYE_ENABLE_HIP)
  {
    bool dev = GpuRenderer::available();
    out.push_back({Backend::HIP, "HIP (GPU)", dev, dev ? "" : "no HIP device"});
  }
#else
  out.push_back({Backend::CUDA, "CUDA / HIP (GPU)", false, "not built"});
#endif

  return out;
}

std::unique_ptr<Renderer> create_renderer(Backend backend) {
  switch (backend) {
    case Backend::CPU:
      return std::make_unique<CpuRenderer>();

    case Backend::Metal:
#if defined(MUEYE_ENABLE_METAL)
    {
      auto r = std::make_unique<MetalRenderer>();
      if (r->ok()) return r;
    }
#endif
      return nullptr;

    case Backend::CUDA:
    case Backend::HIP:
#if defined(MUEYE_ENABLE_CUDA) || defined(MUEYE_ENABLE_HIP)
      if (GpuRenderer::available()) return std::make_unique<GpuRenderer>();
#endif
      return nullptr;
  }
  return nullptr;
}

}  // namespace mueye
