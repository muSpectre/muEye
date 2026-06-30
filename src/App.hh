/**
 * @file   App.hh
 *
 * @brief  Application state and orchestration for muEye.
 *
 * Owns the data (Volume), the renderer, the camera and transfer function, and
 * drives loading + rendering in response to UI changes.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_APP_HH_
#define MUEYE_APP_HH_

#include <memory>
#include <string>

#include "gl/GlTexture.hh"
#include "io/Volume.hh"
#include "io/VolumeLoader.hh"
#include "render/CpuRenderer.hh"
#include "render/Renderer.hh"
#include "ui/OrbitCamera.hh"
#include "ui/TransferFunction.hh"

namespace mueye {

class App {
 public:
  App();

  /** Build and draw all ImGui windows for one frame (implemented in
   *  panels.cc). Re-renders the volume when camera/params changed. */
  void draw_ui();

  /** Open a file (e.g. from the command line) before the UI loop. */
  void load_file(const std::string &path) { open_path(path); }

 private:
  // --- data ------------------------------------------------------------
  void open_path(const std::string &path);
  void reload_volume();

  // --- rendering -------------------------------------------------------
  void render(int width, int height);

  VolumeLoader loader_;
  FileMeta meta_;
  Volume volume_;

  // --- UI / selection state -------------------------------------------
  std::string path_buf_;
  std::string status_ = "Open a muGrid NetCDF (.nc) file to begin.";
  bool has_file_{false};
  int field_index_{0};
  int frame_{0};
  int component_{0};
  Scalarize scalarize_{Scalarize::Component};

  // --- view / appearance ----------------------------------------------
  OrbitCamera camera_;
  TransferFunction tf_;
  RenderMode mode_{RenderMode::DVR};
  float step_{0.5f};          //!< in voxels; converted to world units per render
  float density_scale_{1.0f};
  float iso_value_{0.5f};
  float bg_[3] = {0.05f, 0.06f, 0.08f};

  // --- renderer / output ----------------------------------------------
  CpuRenderer cpu_renderer_;
  Renderer *active_renderer_{&cpu_renderer_};
  bool gpu_available_{false};  //!< GPU backend not built in this pass
  int cpu_threads_{0};         //!< 0 => OpenMP default

  Framebuffer fb_;
  GlTexture texture_;
  int render_downscale_{1};  //!< 1 = full viewport res; 2 = half, etc.
  bool needs_render_{true};
  double last_render_ms_{0.0};
  int last_render_w_{0};
  int last_render_h_{0};
};

}  // namespace mueye

#endif  // MUEYE_APP_HH_
