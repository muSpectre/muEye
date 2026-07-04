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
#include "render/Renderer.hh"
#include "render/RendererFactory.hh"
#include "ui/OrbitCamera.hh"
#include "ui/TransferFunction.hh"

namespace mueye {

/** World-space extents of the volume box: the grid shape normalized so the
 *  longest axis is 1. Cubic grids give {1,1,1}; a 2D grid (nz==1) gives a thin
 *  box that renders as a plane. Keep in sync with RenderParams::box. */
inline Vec3 box_extent(int nx, int ny, int nz) {
  int m = nx;
  if (ny > m) m = ny;
  if (nz > m) m = nz;
  if (m < 1) m = 1;
  float f = 1.0f / static_cast<float>(m);
  return Vec3{nx * f, ny * f, nz * f};
}

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
  void set_backend(Backend backend);
  RenderParams make_render_params() const;
  void sync_renderer_data();  //!< (re-)upload volume / LUT if dirty

  /** Re-render the current scene into a host framebuffer at full viewport
   *  resolution (ignoring the interactive downscale) and write it as a PNG. */
  void save_png(const std::string &path);

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
  float bg_[3] = {1.0f, 1.0f, 1.0f};  // white scene background
  bool show_box_{true};   //!< draw the volume box outline over the viewport
  bool periodic_{false};  //!< tile the volume periodically
  int replicas_[3] = {2, 2, 2};  //!< replicas per axis while periodic_ is on

  /** Replica count actually applied along @p axis (1 unless periodic tiling
   *  is enabled). */
  int rep(int axis) const { return periodic_ ? replicas_[axis] : 1; }

  // --- renderer / output ----------------------------------------------
  std::unique_ptr<Renderer> renderer_;
  Backend current_backend_{Backend::CPU};
  std::vector<BackendInfo> backends_;
  int cpu_threads_{0};       //!< 0 => auto
  bool volume_dirty_{true};  //!< re-upload volume to the backend before next render
  bool tf_dirty_{true};      //!< re-upload transfer function before next render

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
