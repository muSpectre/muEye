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
  void reload_displacement();  //!< (re)read the selected displacement field

  // --- rendering -------------------------------------------------------
  /** Render the scene for a @p width x @p height viewport at 1/@p downscale
   *  of that resolution (the texture is stretched back onto the viewport). */
  void render(int width, int height, int downscale);

  /** Downscale to render with right now: the user's setting, times an
   *  automatic factor while the user is interacting (mouse held) when
   *  adaptive quality is on. The factor is chosen from the last
   *  full-quality render time so an interactive frame takes roughly
   *  kInteractiveTargetMs. */
  int effective_downscale(bool interacting) const;
  void set_backend(Backend backend);
  RenderParams make_render_params() const;
  void sync_renderer_data();  //!< (re-)upload volume / LUT if dirty

  /** Reference box (aspect) the cell is built on: the file's `domain_lengths`
   *  normalized so the longest axis is 1 when present, else the grid shape
   *  (box_extent). C = F * diag(reference_box). */
  Vec3 reference_box() const;

  /** If the file carries a per-frame applied deformation gradient
   *  (`meta_.applied_F`), copy the current frame's tensor into F_ (marking it as
   *  from-file). No-op otherwise. */
  void sync_frame_deformation();

  /** Forward cell matrix C = F * diag(box): its columns are the world-space
   *  edge vectors of the (possibly sheared) rendered cell. */
  Mat3 world_cell() const;
  /** Axis-aligned bounds of the deformed, tiled cell (8 corners of C*[0,rep]),
   *  used to frame the camera on non-orthogonal cells. */
  void cell_bounds(Vec3 &center, float &extent) const;

  /** Frame the camera on the (tiled, sheared) cell. With @p reset_angles the
   *  view direction is reset too: the default oblique 3D view, or face-on
   *  (looking down z) for a 2D field. */
  void frame_view(bool reset_angles);

  /** Re-render the current scene into a host framebuffer at full viewport
   *  resolution (ignoring the interactive downscale) and write it as a PNG. */
  void save_png(const std::string &path);

  VolumeLoader loader_;
  FileMeta meta_;
  Volume volume_;
  DisplacementField disp_;  //!< deformed-geometry warp source (empty = none)

  // --- UI / selection state -------------------------------------------
  std::string path_buf_;
  std::string status_ = "Open a muGrid NetCDF (.nc) file to begin.";
  char path_edit_[1024] = {0};          //!< File-panel path input buffer
  char png_path_[1024] = "mueye.png";   //!< Render-panel PNG file name buffer
  bool layout_initialized_{false};      //!< default docking layout built once
  bool has_file_{false};
  int field_index_{0};
  int frame_{0};
  int component_{0};
  Scalarize scalarize_{Scalarize::Component};
  int loaded_field_{-1};       //!< field index of the volume currently loaded
  Scalarize loaded_scalarize_{Scalarize::Component};  //!< its scalarization
  double last_load_ms_{0.0};   //!< duration of the most recent volume load
  bool frame_pending_{false};  //!< frame changed while scrubbing; load on release

  //! Macroscopic deformation gradient F (row-major 3x3; identity == orthogonal
  //! cell). Populated from the file's deformation_gradient/average_strain
  //! attribute on load, editable in the Cell panel. The rendered cell is
  //! C = F * diag(box_extent).
  float F_[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  bool deformation_from_file_{false};  //!< F_ came from the file (for the UI)
  //! The user edited or reset F by hand. While set, the file's per-frame
  //! applied deformation gradient no longer overwrites F_ on frame changes;
  //! "Use file's F" in the Cell panel clears it. Reset on every file open.
  bool F_user_override_{false};

  /** Restore F_ from the file (per-frame tensor if present, else the global
   *  attribute, else identity) and clear the user override. */
  void adopt_file_deformation();

  //! Deformed-geometry warp: index into meta_.fields of the displacement field
  //! (-1 = none, the default: no deformation). Only fields with exactly
  //! spatial_dim components are eligible. warp_scale_ is a user gain on the
  //! displacement (world units); warp_iters_ is the fixed-point iteration count.
  int disp_field_index_{-1};
  float warp_scale_{1.0f};
  int warp_iters_{6};
  bool disp_dirty_{true};  //!< re-upload displacement to the backend before render

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

  /** True while the deformed-geometry warp is active (a displacement field is
   *  selected and loaded). */
  bool warping() const { return disp_field_index_ >= 0 && !disp_.empty(); }

  /** Replica count actually applied along @p axis: 1 unless periodic tiling is
   *  enabled, and always 1 while warping (the deformed body does not tile).
   *  The single source of truth for the render params, the box overlay and
   *  the camera framing, so they cannot disagree. */
  int rep(int axis) const {
    return (periodic_ && !warping()) ? replicas_[axis] : 1;
  }

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
  //! Progressive rendering: while the mouse is held (orbit, pan, slider drag)
  //! render at a coarser resolution chosen so a frame takes about
  //! kInteractiveTargetMs, then re-render at the user's quality on release.
  bool adaptive_quality_{true};
  static constexpr double kInteractiveTargetMs = 33.0;
  bool needs_render_{true};
  double last_render_ms_{0.0};
  double last_full_ms_{0.0};   //!< duration of the last non-interactive render
  int last_render_w_{0};
  int last_render_h_{0};
  int last_downscale_{1};      //!< downscale the last render actually used
};

}  // namespace mueye

#endif  // MUEYE_APP_HH_
