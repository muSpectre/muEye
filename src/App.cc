/**
 * @file   App.cc
 *
 * @brief  Data loading and render orchestration (UI layout lives in panels.cc).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "App.hh"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

#include "io/PngWriter.hh"

namespace mueye {

namespace {

// Pick the field to show by default when a file is first opened. Priority 1:
// density fields, whose variable name typically contains "rho" or "density"
// (case-insensitive). Falls back to the first field if none match.
int default_field_index(const std::vector<FieldInfo> &fields) {
  for (std::size_t i = 0; i < fields.size(); ++i) {
    std::string lower = fields[i].name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lower.find("density") != std::string::npos ||
        lower.find("rho") != std::string::npos)
      return static_cast<int>(i);
  }
  return 0;
}

}  // namespace

App::App() {
  backends_ = enumerate_backends();

  // Prefer a GPU backend (Metal / CUDA / HIP) when one is available and can be
  // constructed; otherwise fall back to the CPU backend.
  for (const BackendInfo &bi : backends_) {
    if (bi.backend == Backend::CPU || !bi.available) continue;
    if (auto r = create_renderer(bi.backend)) {
      renderer_ = std::move(r);
      current_backend_ = bi.backend;
      break;
    }
  }
  if (!renderer_) {
    renderer_ = create_renderer(Backend::CPU);
    current_backend_ = Backend::CPU;
  }

  status_ = "Open a muGrid NetCDF (.nc) file to begin.";
}

void App::set_backend(Backend backend) {
  if (backend == current_backend_ && renderer_) return;
  auto next = create_renderer(backend);
  if (!next) {
    status_ = std::string("Backend '") + to_string(backend) +
              "' is unavailable; keeping " + to_string(current_backend_) + ".";
    return;
  }
  renderer_ = std::move(next);
  current_backend_ = backend;
  // The new backend has no data yet; re-upload on the next render.
  volume_dirty_ = true;
  tf_dirty_ = true;
  disp_dirty_ = true;  // the new backend has no displacement uploaded yet
  needs_render_ = true;
  status_ = std::string("Switched to the ") + renderer_->name() + " backend.";
}

void App::open_path(const std::string &path) {
  // Introspect into a temporary: if the new file cannot be opened, the
  // currently loaded dataset (metadata, volume, displacement, renderer data)
  // stays fully intact and only the error is reported. Previously the failed
  // metadata replaced meta_ while volume_ kept rendering the old file.
  FileMeta meta = loader_.open(path);
  if (!meta.valid) {
    status_ = "Failed to open '" + path + "': " + meta.error;
    return;
  }
  meta_ = std::move(meta);
  has_file_ = true;
  path_buf_ = path;
  // Mirror into the File panel's edit buffer (e.g. for a command-line load).
  std::snprintf(path_edit_, sizeof(path_edit_), "%s", path.c_str());
  // Default the render output next to the data file: <data-dir>/<stem>.png.
  {
    std::filesystem::path dp{path};
    std::string stem = dp.stem().string();
    std::filesystem::path png =
        dp.parent_path() / ((stem.empty() ? "mueye" : stem) + ".png");
    std::snprintf(png_path_, sizeof(png_path_), "%s", png.string().c_str());
  }
  field_index_ = default_field_index(meta_.fields);
  loaded_field_ = -1;  // a new file always re-derives the default iso level
  frame_ = 0;
  component_ = 0;
  // Start undeformed: no displacement field selected until the user picks one.
  disp_field_index_ = -1;
  disp_ = DisplacementField{};
  disp_dirty_ = true;
  // The periodic shift is per-file view state; the grid may have changed.
  for (int &d : shift_) d = 0;
  // Adopt the file's deformation gradient (identity if the file has none). A
  // per-frame applied_deformation_gradient, if present, supersedes it for the
  // current frame. A new file always discards a manual override.
  adopt_file_deformation();  // frame_ == 0 here
  status_ = "Loaded '" + path + "' (" + std::to_string(meta_.nx) + "x" +
            std::to_string(meta_.ny) + "x" + std::to_string(meta_.nz) + ", " +
            std::to_string(meta_.nb_frames) + " frame(s), " +
            std::to_string(meta_.fields.size()) + " field(s)" +
            (meta_.has_deformation ? ", deformed cell" : "") + ").";
  // Load the volume first so its grid dimensions are known, then aim the camera
  // at the centre of the (possibly non-cubic / sheared) cell. Framing before the
  // load would use stale/empty dimensions and put the orbit pivot at the origin
  // (a corner of the cell) instead of its centre.
  reload_volume();
  frame_view(true);
}

void App::frame_view(bool reset_angles) {
  Vec3 center;
  float extent;
  cell_bounds(center, extent);
  if (reset_angles) {
    camera_.frame_aabb(center, extent);
    // A 2D field is a single slice in the x-y plane; look at it face-on
    // rather than as an obliquely tilted slab.
    if (meta_.spatial_dim == 2) camera_.set_face_on();
  } else {
    camera_.retarget_aabb(center, extent);
  }
  needs_render_ = true;
}

void App::reload_volume() {
  if (!has_file_ || meta_.fields.empty()) return;
  if (field_index_ < 0) field_index_ = 0;
  if (field_index_ >= static_cast<int>(meta_.fields.size()))
    field_index_ = static_cast<int>(meta_.fields.size()) - 1;
  if (frame_ < 0) frame_ = 0;
  if (frame_ >= meta_.nb_frames) frame_ = meta_.nb_frames - 1;

  const FieldInfo &fi = meta_.fields[field_index_];
  // Clamp the component into the new field's range *and write it back*, so
  // the Component slider never displays a value outside its own range after
  // switching from a field with more components.
  if (component_ >= fi.nb_components) component_ = fi.nb_components - 1;
  if (component_ < 0) component_ = 0;

  auto t0 = std::chrono::high_resolution_clock::now();
  std::string err = loader_.load(path_buf_, meta_, fi, frame_, scalarize_,
                                 component_, volume_);
  for (int &d : vol_shift_) d = 0;  // freshly loaded data is unshifted
  last_load_ms_ = std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - t0)
                      .count();
  if (!err.empty()) {
    status_ = err;
    info_ = "(load failed)";
    volume_ = Volume{};
    // The CPU backend borrows volume_.data; it was just freed, so the backend
    // must be re-pointed before it renders again.
    volume_dirty_ = true;
    needs_render_ = true;
    return;
  }
  // Iso level: start at the data midpoint when a *different quantity* is shown
  // (new field or scalarization); otherwise keep the user's level across frame
  // and component changes, clamping it into the new range so scrubbing the
  // frame slider does not silently discard a hand-tuned isosurface.
  const bool new_quantity =
      field_index_ != loaded_field_ || scalarize_ != loaded_scalarize_;
  if (new_quantity || !(iso_value_ >= data_min() && iso_value_ <= data_max())) {
    iso_value_ = new_quantity ? 0.5f * (data_min() + data_max())
                              : std::clamp(iso_value_, data_min(), data_max());
  }
  loaded_field_ = field_index_;
  loaded_scalarize_ = scalarize_;
  info_ = "Field '" + fi.name + "' frame " + std::to_string(frame_) +
          "  range [" + std::to_string(volume_.vmin) + ", " +
          std::to_string(volume_.vmax) + "]";
  if (volume_.nb_nonfinite > 0)
    info_ += "  (" + std::to_string(volume_.nb_nonfinite) +
             " NaN/Inf voxel(s) shown as the minimum)";
  volume_dirty_ = true;  // backend must re-upload the new volume
  needs_render_ = true;

  // If a displacement field is selected, refresh it for the current frame too.
  if (disp_field_index_ >= 0) reload_displacement();
  apply_shift();
}

void App::reload_displacement() {
  disp_ = DisplacementField{};
  if (has_file_ && disp_field_index_ >= 0 &&
      disp_field_index_ < static_cast<int>(meta_.fields.size())) {
    std::string err = loader_.load_displacement(
        path_buf_, meta_, meta_.fields[disp_field_index_], frame_, disp_);
    if (!err.empty()) {
      status_ = err;
      disp_ = DisplacementField{};
      disp_field_index_ = -1;
    }
  }
  for (int &d : disp_shift_) d = 0;  // freshly loaded data is unshifted
  apply_shift();
  disp_dirty_ = true;  // backend must (re-)upload or clear the displacement
  needs_render_ = true;
}

void App::apply_shift() {
  auto differs = [](const int *a, const int *b) {
    return a[0] != b[0] || a[1] != b[1] || a[2] != b[2];
  };
  if (differs(shift_, vol_shift_)) {
    volume_.roll(shift_[0] - vol_shift_[0], shift_[1] - vol_shift_[1],
                 shift_[2] - vol_shift_[2]);
    for (int a = 0; a < 3; ++a) vol_shift_[a] = shift_[a];
    volume_dirty_ = true;
    needs_render_ = true;
  }
  if (differs(shift_, disp_shift_)) {
    disp_.roll(shift_[0] - disp_shift_[0], shift_[1] - disp_shift_[1],
               shift_[2] - disp_shift_[2]);
    for (int a = 0; a < 3; ++a) disp_shift_[a] = shift_[a];
    disp_dirty_ = true;
    needs_render_ = true;
  }
}

// Upload data to the backend only when it changed (cheap for CPU, avoids a
// device round-trip every frame for Metal/CUDA/HIP).
void App::sync_renderer_data() {
  bool uploaded = false;
  if (volume_dirty_) {
    renderer_->set_volume(volume_.data.data(), volume_.nx, volume_.ny,
                          volume_.nz);
    const BrickGrid &b = volume_.bricks;
    renderer_->set_brick_grid(b.empty() ? nullptr : b.bmin.data(),
                              b.empty() ? nullptr : b.bmax.data(), b.bx, b.by,
                              b.bz);
    volume_dirty_ = false;
    uploaded = true;
  }
  if (tf_dirty_) {
    renderer_->set_transfer_function(tf_.data(), tf_.size());
    tf_dirty_ = false;
    uploaded = true;
  }
  if (disp_dirty_) {
    renderer_->set_displacement(disp_.empty() ? nullptr : disp_.data.data(),
                                disp_.nx, disp_.ny, disp_.nz);
    disp_dirty_ = false;
    uploaded = true;
  }
  // Surface upload failures (e.g. volume too large for device memory) instead
  // of silently rendering a blank viewport.
  if (uploaded) {
    const char *err = renderer_->last_error();
    if (err != nullptr && err[0] != '\0') {
      status_ = std::string(renderer_->name()) + ": " + err;
    }
  }
}

Vec3 App::reference_box() const {
  if (meta_.has_domain_lengths) {
    double lx = meta_.domain_lengths[0];
    double ly = meta_.domain_lengths[1];
    double lz = meta_.spatial_dim == 3 ? meta_.domain_lengths[2] : 0.0;
    double m = std::max(lx, std::max(ly, lz));
    if (m <= 0.0) m = 1.0;
    Vec3 b{static_cast<float>(lx / m), static_cast<float>(ly / m), 0.0f};
    if (meta_.spatial_dim == 3) {
      b.z = static_cast<float>(lz / m);
    } else {
      // 2D: a single-slice slab; keep it one voxel thick along the longest
      // in-plane direction (cosmetic, matches the grid-derived convention).
      int mn = std::max(volume_.nx, volume_.ny);
      b.z = std::max(b.x, b.y) / static_cast<float>(mn > 0 ? mn : 1);
    }
    return b;
  }
  return box_extent(volume_.nx, volume_.ny, volume_.nz);
}

void App::sync_frame_deformation() {
  // A hand-edited F wins over the file's per-frame tensor until the user asks
  // for the file's value back; otherwise scrubbing frames would silently undo
  // an edit or a "Reset to identity".
  if (F_user_override_) return;
  if (frame_ >= 0 && frame_ < static_cast<int>(meta_.applied_F.size())) {
    for (int i = 0; i < 9; ++i)
      F_[i] = static_cast<float>(meta_.applied_F[frame_][i]);
    deformation_from_file_ = true;
    needs_render_ = true;
  }
}

void App::adopt_file_deformation() {
  F_user_override_ = false;
  for (int i = 0; i < 9; ++i) F_[i] = static_cast<float>(meta_.F[i]);
  deformation_from_file_ = meta_.has_deformation;
  sync_frame_deformation();
  needs_render_ = true;
}

Mat3 App::world_cell() const {
  Vec3 box = reference_box();
  Mat3 F{{F_[0], F_[1], F_[2], F_[3], F_[4], F_[5], F_[6], F_[7], F_[8]}};
  // C = F * diag(box): scale the reference box by the deformation gradient.
  return mat3_matmul(F, mat3_diag(box));
}

void App::cell_bounds(Vec3 &center, float &extent) const {
  Mat3 C = world_cell();
  float rx = static_cast<float>(rep(0));
  float ry = static_cast<float>(rep(1));
  float rz = static_cast<float>(rep(2));
  Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
  for (int i = 0; i < 8; ++i) {
    Vec3 f{(i & 1) ? rx : 0.0f, (i & 2) ? ry : 0.0f, (i & 4) ? rz : 0.0f};
    Vec3 w = mat3_mul(C, f);
    lo.x = std::min(lo.x, w.x); hi.x = std::max(hi.x, w.x);
    lo.y = std::min(lo.y, w.y); hi.y = std::max(hi.y, w.y);
    lo.z = std::min(lo.z, w.z); hi.z = std::max(hi.z, w.z);
  }
  center = (lo + hi) * 0.5f;
  extent = std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
}

RenderParams App::make_render_params() const {
  RenderParams p;
  p.nx = volume_.nx;
  p.ny = volume_.ny;
  p.nz = volume_.nz;
  p.box = reference_box();
  // World->fractional map: inverse of the (sheared) cell matrix C = F*diag(box).
  p.inv_cell = mat3_inverse(world_cell());
  p.rep_x = rep(0);
  p.rep_y = rep(1);
  p.rep_z = rep(2);
  // step_ is in voxels; the box's longest axis is 1 world unit, so one voxel
  // along that axis is 1/max(nx,ny,nz) in world units.
  int max_dim = volume_.nx;
  if (volume_.ny > max_dim) max_dim = volume_.ny;
  if (volume_.nz > max_dim) max_dim = volume_.nz;
  p.step = step_ / static_cast<float>(max_dim > 0 ? max_dim : 1);
  p.data_min = data_min();
  p.data_max = data_max();
  p.lut_size = tf_.size();
  p.density_scale = density_scale_;
  p.iso_value = iso_value_;
  p.mode = mode_;
  p.bg = Vec3{bg_[0], bg_[1], bg_[2]};
  // Empty-space skipping threshold, from the transfer function's transparent
  // band (only the DVR path uses it; the iso path skips by iso level).
  p.skip_below = tf_.skip_below(p.data_min, p.data_max);

  // Deformed-geometry warp. Active only when a displacement field is loaded.
  // Periodic tiling is unsupported while warping (the deformed body no longer
  // tiles trivially); rep() already returns 1 per axis in that case, for the
  // render, the box overlay and the camera framing alike.
  const bool warp = warping();
  p.warp_enabled = warp ? 1 : 0;
  p.warp_scale = warp_scale_;
  p.warp_iters = warp_iters_;
  if (warp) {
    // World AABB of the deformed body: the single-cell parallelepiped expanded
    // by the largest world-space displacement on every side.
    Mat3 C = world_cell();
    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < 8; ++i) {
      Vec3 w = mat3_mul(C, Vec3{(i & 1) ? 1.0f : 0.0f, (i & 2) ? 1.0f : 0.0f,
                                (i & 4) ? 1.0f : 0.0f});
      lo.x = std::min(lo.x, w.x); hi.x = std::max(hi.x, w.x);
      lo.y = std::min(lo.y, w.y); hi.y = std::max(hi.y, w.y);
      lo.z = std::min(lo.z, w.z); hi.z = std::max(hi.z, w.z);
    }
    float f = 1.0f / static_cast<float>(max_dim > 0 ? max_dim : 1);
    float margin = warp_scale_ * f * disp_.max_mag;
    p.warp_lo = Vec3{lo.x - margin, lo.y - margin, lo.z - margin};
    p.warp_hi = Vec3{hi.x + margin, hi.y + margin, hi.z + margin};
  }
  return p;
}

std::filesystem::path App::resolve_output(const std::string &path) const {
  std::filesystem::path out{path};
  if (out.is_relative() && !path_buf_.empty()) {
    out = std::filesystem::path(path_buf_).parent_path() / out;
  }
  return out;
}

void App::save_png(const std::string &path) {
  if (path.empty()) {
    status_ = "Enter a file name to save the PNG.";
    return;
  }
  if (volume_.empty() || !renderer_) {
    status_ = "Nothing to save — load a file first.";
    return;
  }
  const std::string out_path = resolve_output(path).string();
  // Full viewport resolution regardless of the interactive downscale (the
  // last render may have been a coarse interactive one).
  int w = last_render_w_ > 0 ? last_render_w_ * last_downscale_ : 1280;
  int h = last_render_h_ > 0 ? last_render_h_ * last_downscale_ : 720;

  // Always go through the host-framebuffer path: with a zero-copy backend
  // (render_to_gl) the pixels never reach fb_, so render afresh either way.
  sync_renderer_data();
  renderer_->set_num_threads(cpu_threads_);
  Framebuffer fb;
  fb.resize(w, h);
  renderer_->render(make_render_params(),
                    camera_.to_camera(static_cast<float>(w) / h), fb);

  if (write_png(out_path, fb)) {
    status_ = "Saved '" + out_path + "' (" + std::to_string(w) + "x" +
              std::to_string(h) + ").";
  } else {
    status_ = "Failed to write '" + out_path + "'.";
  }
}

int App::effective_downscale(bool interacting) const {
  int d = render_downscale_ < 1 ? 1 : render_downscale_;
  if (!interacting || !adaptive_quality_ || last_full_ms_ <= 0.0) return d;
  // Render time scales with the pixel count, so the extra factor needed to
  // reach the target is sqrt(t_full / t_target); never coarser than 1/8.
  double f = std::sqrt(last_full_ms_ / kInteractiveTargetMs);
  int extra = static_cast<int>(std::ceil(f));
  if (extra < 1) extra = 1;
  if (extra > 8) extra = 8;
  return d * extra;
}

void App::render(int width, int height, int downscale) {
  if (width <= 0 || height <= 0) return;
  if (downscale < 1) downscale = 1;
  int rw = width / downscale;
  int rh = height / downscale;
  if (rw < 1) rw = 1;
  if (rh < 1) rh = 1;
  last_downscale_ = downscale;

  if (volume_.empty() || !renderer_) {
    // Clear to background.
    fb_.resize(rw, rh);
    for (std::size_t i = 0; i < fb_.rgba.size(); i += 4) {
      fb_.rgba[i + 0] = static_cast<std::uint8_t>(bg_[0] * 255);
      fb_.rgba[i + 1] = static_cast<std::uint8_t>(bg_[1] * 255);
      fb_.rgba[i + 2] = static_cast<std::uint8_t>(bg_[2] * 255);
      fb_.rgba[i + 3] = 255;
    }
    texture_.upload(fb_);
    last_render_w_ = rw;
    last_render_h_ = rh;
    return;
  }

  sync_renderer_data();
  renderer_->set_num_threads(cpu_threads_);  // no-op for GPU backends

  RenderParams p = make_render_params();
  Camera cam = camera_.to_camera(static_cast<float>(rw) / rh);

  auto t0 = std::chrono::high_resolution_clock::now();
  // Prefer a backend's zero-copy path straight into the GL texture (GPU
  // backends that share memory with GL); otherwise render to the host
  // Framebuffer and upload it.
  unsigned int gl_tex = texture_.ensure(rw, rh);
  if (!renderer_->render_to_gl(p, cam, gl_tex, rw, rh)) {
    // Host-framebuffer fallback; fb_ is only (re)sized on this path so the
    // zero-copy path does not keep a dead host-side copy around.
    fb_.resize(rw, rh);
    renderer_->render(p, cam, fb_);
    texture_.upload(fb_);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  last_render_ms_ =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  // Only a render at the user's own quality setting calibrates the adaptive
  // factor; scale a coarse frame's time back up would compound rounding.
  if (downscale == render_downscale_) last_full_ms_ = last_render_ms_;

  last_render_w_ = rw;
  last_render_h_ = rh;
}

}  // namespace mueye
