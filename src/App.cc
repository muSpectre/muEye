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
#include <cstdio>

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

  status_ = std::string("Renderer: ") + renderer_->name() +
            "  —  open a muGrid NetCDF (.nc) file to begin.";
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
  needs_render_ = true;
  status_ = std::string("Renderer: ") + renderer_->name();
}

void App::open_path(const std::string &path) {
  meta_ = loader_.open(path);
  if (!meta_.valid) {
    has_file_ = false;
    status_ = "Failed to open '" + path + "': " + meta_.error;
    return;
  }
  has_file_ = true;
  path_buf_ = path;
  // Mirror into the File panel's edit buffer (e.g. for a command-line load).
  std::snprintf(path_edit_, sizeof(path_edit_), "%s", path.c_str());
  field_index_ = default_field_index(meta_.fields);
  frame_ = 0;
  component_ = 0;
  status_ = "Loaded '" + path + "' (" + std::to_string(meta_.nx) + "x" +
            std::to_string(meta_.ny) + "x" + std::to_string(meta_.nz) + ", " +
            std::to_string(meta_.nb_frames) + " frame(s), " +
            std::to_string(meta_.fields.size()) + " field(s)).";
  // Aim the camera at the centre of the (possibly non-cubic / planar) box.
  camera_.frame_box(box_extent(meta_.nx, meta_.ny, meta_.nz));
  reload_volume();
}

void App::reload_volume() {
  if (!has_file_ || meta_.fields.empty()) return;
  if (field_index_ < 0) field_index_ = 0;
  if (field_index_ >= static_cast<int>(meta_.fields.size()))
    field_index_ = static_cast<int>(meta_.fields.size()) - 1;
  if (frame_ < 0) frame_ = 0;
  if (frame_ >= meta_.nb_frames) frame_ = meta_.nb_frames - 1;

  const FieldInfo &fi = meta_.fields[field_index_];
  int comp = component_;
  if (comp >= fi.nb_components) comp = fi.nb_components - 1;

  auto t0 = std::chrono::high_resolution_clock::now();
  std::string err = loader_.load(path_buf_, meta_, fi, frame_, scalarize_, comp,
                                 volume_);
  last_load_ms_ = std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - t0)
                      .count();
  if (!err.empty()) {
    status_ = err;
    volume_ = Volume{};
    return;
  }
  // Sensible default iso value at the data midpoint on (re)load.
  iso_value_ = 0.5f * (volume_.vmin + volume_.vmax);
  status_ = "Field '" + fi.name + "' frame " + std::to_string(frame_) +
            "  range [" + std::to_string(volume_.vmin) + ", " +
            std::to_string(volume_.vmax) + "]";
  volume_dirty_ = true;  // backend must re-upload the new volume
  needs_render_ = true;
}

// Upload data to the backend only when it changed (cheap for CPU, avoids a
// device round-trip every frame for Metal/CUDA/HIP).
void App::sync_renderer_data() {
  bool uploaded = false;
  if (volume_dirty_) {
    renderer_->set_volume(volume_.data.data(), volume_.nx, volume_.ny,
                          volume_.nz);
    volume_dirty_ = false;
    uploaded = true;
  }
  if (tf_dirty_) {
    renderer_->set_transfer_function(tf_.data(), tf_.size());
    tf_dirty_ = false;
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

RenderParams App::make_render_params() const {
  RenderParams p;
  p.nx = volume_.nx;
  p.ny = volume_.ny;
  p.nz = volume_.nz;
  p.box = box_extent(volume_.nx, volume_.ny, volume_.nz);
  p.rep_x = rep(0);
  p.rep_y = rep(1);
  p.rep_z = rep(2);
  // step_ is in voxels; the box's longest axis is 1 world unit, so one voxel
  // along that axis is 1/max(nx,ny,nz) in world units.
  int max_dim = volume_.nx;
  if (volume_.ny > max_dim) max_dim = volume_.ny;
  if (volume_.nz > max_dim) max_dim = volume_.nz;
  p.step = step_ / static_cast<float>(max_dim > 0 ? max_dim : 1);
  p.data_min = volume_.vmin;
  p.data_max = volume_.vmax;
  p.lut_size = tf_.size();
  p.density_scale = density_scale_;
  p.iso_value = iso_value_;
  p.mode = mode_;
  p.bg = Vec3{bg_[0], bg_[1], bg_[2]};
  return p;
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
  // Full viewport resolution regardless of the interactive downscale.
  int w = last_render_w_ > 0 ? last_render_w_ * render_downscale_ : 1280;
  int h = last_render_h_ > 0 ? last_render_h_ * render_downscale_ : 720;

  // Always go through the host-framebuffer path: with a zero-copy backend
  // (render_to_gl) the pixels never reach fb_, so render afresh either way.
  sync_renderer_data();
  renderer_->set_num_threads(cpu_threads_);
  Framebuffer fb;
  fb.resize(w, h);
  renderer_->render(make_render_params(),
                    camera_.to_camera(static_cast<float>(w) / h), fb);

  if (write_png(path, fb)) {
    status_ = "Saved '" + path + "' (" + std::to_string(w) + "x" +
              std::to_string(h) + ").";
  } else {
    status_ = "Failed to write '" + path + "'.";
  }
}

void App::render(int width, int height) {
  if (width <= 0 || height <= 0) return;
  int rw = width / render_downscale_;
  int rh = height / render_downscale_;
  if (rw < 1) rw = 1;
  if (rh < 1) rh = 1;

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

  last_render_w_ = rw;
  last_render_h_ = rh;
}

}  // namespace mueye
