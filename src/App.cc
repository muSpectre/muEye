/**
 * @file   App.cc
 *
 * @brief  Data loading and render orchestration (UI layout lives in panels.cc).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "App.hh"

#include <chrono>

namespace mueye {

App::App() {
  backends_ = enumerate_backends();
  renderer_ = create_renderer(Backend::CPU);
  current_backend_ = Backend::CPU;
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
  field_index_ = 0;
  frame_ = 0;
  component_ = 0;
  status_ = "Loaded '" + path + "' (" + std::to_string(meta_.nx) + "x" +
            std::to_string(meta_.ny) + "x" + std::to_string(meta_.nz) + ", " +
            std::to_string(meta_.nb_frames) + " frame(s), " +
            std::to_string(meta_.fields.size()) + " field(s)).";
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

  std::string err = loader_.load(path_buf_, meta_, fi, frame_, scalarize_, comp,
                                 volume_);
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

void App::render(int width, int height) {
  if (width <= 0 || height <= 0) return;
  int rw = width / render_downscale_;
  int rh = height / render_downscale_;
  if (rw < 1) rw = 1;
  if (rh < 1) rh = 1;

  fb_.resize(rw, rh);

  if (volume_.empty() || !renderer_) {
    // Clear to background.
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

  // Upload data to the backend only when it changed (cheap for CPU, avoids a
  // device round-trip every frame for Metal/CUDA/HIP).
  if (volume_dirty_) {
    renderer_->set_volume(volume_.data.data(), volume_.nx, volume_.ny,
                          volume_.nz);
    volume_dirty_ = false;
  }
  if (tf_dirty_) {
    renderer_->set_transfer_function(tf_.data(), tf_.size());
    tf_dirty_ = false;
  }
  renderer_->set_num_threads(cpu_threads_);  // no-op for GPU backends

  RenderParams p;
  p.nx = volume_.nx;
  p.ny = volume_.ny;
  p.nz = volume_.nz;
  // step_ is in voxels; the volume spans the unit box, so one voxel along the
  // longest axis is 1/max(nx,ny,nz) in world units.
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

  Camera cam = camera_.to_camera(static_cast<float>(rw) / rh);

  auto t0 = std::chrono::high_resolution_clock::now();
  renderer_->render(p, cam, fb_);
  auto t1 = std::chrono::high_resolution_clock::now();
  last_render_ms_ =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  texture_.upload(fb_);
  last_render_w_ = rw;
  last_render_h_ = rh;
}

}  // namespace mueye
