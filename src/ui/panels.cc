/**
 * @file   panels.cc
 *
 * @brief  ImGui window layout for muEye (implements App::draw_ui).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "App.hh"
#include "imgui.h"
#include "imgui_internal.h"  // DockBuilder* for the default layout
#include "portable-file-dialogs.h"  // native file chooser (third_party/)

namespace mueye {

namespace {

// Project a world-space point through the pinhole camera into pixel
// coordinates on the viewport image (the inverse of render_core's
// primary_ray_dir). Callers must ensure the point is in front of the camera.
ImVec2 project_to_image(const Camera &cam, const Vec3 &pt, const ImVec2 &origin,
                        float w, float h) {
  Vec3 d = pt - cam.eye;
  float z = dot(d, cam.forward);
  float u = 0.5f * (dot(d, cam.right) / (z * cam.tan_half_fov * cam.aspect) +
                    1.0f);
  float v = 0.5f * (1.0f - dot(d, cam.up) / (z * cam.tan_half_fov));
  return ImVec2(origin.x + u * w, origin.y + v * h);
}

// Draw one box edge as an overlay line, clipping it at the camera near plane
// so edges partially behind the eye still draw correctly.
void draw_box_edge(ImDrawList *dl, const Camera &cam, Vec3 a, Vec3 b,
                   const ImVec2 &origin, float w, float h, ImU32 col) {
  const float z_near = 1e-3f;
  float za = dot(a - cam.eye, cam.forward);
  float zb = dot(b - cam.eye, cam.forward);
  if (za < z_near && zb < z_near) return;
  if (za < z_near) {
    a = a + (b - a) * ((z_near - za) / (zb - za));
  } else if (zb < z_near) {
    b = b + (a - b) * ((z_near - zb) / (za - zb));
  }
  dl->AddLine(project_to_image(cam, a, origin, w, h),
              project_to_image(cam, b, origin, w, h), col, 1.5f);
}

}  // namespace

// Arrange the panels into a default layout: a left control column (grouped into
// three stacked tab-nodes) and a large viewport filling the rest. Called once
// when there is no docking layout yet, so a saved imgui.ini still wins.
static void build_default_layout(ImGuiID dockspace_id) {
  ImGui::DockBuilderRemoveNode(dockspace_id);
  ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

  ImGuiID center = dockspace_id;
  // A slim status bar across the bottom (no tab bar, so it reads as a bar).
  ImGuiID bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.04f,
                                               nullptr, &center);
  if (ImGuiDockNode *bn = ImGui::DockBuilderGetNode(bottom))
    bn->LocalFlags |= ImGuiDockNodeFlags_NoTabBar |
                      ImGuiDockNodeFlags_NoDockingOverMe;
  ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.26f,
                                             nullptr, &center);
  ImGuiID left_rest = left;
  ImGuiID left_top = ImGui::DockBuilderSplitNode(left_rest, ImGuiDir_Up, 0.34f,
                                                 nullptr, &left_rest);
  ImGuiID left_mid = ImGui::DockBuilderSplitNode(left_rest, ImGuiDir_Up, 0.5f,
                                                 nullptr, &left_rest);
  ImGuiID left_bot = left_rest;

  ImGui::DockBuilderDockWindow("Dataset", left_top);
  ImGui::DockBuilderDockWindow("Cell", left_top);
  ImGui::DockBuilderDockWindow("Render", left_mid);
  ImGui::DockBuilderDockWindow("Transfer function", left_mid);
  ImGui::DockBuilderDockWindow("Device", left_bot);
  ImGui::DockBuilderDockWindow("Stats", left_bot);
  ImGui::DockBuilderDockWindow("Viewport", center);
  ImGui::DockBuilderDockWindow("Status", bottom);
  ImGui::DockBuilderFinish(dockspace_id);
}

void App::draw_ui() {
  // A full-window dock space so panels and the viewport can be arranged freely.
  // Called with no arguments so it compiles across docking-branch versions
  // (the dockspace-id overload was added later but keeps the same defaults).
  ImGuiID dockspace_id = ImGui::DockSpaceOverViewport();

  // First-run default layout: only build it if this dockspace has no nodes yet
  // (i.e. no imgui.ini restored a previous arrangement).
  if (!layout_initialized_) {
    layout_initialized_ = true;
    ImGuiDockNode *node = ImGui::DockBuilderGetNode(dockspace_id);
    if (node == nullptr || node->IsLeafNode()) {
      build_default_layout(dockspace_id);
    }
  }

  // ----------------------------------------------------------------- Dataset
  // File loader on top, then (once a file is open) the field / frame / scalar
  // controls. This single panel replaces the former separate "muEye" tab.
  ImGui::Begin("Dataset");
  {
    // File path box (no trailing label) + Load / Browse.
    ImGui::InputText("##File", path_edit_, sizeof(path_edit_));
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
      open_path(path_edit_);
    }
    ImGui::SameLine();
    // Probed once: on Linux, available() runs subprocesses to look for a
    // dialog backend (zenity/kdialog/...), so it must not run every frame.
    // An immutable capability, not UI state — a static const is fine.
    static const bool can_browse = pfd::settings::available();
    ImGui::BeginDisabled(!can_browse);
    // Native file chooser; blocks the UI loop while the modal dialog is open.
    if (ImGui::Button("Browse...")) {
      // Default to the *.nc filter. On macOS (pfd's osascript backend) any "*"
      // pattern disables filtering altogether, so the all-files escape hatch is
      // omitted there; on Windows/Linux it is kept as a switchable second entry
      // (the *.nc filter stays first, i.e. the default).
      std::vector<std::string> filters{"NetCDF files (*.nc)", "*.nc"};
#ifndef __APPLE__
      filters.push_back("All files");
      filters.push_back("*");
#endif
      // Open in the current file's *directory*. Passing the file path itself as
      // the default location makes the macOS (osascript) dialog error out and
      // silently fail to reopen after a file has been loaded, so derive the
      // parent directory instead.
      std::string start_dir;
      try {
        std::filesystem::path p{path_edit_};
        start_dir = (p.has_filename() ? p.parent_path() : p).string();
      } catch (...) {
        start_dir.clear();
      }
      auto sel = pfd::open_file("Open muGrid NetCDF file", start_dir, filters)
                     .result();
      if (!sel.empty()) {
        // Show the choice in the field even if the load then fails.
        std::snprintf(path_edit_, sizeof(path_edit_), "%s",
                      sel.front().c_str());
        open_path(sel.front());
      }
    }
    ImGui::EndDisabled();
    // Errors and action results go to the status bar; this panel only shows
    // the persistent description of what is loaded.

    if (has_file_ && !meta_.fields.empty()) {
      ImGui::Separator();
      bool reload = false;

      // Frame slider, directly beneath the file box.
      if (meta_.nb_frames > 1) {
        int f = frame_;
        if (ImGui::SliderInt("Frame", &f, 0, meta_.nb_frames - 1) &&
            f != frame_) {
          frame_ = f;
          // A per-frame applied deformation gradient (if the file has one)
          // updates the cell for the new frame.
          sync_frame_deformation();
          // Live-reload while dragging only when loads are quick; a slow load
          // per slider tick would freeze the UI, so defer it to release.
          if (last_load_ms_ <= 50.0) {
            reload = true;
          } else {
            frame_pending_ = true;
          }
        }
        if (frame_pending_ && ImGui::IsItemDeactivatedAfterEdit()) {
          frame_pending_ = false;
          reload = true;
        }
      }

      // Dataset info string ("Field '<name>' frame ...") just before the field
      // selector. Persistent: unlike the status bar it is never overwritten
      // by "Saved foo.png" and the like.
      ImGui::TextWrapped("%s", info_.c_str());

      std::vector<const char *> names;
      names.reserve(meta_.fields.size());
      for (auto &f : meta_.fields) names.push_back(f.name.c_str());
      reload |= ImGui::Combo("Field", &field_index_, names.data(),
                             static_cast<int>(names.size()));

      const FieldInfo &fi = meta_.fields[field_index_];
      const char *modes[] = {"Component", "Magnitude", "von Mises (tensor)",
                             "Trace (tensor)"};
      int sm = static_cast<int>(scalarize_);
      if (ImGui::Combo("Scalar", &sm, modes, IM_ARRAYSIZE(modes))) {
        scalarize_ = static_cast<Scalarize>(sm);
        reload = true;
      }
      if (scalarize_ == Scalarize::Component && fi.nb_components > 1) {
        int c = component_;
        if (ImGui::SliderInt("Component", &c, 0, fi.nb_components - 1)) {
          component_ = c;
          reload = true;
        }
        // muGrid flattens tensor components column-major (flat = row + d*col),
        // so spell out which entry the flat index denotes.
        if (int d = tensor_dim(fi.nb_components); d > 0) {
          ImGui::SameLine();
          ImGui::TextDisabled("= (%d,%d)", component_ % d, component_ / d);
        }
      }
      ImGui::Text("components: %d   sub-points: %d", fi.nb_components,
                  fi.nb_sub_pts);

      // ---- Deformed geometry from a displacement field ----
      // Eligible fields have exactly spatial_dim components (a 3-vector in 3D,
      // 2-vector in 2D). "(none)" is the default: undeformed.
      ImGui::Separator();
      std::vector<const char *> disp_names;
      std::vector<int> disp_map;  // combo entry -> field index (-1 for none)
      disp_names.push_back("(none)");
      disp_map.push_back(-1);
      for (std::size_t i = 0; i < meta_.fields.size(); ++i) {
        if (meta_.fields[i].nb_components == meta_.spatial_dim) {
          disp_names.push_back(meta_.fields[i].name.c_str());
          disp_map.push_back(static_cast<int>(i));
        }
      }
      int disp_cur = 0;
      for (std::size_t k = 0; k < disp_map.size(); ++k)
        if (disp_map[k] == disp_field_index_) disp_cur = static_cast<int>(k);
      if (ImGui::Combo("Displacement", &disp_cur, disp_names.data(),
                       static_cast<int>(disp_names.size()))) {
        disp_field_index_ = disp_map[disp_cur];
        reload_displacement();
        // Warping switches tiling off (and back on when deselected), so the
        // framed extent changes: recentre without touching the view angles.
        if (periodic_ && !volume_.empty()) frame_view(false);
      }
      if (disp_field_index_ >= 0) {
        if (ImGui::DragFloat("Warp scale", &warp_scale_, 0.05f, 0.0f, 1.0e6f,
                             "%.3f"))
          needs_render_ = true;
        if (ImGui::SliderInt("Warp iters", &warp_iters_, 1, 16))
          needs_render_ = true;
        if (periodic_)
          ImGui::TextDisabled("(periodic tiling is disabled while warping)");
      }

      if (reload) reload_volume();
    }
  }
  ImGui::End();

  // ----------------------------------------------------------------- Cell
  // Deformation gradient F: shears the reference box into a (Bravais) cell.
  // Editing F rebuilds inv_cell in make_render_params, so only a re-render is
  // needed. Identity => the orthogonal box exactly as before.
  if (has_file_ && !volume_.empty()) {
    ImGui::Begin("Cell");
    const bool is_2d = meta_.spatial_dim == 2;
    ImGui::TextWrapped(
        F_user_override_
            ? "Deformation gradient F (edited by hand). C = F * box."
            : deformation_from_file_
                  ? "Deformation gradient F (read from file). C = F * box."
                  : "Deformation gradient F (identity = orthogonal). C = F * box.");
    const int dim = is_2d ? 2 : 3;
    bool changed = false;
    // Edit the leading dim x dim block row by row; the z row/col stay identity
    // for 2D data. Row-major index into F_ is 3*r + c.
    for (int r = 0; r < dim; ++r) {
      float row[3];
      for (int c = 0; c < dim; ++c) row[c] = F_[3 * r + c];
      ImGui::PushID(r);
      ImGui::SetNextItemWidth(220.0f);
      if (ImGui::InputScalarN("##Frow", ImGuiDataType_Float, row, dim, nullptr,
                              nullptr, "%.4f")) {
        for (int c = 0; c < dim; ++c) F_[3 * r + c] = row[c];
        deformation_from_file_ = false;
        F_user_override_ = true;  // stop frame changes from overwriting it
        changed = true;
      }
      ImGui::PopID();
    }
    if (ImGui::Button("Reset to identity")) {
      for (int i = 0; i < 9; ++i)
        F_[i] = (i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f;
      deformation_from_file_ = false;
      F_user_override_ = true;
      changed = true;
    }
    // Offer the file's tensor back once the user has overridden it.
    const bool file_has_F = meta_.has_deformation || !meta_.applied_F.empty();
    if (F_user_override_ && file_has_F) {
      ImGui::SameLine();
      if (ImGui::Button("Use file's F")) adopt_file_deformation();
    }
    ImGui::SameLine();
    if (ImGui::Button("Frame cell")) frame_view(true);
    if (changed) needs_render_ = true;
    ImGui::End();
  }

  // -------------------------------------------------------------- Render
  ImGui::Begin("Render");
  {
    int m = static_cast<int>(mode_);
    if (ImGui::RadioButton("DVR", &m, 0)) {
      mode_ = RenderMode::DVR;
      needs_render_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Isosurface", &m, 1)) {
      mode_ = RenderMode::Isosurface;
      needs_render_ = true;
    }

    if (ImGui::SliderFloat("Step (voxels)", &step_, 0.1f, 2.0f, "%.2f"))
      needs_render_ = true;

    if (mode_ == RenderMode::DVR) {
      // The one global opacity control (the transfer function's former
      // "Opacity" slider was a second multiplier on the same quantity).
      if (ImGui::SliderFloat("Opacity", &density_scale_, 0.05f, 5.0f, "%.2f"))
        needs_render_ = true;
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Global multiplier on the transfer function's "
                          "per-voxel opacity.");
    } else {
      float lo = volume_.empty() ? 0.f : data_min();
      float hi = volume_.empty() ? 1.f : data_max();
      if (ImGui::SliderFloat("Iso value", &iso_value_, lo, hi, "%.4g"))
        needs_render_ = true;
    }
    // Fixed value range for colours and the iso slider, so frames of a time
    // series are comparable instead of each being stretched to its own
    // min/max.
    if (ImGui::Checkbox("Lock range", &range_locked_)) {
      if (range_locked_ && !volume_.empty()) {
        range_min_ = volume_.vmin;
        range_max_ = volume_.vmax;
      }
      needs_render_ = true;
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Map colours and the iso slider to a fixed value range "
                        "instead of each frame's own minimum and maximum.");
    if (range_locked_) {
      float r[2] = {range_min_, range_max_};
      if (ImGui::InputFloat2("Range", r, "%.4g") && r[1] > r[0]) {
        range_min_ = r[0];
        range_max_ = r[1];
        needs_render_ = true;
      }
    }

    if (ImGui::ColorEdit3("Background", bg_)) needs_render_ = true;

    if (ImGui::SliderInt("Downscale", &render_downscale_, 1, 4))
      needs_render_ = true;
    ImGui::Checkbox("Adaptive quality while interacting", &adaptive_quality_);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Render coarser while the mouse is held so orbiting "
                        "stays fluid, then refine at the chosen quality on "
                        "release.");

    ImGui::Separator();
    // Drawn as a viewport overlay, so toggling needs no re-render.
    ImGui::Checkbox("Show box", &show_box_);

    bool tiling_changed = ImGui::Checkbox("Periodic images", &periodic_);
    if (periodic_) {
      tiling_changed |= ImGui::InputInt3("Replicas", replicas_);
    }
    if (tiling_changed) {
      for (int &r : replicas_) r = r < 1 ? 1 : (r > 8 ? 8 : r);
      // Keep the view direction but recentre on the tiled (possibly sheared) cell.
      if (!volume_.empty()) frame_view(false);
      needs_render_ = true;
    }

    ImGui::Separator();
    // Snapshot of the ray-traced scene (at full viewport resolution; the box
    // outline is a UI overlay and is not part of the saved image).
    ImGui::InputText("PNG file", png_path_, sizeof(png_path_));
    ImGui::BeginDisabled(volume_.empty());
    if (ImGui::Button("Save PNG")) save_png(png_path_);
    ImGui::EndDisabled();
  }
  ImGui::End();

  // --------------------------------------------------- Transfer function
  if (mode_ == RenderMode::DVR) {
    ImGui::Begin("Transfer function");
    const char *cmaps[] = {"Viridis", "Grayscale", "Cool-Warm"};
    int cm = static_cast<int>(tf_.colormap());
    if (ImGui::Combo("Colormap", &cm, cmaps, IM_ARRAYSIZE(cmaps))) {
      tf_.set_colormap(static_cast<Colormap>(cm));
      tf_dirty_ = true;
      needs_render_ = true;
    }
    if (ImGui::BeginCombo("Opacity ramp", to_string(tf_.ramp()))) {
      for (OpacityRamp r : {OpacityRamp::Ascending, OpacityRamp::Descending,
                            OpacityRamp::Symmetric}) {
        if (ImGui::Selectable(to_string(r), r == tf_.ramp()) &&
            r != tf_.ramp()) {
          tf_.set_ramp(r);
          tf_dirty_ = true;
          needs_render_ = true;
        }
      }
      ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Which values are transparent. Use Symmetric with a "
                        "diverging colormap (Cool-Warm) so both signs show.");
    float g = tf_.opacity_gamma();
    if (ImGui::SliderFloat("Opacity gamma", &g, 0.2f, 4.0f, "%.2f")) {
      tf_.set_opacity_gamma(g);
      tf_dirty_ = true;
      needs_render_ = true;
    }
    float cut = tf_.opacity_cutoff();
    if (ImGui::SliderFloat("Cutoff", &cut, 0.0f, 0.9f, "%.2f")) {
      tf_.set_opacity_cutoff(cut);
      tf_dirty_ = true;
      needs_render_ = true;
    }
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Values below this fraction of the range are fully "
                        "transparent (and skipped by the ray marcher).");

    // Preview: the colormap drawn *with its opacity* over a checkerboard, so
    // the transparent band and the ramp are visible, plus the alpha curve.
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    float h = 48.0f;
    const float cs = 8.0f;  // checker size
    for (int cy = 0; cy * cs < h; ++cy)
      for (int cx = 0; cx * cs < w; ++cx) {
        ImU32 cc = ((cx + cy) & 1) ? IM_COL32(150, 150, 150, 255)
                                   : IM_COL32(215, 215, 215, 255);
        dl->AddRectFilled(
            ImVec2(p0.x + cx * cs, p0.y + cy * cs),
            ImVec2(std::min(p0.x + (cx + 1) * cs, p0.x + w),
                   std::min(p0.y + (cy + 1) * cs, p0.y + h)),
            cc);
      }
    const int n = tf_.size();
    const Vec4 *lut = tf_.data();
    const int steps = 128;
    std::vector<ImVec2> curve;
    curve.reserve(steps + 1);
    for (int i = 0; i <= steps; ++i) {
      float t0 = static_cast<float>(i) / steps;
      Vec4 c = lut[std::min(n - 1, static_cast<int>(t0 * (n - 1)))];
      if (i < steps) {
        float t1 = static_cast<float>(i + 1) / steps;
        ImU32 col = IM_COL32(static_cast<int>(c.x * 255),
                             static_cast<int>(c.y * 255),
                             static_cast<int>(c.z * 255),
                             static_cast<int>(c.w * 255));
        dl->AddRectFilled(ImVec2(p0.x + w * t0, p0.y),
                          ImVec2(p0.x + w * t1, p0.y + h), col);
      }
      curve.push_back(ImVec2(p0.x + w * t0, p0.y + h * (1.0f - c.w)));
    }
    dl->AddPolyline(curve.data(), static_cast<int>(curve.size()),
                    IM_COL32(20, 20, 20, 255), ImDrawFlags_None, 1.5f);
    ImGui::Dummy(ImVec2(w, h));
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Colour and opacity over the value range; the curve "
                        "is the opacity.");
    ImGui::End();
  }

  // -------------------------------------------------------------- Device
  ImGui::Begin("Device");
  {
    ImGui::TextDisabled("Rendering backend");
    for (const BackendInfo &bi : backends_) {
      bool selected = (bi.backend == current_backend_);
      ImGui::BeginDisabled(!bi.available);
      if (ImGui::RadioButton(bi.name, selected) && !selected) {
        set_backend(bi.backend);
      }
      ImGui::EndDisabled();
      if (!bi.available && bi.note && bi.note[0]) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%s)", bi.note);
      }
    }
    ImGui::Separator();
    if (renderer_)
      ImGui::Text("Active: %s", renderer_->name());
    if (current_backend_ == Backend::CPU) {
      ImGui::SliderInt("CPU threads (0=auto)", &cpu_threads_, 0, 64);
    }
  }
  ImGui::End();

  // --------------------------------------------------------------- Stats
  ImGui::Begin("Stats");
  {
    if (!volume_.empty()) {
      ImGui::Text("Grid: %d x %d x %d", volume_.nx, volume_.ny, volume_.nz);
      ImGui::Text("Value range: [%.6g, %.6g]", volume_.vmin, volume_.vmax);
    } else {
      ImGui::TextDisabled("No volume loaded.");
    }
    if (last_downscale_ > render_downscale_)
      ImGui::Text("Render: %d x %d (interactive, 1/%d)", last_render_w_,
                  last_render_h_, last_downscale_);
    else
      ImGui::Text("Render: %d x %d", last_render_w_, last_render_h_);
    // Rendering is on demand, so a "frame rate" derived from one render would
    // mislead; report the durations instead. The UI refresh rate is throttled
    // to a few Hz when idle by design.
    ImGui::Text("Render time: %.2f ms", last_render_ms_);
    ImGui::Text("Load time: %.1f ms", last_load_ms_);
    ImGui::TextDisabled("UI refresh: %.0f Hz (throttled when idle)",
                        ImGui::GetIO().Framerate);
  }
  ImGui::End();

  // -------------------------------------------------------------- Status
  // One-line message bar: the result of the last action (load, save, backend
  // switch) or the last error, plus the active renderer on the right.
  {
    ImGui::Begin("Status", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);
    const char *backend = renderer_ ? renderer_->name() : "-";
    float right_w = ImGui::CalcTextSize(backend).x;
    float avail = ImGui::GetContentRegionAvail().x;
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + avail - right_w -
                           ImGui::GetStyle().ItemSpacing.x * 2);
    ImGui::TextUnformatted(status_.c_str());
    ImGui::PopTextWrapPos();
    if (ImGui::IsItemHovered() && !status_.empty())
      ImGui::SetTooltip("%s", status_.c_str());
    ImGui::SameLine(avail - right_w);
    ImGui::TextDisabled("%s", backend);
    ImGui::End();
  }

  // ------------------------------------------------------------ Viewport
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::Begin("Viewport");
  {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    int vw = static_cast<int>(avail.x);
    int vh = static_cast<int>(avail.y);

    if (vw > 0 && vh > 0) {
      // Re-render on demand or when the viewport was resized. While a mouse
      // button is held (orbit/pan, slider drags) the effective downscale grows
      // adaptively; on release it drops back to the user's setting, which
      // shows up here as a size change and triggers the full-quality refine.
      const int eff = effective_downscale(ImGui::IsAnyMouseDown());
      int want_w = std::max(1, vw / eff);
      int want_h = std::max(1, vh / eff);
      if (needs_render_ || want_w != last_render_w_ || want_h != last_render_h_) {
        render(vw, vh, eff);
        needs_render_ = false;
      }

      // C-style cast so this works whether ImTextureID is a pointer (older
      // ImGui) or an integer handle (ImU64 in recent versions).
      const ImVec2 img_pos = ImGui::GetCursorScreenPos();
      const ImVec2 img_size(static_cast<float>(vw), static_cast<float>(vh));
      ImGui::Image((ImTextureID)(std::uintptr_t)texture_.id(), img_size);
      // An invisible button over the image owns the mouse input: unlike a
      // hover test on the (non-interactive) image it stays *active* while a
      // button pressed on it is held, so an orbit or pan continues even when
      // the cursor leaves the viewport mid-drag.
      ImGui::SetCursorScreenPos(img_pos);
      ImGui::InvisibleButton("##viewport_input", img_size,
                             ImGuiButtonFlags_MouseButtonLeft |
                                 ImGuiButtonFlags_MouseButtonRight |
                                 ImGuiButtonFlags_MouseButtonMiddle);
      const bool input_active = ImGui::IsItemActive();
      const bool input_hovered = ImGui::IsItemHovered();

      // Box outline: project the edges of the (possibly tiled) volume box
      // with the render camera and draw them over the image. Backend-agnostic
      // by construction — no ray-march kernel is involved.
      if (show_box_ && !volume_.empty()) {
        // Edge vectors of the (possibly sheared) tiled cell are the columns of
        // C scaled by the replica counts; corners are C * frac, frac in {0,rep}.
        Mat3 C = world_cell();
        float rx = static_cast<float>(rep(0));
        float ry = static_cast<float>(rep(1));
        float rz = static_cast<float>(rep(2));
        // Use the aspect the frame was actually rendered with (the downscaled
        // framebuffer is stretched onto the viewport), so the outline lands
        // exactly on the rendered box.
        float aspect = last_render_h_ > 0
                           ? static_cast<float>(last_render_w_) / last_render_h_
                           : static_cast<float>(vw) / vh;
        Camera cam = camera_.to_camera(aspect);
        // Contrast the line with the background.
        float lum = 0.2126f * bg_[0] + 0.7152f * bg_[1] + 0.0722f * bg_[2];
        ImU32 col = lum > 0.5f ? IM_COL32(30, 30, 30, 200)
                               : IM_COL32(225, 225, 225, 200);
        ImDrawList *dl = ImGui::GetWindowDrawList();
        // The 12 edges connect the corner pairs differing in one axis bit.
        for (int i = 0; i < 8; ++i) {
          Vec3 c0 = mat3_mul(C, Vec3{(i & 1) ? rx : 0.0f, (i & 2) ? ry : 0.0f,
                                     (i & 4) ? rz : 0.0f});
          for (int bit = 1; bit <= 4; bit <<= 1) {
            if (i & bit) continue;
            int j = i | bit;
            Vec3 c1 = mat3_mul(C, Vec3{(j & 1) ? rx : 0.0f, (j & 2) ? ry : 0.0f,
                                       (j & 4) ? rz : 0.0f});
            draw_box_edge(dl, cam, c0, c1, img_pos,
                          static_cast<float>(vw), static_cast<float>(vh), col);
          }
        }
      }

      // Mouse interaction drives the orbit camera: drags while the viewport
      // button is active (pressed on the image, wherever the cursor is now),
      // wheel zoom while hovering it.
      {
        ImGuiIO &io = ImGui::GetIO();
        if (input_active) {
          if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            camera_.orbit(io.MouseDelta.x * 0.01f, io.MouseDelta.y * 0.01f);
            needs_render_ = true;
          } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Right) ||
                     ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            camera_.pan(io.MouseDelta.x / vw, io.MouseDelta.y / vh,
                        static_cast<float>(vw) / vh);
            needs_render_ = true;
          }
        }
        if (input_hovered && io.MouseWheel != 0.0f) {
          camera_.zoom(io.MouseWheel);
          needs_render_ = true;
        }
      }
    }
  }
  ImGui::End();
  ImGui::PopStyleVar();
}

}  // namespace mueye
