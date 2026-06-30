/**
 * @file   panels.cc
 *
 * @brief  ImGui window layout for muEye (implements App::draw_ui).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "App.hh"
#include "imgui.h"
#include "imgui_internal.h"  // DockBuilder* for the default layout

namespace mueye {

// Arrange the panels into a default layout: a left control column (grouped into
// three stacked tab-nodes) and a large viewport filling the rest. Called once
// when there is no docking layout yet, so a saved imgui.ini still wins.
static void build_default_layout(ImGuiID dockspace_id) {
  ImGui::DockBuilderRemoveNode(dockspace_id);
  ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

  ImGuiID center = dockspace_id;
  ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.26f,
                                             nullptr, &center);
  ImGuiID left_rest = left;
  ImGuiID left_top = ImGui::DockBuilderSplitNode(left_rest, ImGuiDir_Up, 0.34f,
                                                 nullptr, &left_rest);
  ImGuiID left_mid = ImGui::DockBuilderSplitNode(left_rest, ImGuiDir_Up, 0.5f,
                                                 nullptr, &left_rest);
  ImGuiID left_bot = left_rest;

  ImGui::DockBuilderDockWindow("muEye", left_top);
  ImGui::DockBuilderDockWindow("Dataset", left_top);
  ImGui::DockBuilderDockWindow("Render", left_mid);
  ImGui::DockBuilderDockWindow("Transfer function", left_mid);
  ImGui::DockBuilderDockWindow("Device", left_bot);
  ImGui::DockBuilderDockWindow("Stats", left_bot);
  ImGui::DockBuilderDockWindow("Viewport", center);
  ImGui::DockBuilderFinish(dockspace_id);
}

void App::draw_ui() {
  // A full-window dock space so panels and the viewport can be arranged freely.
  // Called with no arguments so it compiles across docking-branch versions
  // (the dockspace-id overload was added later but keeps the same defaults).
  ImGuiID dockspace_id = ImGui::DockSpaceOverViewport();

  // First-run default layout: only build it if this dockspace has no nodes yet
  // (i.e. no imgui.ini restored a previous arrangement).
  static bool layout_initialized = false;
  if (!layout_initialized) {
    layout_initialized = true;
    ImGuiDockNode *node = ImGui::DockBuilderGetNode(dockspace_id);
    if (node == nullptr || node->IsLeafNode()) {
      build_default_layout(dockspace_id);
    }
  }

  // ---------------------------------------------------------------- File
  ImGui::Begin("muEye");
  {
    ImGui::TextWrapped("muGrid NetCDF viewer — real-time volume ray tracer.");
    ImGui::Separator();

    static char path[1024] = {0};
    if (!path_buf_.empty() && path[0] == '\0') {
      std::snprintf(path, sizeof(path), "%s", path_buf_.c_str());
    }
    ImGui::InputText("File", path, sizeof(path));
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
      open_path(path);
    }
    ImGui::TextWrapped("%s", status_.c_str());
  }
  ImGui::End();

  // ------------------------------------------------------------- Dataset
  if (has_file_ && !meta_.fields.empty()) {
    ImGui::Begin("Dataset");
    bool reload = false;

    std::vector<const char *> names;
    names.reserve(meta_.fields.size());
    for (auto &f : meta_.fields) names.push_back(f.name.c_str());
    reload |= ImGui::Combo("Field", &field_index_, names.data(),
                           static_cast<int>(names.size()));

    if (meta_.nb_frames > 1) {
      int f = frame_;
      if (ImGui::SliderInt("Frame", &f, 0, meta_.nb_frames - 1)) {
        frame_ = f;
        reload = true;
      }
    }

    const FieldInfo &fi = meta_.fields[field_index_];
    const char *modes[] = {"Component", "Magnitude", "von Mises (3x3)",
                           "Trace (3x3)"};
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
    }
    ImGui::Text("components: %d   sub-points: %d", fi.nb_components,
                fi.nb_sub_pts);

    if (reload) reload_volume();
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
      if (ImGui::SliderFloat("Density", &density_scale_, 0.05f, 5.0f, "%.2f"))
        needs_render_ = true;
    } else {
      float lo = volume_.empty() ? 0.f : volume_.vmin;
      float hi = volume_.empty() ? 1.f : volume_.vmax;
      if (ImGui::SliderFloat("Iso value", &iso_value_, lo, hi, "%.4g"))
        needs_render_ = true;
    }

    if (ImGui::ColorEdit3("Background", bg_)) needs_render_ = true;

    if (ImGui::SliderInt("Downscale", &render_downscale_, 1, 4))
      needs_render_ = true;
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
    float op = tf_.opacity_scale();
    if (ImGui::SliderFloat("Opacity", &op, 0.0f, 2.0f, "%.2f")) {
      tf_.set_opacity_scale(op);
      tf_dirty_ = true;
      needs_render_ = true;
    }
    float g = tf_.opacity_gamma();
    if (ImGui::SliderFloat("Opacity gamma", &g, 0.2f, 4.0f, "%.2f")) {
      tf_.set_opacity_gamma(g);
      tf_dirty_ = true;
      needs_render_ = true;
    }

    // A small preview strip of the colormap.
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    float h = 24.0f;
    int steps = 64;
    for (int i = 0; i < steps; ++i) {
      float t0 = static_cast<float>(i) / steps;
      const Vec4 *lut = tf_.data();
      int li = static_cast<int>(t0 * (tf_.size() - 1));
      Vec4 c = lut[li];
      ImU32 col = IM_COL32(static_cast<int>(c.x * 255), static_cast<int>(c.y * 255),
                           static_cast<int>(c.z * 255), 255);
      dl->AddRectFilled(ImVec2(p0.x + w * t0, p0.y),
                        ImVec2(p0.x + w * (i + 1) / steps, p0.y + h), col);
    }
    ImGui::Dummy(ImVec2(w, h));
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
    ImGui::Text("Render: %d x %d", last_render_w_, last_render_h_);
    ImGui::Text("Frame time: %.2f ms (%.1f fps)", last_render_ms_,
                last_render_ms_ > 0 ? 1000.0 / last_render_ms_ : 0.0);
    ImGui::Text("UI: %.1f fps", ImGui::GetIO().Framerate);
  }
  ImGui::End();

  // ------------------------------------------------------------ Viewport
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::Begin("Viewport");
  {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    int vw = static_cast<int>(avail.x);
    int vh = static_cast<int>(avail.y);

    if (vw > 0 && vh > 0) {
      // Re-render on demand or when the viewport was resized.
      int want_w = vw / render_downscale_;
      int want_h = vh / render_downscale_;
      if (needs_render_ || want_w != last_render_w_ || want_h != last_render_h_) {
        render(vw, vh);
        needs_render_ = false;
      }

      // C-style cast so this works whether ImTextureID is a pointer (older
      // ImGui) or an integer handle (ImU64 in recent versions).
      ImGui::Image((ImTextureID)(std::uintptr_t)texture_.id(),
                   ImVec2(static_cast<float>(vw), static_cast<float>(vh)));

      // Mouse interaction over the image drives the orbit camera.
      if (ImGui::IsItemHovered()) {
        ImGuiIO &io = ImGui::GetIO();
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
          camera_.orbit(io.MouseDelta.x * 0.01f, io.MouseDelta.y * 0.01f);
          needs_render_ = true;
        } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Right) ||
                   ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
          camera_.pan(io.MouseDelta.x / vw, io.MouseDelta.y / vh);
          needs_render_ = true;
        }
        if (io.MouseWheel != 0.0f) {
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
