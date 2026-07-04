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
  if (!layout_initialized_) {
    layout_initialized_ = true;
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

    ImGui::InputText("File", path_edit_, sizeof(path_edit_));
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
      open_path(path_edit_);
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
      if (ImGui::SliderInt("Frame", &f, 0, meta_.nb_frames - 1) &&
          f != frame_) {
        frame_ = f;
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

    ImGui::Separator();
    // Drawn as a viewport overlay, so toggling needs no re-render.
    ImGui::Checkbox("Show box", &show_box_);

    bool tiling_changed = ImGui::Checkbox("Periodic images", &periodic_);
    if (periodic_) {
      tiling_changed |= ImGui::InputInt3("Replicas", replicas_);
    }
    if (tiling_changed) {
      for (int &r : replicas_) r = r < 1 ? 1 : (r > 8 ? 8 : r);
      // Keep the view direction but recentre on the tiled box.
      if (!volume_.empty()) {
        Vec3 b = box_extent(volume_.nx, volume_.ny, volume_.nz);
        camera_.retarget_box(Vec3{b.x * rep(0), b.y * rep(1), b.z * rep(2)});
      }
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

      // Box outline: project the edges of the (possibly tiled) volume box
      // with the render camera and draw them over the image. Backend-agnostic
      // by construction — no ray-march kernel is involved.
      if (show_box_ && !volume_.empty()) {
        ImVec2 img_pos = ImGui::GetItemRectMin();
        Vec3 b = box_extent(volume_.nx, volume_.ny, volume_.nz);
        Vec3 tb{b.x * rep(0), b.y * rep(1), b.z * rep(2)};
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
          Vec3 c0{(i & 1) ? tb.x : 0.0f, (i & 2) ? tb.y : 0.0f,
                  (i & 4) ? tb.z : 0.0f};
          for (int bit = 1; bit <= 4; bit <<= 1) {
            if (i & bit) continue;
            int j = i | bit;
            Vec3 c1{(j & 1) ? tb.x : 0.0f, (j & 2) ? tb.y : 0.0f,
                    (j & 4) ? tb.z : 0.0f};
            draw_box_edge(dl, cam, c0, c1, img_pos,
                          static_cast<float>(vw), static_cast<float>(vh), col);
          }
        }
      }

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
