/**
 * @file   main.cc
 *
 * @brief  Entry point: GLFW window, OpenGL 3 context, Dear ImGui bootstrap and
 *         the main loop.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include <cstdio>
#include <filesystem>

#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#endif

#include "App.hh"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"

// GLFW must come after the ImGui OpenGL3 backend on some platforms.
#include <GLFW/glfw3.h>

static void glfw_error_callback(int error, const char *description) {
  std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

// Load the platform's native UI font so muEye matches the look of the host
// desktop, instead of Dear ImGui's built-in bitmap font (ProggyClean).
//
//   - macOS:   San Francisco (SFNS.ttf) — the system font; Helvetica fallback.
//   - Windows: Segoe UI (segoeui.ttf)   — the system font.
//   - Linux:   no single default; try Ubuntu / Cantarell / Noto, falling back
//              to the near-ubiquitous DejaVu Sans / Liberation Sans.
//
// The font is rasterised at base_px * content_scale and then scaled back down
// with FontGlobalScale so text stays crisp on HiDPI / Retina displays.
static void load_platform_font(GLFWwindow *window) {
  ImGuiIO &io = ImGui::GetIO();

#if defined(__APPLE__)
  const char *candidates[] = {
      "/System/Library/Fonts/SFNS.ttf",       // San Francisco (system font)
      "/System/Library/Fonts/SFNSText.ttf",   // older naming
      "/System/Library/Fonts/Helvetica.ttc",  // robust fallback
  };
#elif defined(_WIN32)
  const char *candidates[] = {
      "C:\\Windows\\Fonts\\segoeui.ttf",  // Segoe UI (system font)
      "C:\\Windows\\Fonts\\tahoma.ttf",
      "C:\\Windows\\Fonts\\arial.ttf",
  };
#else  // Linux / other Unix
  const char *candidates[] = {
      "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
      "/usr/share/fonts/cantarell/Cantarell-Regular.otf",
      "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/TTF/DejaVuSans.ttf",  // Arch
      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
  };
#endif

  float xscale = 1.0f, yscale = 1.0f;
  glfwGetWindowContentScale(window, &xscale, &yscale);
  if (xscale <= 0.0f) xscale = 1.0f;
  const float base_px = 16.0f;

  for (const char *path : candidates) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) continue;
    if (io.Fonts->AddFontFromFileTTF(path, base_px * xscale) != nullptr) {
      io.FontGlobalScale = 1.0f / xscale;
      std::printf("muEye: using UI font %s\n", path);
      return;
    }
  }
  // Nothing found — keep Dear ImGui's built-in font.
  io.Fonts->AddFontDefault();
  std::printf("muEye: no platform font found; using built-in font\n");
}

int main(int argc, char **argv) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    std::fprintf(stderr, "Failed to initialize GLFW\n");
    return 1;
  }

  // Request an OpenGL 3.2 core context (works on macOS and Linux).
#if defined(__APPLE__)
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

  GLFWwindow *window =
      glfwCreateWindow(1440, 900, "muEye — muGrid viewer", nullptr, nullptr);
  if (window == nullptr) {
    std::fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // vsync

  // Dear ImGui setup.
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ImGui::StyleColorsLight();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Use the host platform's native UI font.
  load_platform_font(window);

  mueye::App app;
  if (argc > 1) {
    app.load_file(argv[1]);
  }

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    app.draw_ui();

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
