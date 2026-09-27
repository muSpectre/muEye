/**
 * @file   main.cc
 *
 * @brief  Entry point: GLFW window, OpenGL 3 context, Dear ImGui bootstrap and
 *         the main loop.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

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
// The size given here is a base size in UI points: ImGui >= 1.92 rasterizes
// fonts dynamically at the effective scale (framebuffer scale for Retina
// crispness, style.FontScaleDpi for monitor DPI — see io.ConfigDpiScaleFonts
// in main()), so no manual content-scale baking is needed.
static void load_platform_font() {
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

  const float base_px = 16.0f;

  for (const char *path : candidates) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) continue;
    if (io.Fonts->AddFontFromFileTTF(path, base_px) != nullptr) {
      std::printf("muEye: using UI font %s\n", path);
      return;
    }
  }
  // Nothing found — keep Dear ImGui's built-in font.
  io.Fonts->AddFontDefault();
  std::printf("muEye: no platform font found; using built-in font\n");
}

// Where Dear ImGui persists the window/docking layout. ImGui's default is
// "imgui.ini" in the *current working directory*, which made the layout depend
// on where muEye was launched from and littered data directories (and the
// source tree) with ini files. Use the platform's per-user configuration
// directory instead; fall back to the CWD default only if no home is known.
//
//   Linux/Unix: $XDG_CONFIG_HOME/muEye/imgui.ini  (default ~/.config)
//   macOS:      ~/Library/Application Support/muEye/imgui.ini
//   Windows:    %APPDATA%\muEye\imgui.ini
//
// Returns an empty string when no suitable directory could be found or made.
static std::string user_ini_path() {
  namespace fs = std::filesystem;
  fs::path dir;
#if defined(_WIN32)
  if (const char *appdata = std::getenv("APPDATA"); appdata && *appdata)
    dir = fs::path(appdata);
#elif defined(__APPLE__)
  if (const char *home = std::getenv("HOME"); home && *home)
    dir = fs::path(home) / "Library" / "Application Support";
#else
  if (const char *xdg = std::getenv("XDG_CONFIG_HOME"); xdg && *xdg) {
    dir = fs::path(xdg);
  } else if (const char *home = std::getenv("HOME"); home && *home) {
    dir = fs::path(home) / ".config";
  }
#endif
  if (dir.empty()) return "";
  dir /= "muEye";
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec && !fs::is_directory(dir, ec)) return "";
  return (dir / "imgui.ini").string();
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
  // Rescale fonts automatically when the window moves to a monitor with a
  // different DPI (the GLFW backend reports per-monitor content scale, and
  // ImGui >= 1.92 rasterizes fonts dynamically at the effective scale).
  io.ConfigDpiScaleFonts = true;
  // Persist the layout per user rather than in the CWD. The string must
  // outlive the ImGui context, which only stores the pointer.
  static const std::string ini_path = user_ini_path();
  if (!ini_path.empty()) io.IniFilename = ini_path.c_str();
  ImGui::StyleColorsLight();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Use the host platform's native UI font.
  load_platform_font();

  // Scope the app so its GL-owning members (display texture, GPU renderers
  // with GL-interop buffers) are destroyed while the GL context is still
  // current — i.e. before the ImGui/GLFW teardown below destroys it.
  {
    mueye::App app;
    // Dropping a file onto the window opens it (the first path of a
    // multi-file drop). The callback runs from glfwPollEvents /
    // glfwWaitEventsTimeout, i.e. between ImGui frames, so loading there is
    // safe.
    glfwSetWindowUserPointer(window, &app);
    glfwSetDropCallback(window, [](GLFWwindow *w, int count,
                                   const char **paths) {
      if (count <= 0 || paths == nullptr || paths[0] == nullptr) return;
      auto *a = static_cast<mueye::App *>(glfwGetWindowUserPointer(w));
      if (a != nullptr) a->load_file(paths[0]);
    });
    if (argc > 1) {
      app.load_file(argv[1]);
    }

    // Idle throttle: while the user interacts, render at full rate; once the
    // input goes quiet for a few frames, park in glfwWaitEventsTimeout so an
    // idle viewer costs almost no CPU. Any event (mouse, key, resize) wakes
    // the loop immediately and renders a frame, so responsiveness is
    // unaffected; the timeout keeps slow animations (cursor blink, fades)
    // ticking over. Start hot so the initial docking layout settles.
    int hot_frames = 10;
    while (!glfwWindowShouldClose(window)) {
      if (hot_frames > 0) {
        --hot_frames;
        glfwPollEvents();
      } else {
        glfwWaitEventsTimeout(0.25);
      }

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      // Mouse activity re-arms continuous rendering (drags stream events, but
      // ImGui needs a few extra frames after the last one for hover state and
      // animations to settle).
      ImGuiIO &loop_io = ImGui::GetIO();
      if (loop_io.MouseDelta.x != 0.0f || loop_io.MouseDelta.y != 0.0f ||
          loop_io.MouseWheel != 0.0f || ImGui::IsAnyMouseDown()) {
        hot_frames = 3;
      }

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
    // The drop callback holds the App by pointer; detach it before the App
    // is destroyed at the end of this scope.
    glfwSetDropCallback(window, nullptr);
    glfwSetWindowUserPointer(window, nullptr);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
