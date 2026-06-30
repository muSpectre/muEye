/**
 * @file   main.cc
 *
 * @brief  Entry point: GLFW window, OpenGL 3 context, Dear ImGui bootstrap and
 *         the main loop.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include <cstdio>

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
