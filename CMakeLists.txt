#
# muEye — a real-time ray-tracing viewer for muGrid data.
#
# This is a standalone CMake project. It consumes muGrid via add_subdirectory
# (see README.md for why find_package does not work here) and fetches GLFW and
# Dear ImGui via FetchContent.
#
cmake_minimum_required(VERSION 3.18)
project(muEye CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(MUEYE_MUGRID_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../muGrid"
    CACHE PATH "Path to the muGrid source tree (consumed via add_subdirectory)")

# GPU backend is declared but deferred to a follow-up pass. When implemented,
# enabling these will add src/render/render_gpu.cc (compiled with LANGUAGE
# CUDA/HIP, target archs, --extended-lambda --expt-relaxed-constexpr) and reuse
# muGrid's Device + memory/gpu_runtime.hh. The default build is CPU-only and
# must configure cleanly on macOS (which has no CUDA/HIP).
option(MUEYE_ENABLE_CUDA "Build the CUDA GPU ray tracer (not yet implemented)" OFF)
option(MUEYE_ENABLE_HIP  "Build the HIP GPU ray tracer (not yet implemented)"  OFF)
if(MUEYE_ENABLE_CUDA OR MUEYE_ENABLE_HIP)
    message(WARNING "muEye GPU backend is not implemented yet; building CPU-only.")
endif()

# ---------------------------------------------------------------------------
# muGrid (serial, with NetCDF; no Python/MPI/tests/examples needed here)
# ---------------------------------------------------------------------------
set(MUGRID_ENABLE_PYTHON   OFF CACHE BOOL "" FORCE)
set(MUGRID_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
set(MUGRID_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MUGRID_ENABLE_MPI      OFF CACHE BOOL "" FORCE)
set(MUGRID_ENABLE_NETCDF   ON  CACHE BOOL "" FORCE)

if(NOT EXISTS "${MUEYE_MUGRID_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "muGrid source not found at '${MUEYE_MUGRID_SOURCE_DIR}'.\n"
        "Set -DMUEYE_MUGRID_SOURCE_DIR=/path/to/muGrid.")
endif()
add_subdirectory(${MUEYE_MUGRID_SOURCE_DIR} mugrid_build)

# ---------------------------------------------------------------------------
# NetCDF (serial) — muGrid links it PRIVATEly, but muEye includes
# io/file_io_netcdf.hh (which pulls <netcdf.h>) and calls the netcdf-c API
# directly for file introspection, so we must find it ourselves.
# ---------------------------------------------------------------------------
find_package(NetCDF REQUIRED)

# ---------------------------------------------------------------------------
# OpenGL + GLFW + Dear ImGui
# ---------------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Dear ImGui has no CMake build of its own; we fetch sources and compile a small
# static library here. The docking branch provides DockSpaceOverViewport.
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        docking
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# ---------------------------------------------------------------------------
# OpenMP (optional) for the CPU renderer
# ---------------------------------------------------------------------------
find_package(OpenMP QUIET)

# ---------------------------------------------------------------------------
# muEye executable
# ---------------------------------------------------------------------------
add_executable(muEye
    src/main.cc
    src/App.cc
    src/ui/panels.cc
    src/ui/OrbitCamera.cc
    src/ui/TransferFunction.cc
    src/io/Volume.cc
    src/io/VolumeLoader.cc
    src/render/CpuRenderer.cc
    src/gl/GlTexture.cc)

target_include_directories(muEye PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NetCDF_INCLUDE_DIRS})

target_link_libraries(muEye PRIVATE
    muGrid           # note: under add_subdirectory the target is `muGrid`, no namespace alias
    imgui
    glfw
    OpenGL::GL
    ${NetCDF_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(muEye PRIVATE OpenMP::OpenMP_CXX)
endif()

# Helpful warnings (kept non-fatal for a draft).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(muEye PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Headless verification tool (no ImGui/OpenGL): writes a demo .nc via muGrid,
# reads it back through the loader, scalarizes and ray-traces offscreen.
# Build with: cmake --build build --target muEye_check
# ---------------------------------------------------------------------------
# Only the data + render + camera/transfer-function layers — no App/panels (which
# pull in ImGui) and no GlTexture (which pulls in OpenGL).
add_executable(muEye_check
    tools/offscreen_check.cc
    src/io/Volume.cc
    src/io/VolumeLoader.cc
    src/render/CpuRenderer.cc
    src/ui/OrbitCamera.cc
    src/ui/TransferFunction.cc)
set_target_properties(muEye_check PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_include_directories(muEye_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NetCDF_INCLUDE_DIRS})
target_link_libraries(muEye_check PRIVATE muGrid ${NetCDF_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(muEye_check PRIVATE OpenMP::OpenMP_CXX)
endif()

message(STATUS "muEye configured: muGrid=${MUEYE_MUGRID_SOURCE_DIR}, "
               "OpenMP=${OpenMP_CXX_FOUND}, NetCDF=${NetCDF_LIBRARIES}")
