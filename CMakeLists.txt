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

# Rendering backends. CPU is always built. Metal is built by default on Apple.
# CUDA/HIP are opt-in and require the respective toolchain (not available on the
# CPU-only macOS dev machine).
if(APPLE)
    option(MUEYE_ENABLE_METAL "Build the Metal GPU backend" ON)
else()
    option(MUEYE_ENABLE_METAL "Build the Metal GPU backend" OFF)
endif()
option(MUEYE_ENABLE_CUDA "Build the CUDA GPU backend" OFF)
option(MUEYE_ENABLE_HIP  "Build the HIP GPU backend"  OFF)
if(MUEYE_ENABLE_CUDA AND MUEYE_ENABLE_HIP)
    message(FATAL_ERROR "Enable only one of MUEYE_ENABLE_CUDA / MUEYE_ENABLE_HIP.")
endif()
if(MUEYE_ENABLE_METAL AND NOT APPLE)
    message(FATAL_ERROR "MUEYE_ENABLE_METAL requires an Apple platform.")
endif()

# Additional languages must be enabled early — right after project() and before
# add_subdirectory(muGrid) — so CMake loads their compile rules correctly.
if(MUEYE_ENABLE_METAL)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()
if(MUEYE_ENABLE_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
    endif()
elseif(MUEYE_ENABLE_HIP)
    enable_language(HIP)
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
# CPU parallelism: OpenMP if available, else the std::thread fallback (Threads).
# ---------------------------------------------------------------------------
find_package(OpenMP QUIET)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Assemble the rendering-backend sources, libraries and compile definitions.
# CpuRenderer + RendererFactory are always present; Metal / CUDA / HIP are
# added according to the options above.
# ---------------------------------------------------------------------------
set(MUEYE_RENDER_SOURCES
    src/render/CpuRenderer.cc
    src/render/RendererFactory.cc)
set(MUEYE_BACKEND_LIBS "")
set(MUEYE_BACKEND_DEFS "")

if(MUEYE_ENABLE_METAL)
    list(APPEND MUEYE_RENDER_SOURCES src/render/MetalRenderer.mm)
    # ARC for the Objective-C++ implementation.
    set_source_files_properties(src/render/MetalRenderer.mm PROPERTIES
        COMPILE_OPTIONS "-fobjc-arc")
    find_library(METAL_LIB Metal REQUIRED)
    find_library(FOUNDATION_LIB Foundation REQUIRED)
    list(APPEND MUEYE_BACKEND_LIBS ${METAL_LIB} ${FOUNDATION_LIB})
    list(APPEND MUEYE_BACKEND_DEFS MUEYE_ENABLE_METAL)
endif()

if(MUEYE_ENABLE_CUDA)
    list(APPEND MUEYE_RENDER_SOURCES src/render/GpuRenderer.cc)
    set_source_files_properties(src/render/GpuRenderer.cc PROPERTIES
        LANGUAGE CUDA)
    # MUGRID_ENABLE_CUDA switches muGrid's gpu_runtime.hh into its CUDA path.
    list(APPEND MUEYE_BACKEND_DEFS MUEYE_ENABLE_CUDA MUGRID_ENABLE_CUDA)
elseif(MUEYE_ENABLE_HIP)
    list(APPEND MUEYE_RENDER_SOURCES src/render/GpuRenderer.cc)
    set_source_files_properties(src/render/GpuRenderer.cc PROPERTIES
        LANGUAGE HIP)
    list(APPEND MUEYE_BACKEND_DEFS MUEYE_ENABLE_HIP MUGRID_ENABLE_HIP)
endif()

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
    src/gl/GlTexture.cc
    ${MUEYE_RENDER_SOURCES})

target_include_directories(muEye PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NetCDF_INCLUDE_DIRS})

target_compile_definitions(muEye PRIVATE ${MUEYE_BACKEND_DEFS})

target_link_libraries(muEye PRIVATE
    muGrid           # note: under add_subdirectory the target is `muGrid`, no namespace alias
    imgui
    glfw
    OpenGL::GL
    Threads::Threads
    ${NetCDF_LIBRARIES}
    ${MUEYE_BACKEND_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(muEye PRIVATE OpenMP::OpenMP_CXX)
endif()

# Helpful warnings (kept non-fatal for a draft).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(muEye PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Headless verification tool (no ImGui/OpenGL): writes a demo .nc via muGrid,
# reads it back through the loader, scalarizes, ray-traces offscreen and
# cross-checks every available backend against the CPU reference.
# Build with: cmake --build build --target muEye_check
# ---------------------------------------------------------------------------
add_executable(muEye_check
    tools/offscreen_check.cc
    src/io/Volume.cc
    src/io/VolumeLoader.cc
    src/ui/OrbitCamera.cc
    src/ui/TransferFunction.cc
    ${MUEYE_RENDER_SOURCES})
set_target_properties(muEye_check PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_include_directories(muEye_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NetCDF_INCLUDE_DIRS})
target_compile_definitions(muEye_check PRIVATE ${MUEYE_BACKEND_DEFS})
target_link_libraries(muEye_check PRIVATE
    muGrid
    Threads::Threads
    ${NetCDF_LIBRARIES}
    ${MUEYE_BACKEND_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(muEye_check PRIVATE OpenMP::OpenMP_CXX)
endif()

set(_mueye_backends "CPU")
if(MUEYE_ENABLE_METAL)
    string(APPEND _mueye_backends ", Metal")
endif()
if(MUEYE_ENABLE_CUDA)
    string(APPEND _mueye_backends ", CUDA")
endif()
if(MUEYE_ENABLE_HIP)
    string(APPEND _mueye_backends ", HIP")
endif()
message(STATUS "muEye backends: ${_mueye_backends}")
message(STATUS "muEye configured: muGrid=${MUEYE_MUGRID_SOURCE_DIR}, "
               "OpenMP=${OpenMP_CXX_FOUND}, NetCDF=${NetCDF_LIBRARIES}")
