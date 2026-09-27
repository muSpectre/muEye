#
# muEye — a real-time ray-tracing viewer for muGrid data.
#
# This is a standalone CMake project. It consumes muGrid via add_subdirectory
# (see README.md for why find_package does not work here) and fetches GLFW and
# Dear ImGui via FetchContent.
#
cmake_minimum_required(VERSION 3.18)
project(muEye VERSION 0.1.0 LANGUAGES CXX)

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

# netcdf-c exports the MSVC flag "/utf-8" as a PUBLIC compile option with no
# language guard, so it also reaches nvcc when compiling GpuRenderer.cc, and
# nvcc takes it for a second input file. Hand the target's options to the host
# compiler (-Xcompiler) for CUDA sources and leave them unchanged elsewhere.
if(MUEYE_ENABLE_CUDA AND MSVC AND TARGET netCDF::netcdf)
    get_target_property(_nc_opts netCDF::netcdf INTERFACE_COMPILE_OPTIONS)
    if(_nc_opts)
        set(_nc_wrapped "")
        foreach(_opt IN LISTS _nc_opts)
            list(APPEND _nc_wrapped
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_opt}>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_opt}>")
        endforeach()
        set_target_properties(netCDF::netcdf PROPERTIES
            INTERFACE_COMPILE_OPTIONS "${_nc_wrapped}")
    endif()
endif()

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
# On Linux, GLFW 3.4 builds both X11 and Wayland backends by default; the Wayland
# one requires wayland-scanner + protocol dev packages at configure time. We only
# need X11 (Wayland compositors run it via XWayland), so disable Wayland to keep
# the build self-contained with just the X11 dev headers (xorg-dev). No effect on
# macOS/Windows, which use their native backend.
set(GLFW_BUILD_WAYLAND  OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11      ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Dear ImGui has no CMake build of its own; we fetch sources and compile a small
# static library here. Pinned to a release tag of the *docking* line (plain
# vX.Y.Z tags are master and lack DockSpaceOverViewport / DockBuilder, which
# the panel layout needs) rather than the moving `docking` branch, whose head
# makes builds non-reproducible. Bump deliberately.
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.92.8-docking
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
    # MUGRID_ENABLE_CUDA switches muGrid's gpu_runtime.hh into its CUDA path,
    # which pulls in <cuda_runtime.h>. Scope it to the nvcc-compiled source
    # ONLY — not the whole target — or host TUs that transitively include muGrid
    # headers (e.g. offscreen_check.cc via file_io_netcdf.hh) would try to
    # include it under a plain C++ compiler and fail.
    set_source_files_properties(src/render/GpuRenderer.cc PROPERTIES
        LANGUAGE CUDA
        COMPILE_DEFINITIONS MUGRID_ENABLE_CUDA)
    list(APPEND MUEYE_BACKEND_DEFS MUEYE_ENABLE_CUDA)
elseif(MUEYE_ENABLE_HIP)
    list(APPEND MUEYE_RENDER_SOURCES src/render/GpuRenderer.cc)
    set_source_files_properties(src/render/GpuRenderer.cc PROPERTIES
        LANGUAGE HIP
        COMPILE_DEFINITIONS MUGRID_ENABLE_HIP)
    list(APPEND MUEYE_BACKEND_DEFS MUEYE_ENABLE_HIP)
endif()

# ---------------------------------------------------------------------------
# mueye_core: everything shared between the muEye executable and the
# muEye_check tool (I/O, camera, transfer function, rendering backends) —
# compiled once, and the two consumers cannot drift apart. Usage requirements
# (includes, backend defines, libraries) are PUBLIC so consumers inherit them.
# ---------------------------------------------------------------------------
add_library(mueye_core STATIC
    src/io/Volume.cc
    src/io/VolumeLoader.cc
    src/io/PngWriter.cc
    src/ui/OrbitCamera.cc
    src/ui/TransferFunction.cc
    ${MUEYE_RENDER_SOURCES})

target_include_directories(mueye_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NetCDF_INCLUDE_DIRS})
# Vendored single-header libs (stb_image_write); SYSTEM keeps -Wall quiet there.
target_include_directories(mueye_core SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

target_compile_definitions(mueye_core PUBLIC ${MUEYE_BACKEND_DEFS})

target_link_libraries(mueye_core PUBLIC
    muGrid           # note: under add_subdirectory the target is `muGrid`, no namespace alias
    Threads::Threads
    ${NetCDF_LIBRARIES}
    ${MUEYE_BACKEND_LIBS})

if(MUEYE_ENABLE_CUDA OR MUEYE_ENABLE_HIP)
    # GpuRenderer.cc pulls in OpenGL for its PBO interop path, so even headless
    # consumers (muEye_check) must link GL when a CUDA/HIP backend is built in.
    target_link_libraries(mueye_core PUBLIC OpenGL::GL)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(mueye_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# muEye executable (the GUI: GLFW window, ImGui panels, GL display texture)
# ---------------------------------------------------------------------------
add_executable(muEye
    src/main.cc
    src/App.cc
    src/ui/panels.cc
    src/gl/GlTexture.cc)

target_link_libraries(muEye PRIVATE
    mueye_core
    imgui
    glfw
    OpenGL::GL)

# On macOS, build a .app bundle so the µSpectre logo shows as the Dock/Finder
# icon. The .icns (generated from resources/muspectre_icon.png with iconutil) is
# copied into the bundle's Resources and referenced by the Info.plist that CMake
# generates from MACOSX_BUNDLE_*. MACOSX_BUNDLE is ignored on other platforms.
if(APPLE)
    set(_mueye_icns ${CMAKE_CURRENT_SOURCE_DIR}/resources/muEye.icns)
    target_sources(muEye PRIVATE ${_mueye_icns})
    set_source_files_properties(${_mueye_icns} PROPERTIES
        MACOSX_PACKAGE_LOCATION "Resources")
    set_target_properties(muEye PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "muEye"
        MACOSX_BUNDLE_GUI_IDENTIFIER "org.muspectre.muEye"
        MACOSX_BUNDLE_ICON_FILE "muEye.icns"
        MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
endif()

# Install rules, used by CI to stage the self-contained release packages (see
# .github/workflows/build.yml). Everything is in the `viewer` component so that
# `cmake --install --component viewer` skips muGrid's own install rules.
include(GNUInstallDirs)
if(WIN32)
    set(_mueye_bindir .)
else()
    set(_mueye_bindir ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS muEye
    BUNDLE  DESTINATION .                COMPONENT viewer
    RUNTIME DESTINATION ${_mueye_bindir} COMPONENT viewer)
if(WIN32)
    # Ship the MSVC runtime next to muEye.exe so no VC++ redistributable is needed.
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${_mueye_bindir})
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_COMPONENT viewer)
    include(InstallRequiredSystemLibraries)
endif()

# Helpful warnings (kept non-fatal for a draft); scoped to C++ TUs so they are
# not handed to a CUDA/HIP device compiler.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(muEye PRIVATE -Wall -Wextra)
    target_compile_options(mueye_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

# ---------------------------------------------------------------------------
# Headless verification tool (no ImGui/OpenGL): writes a demo .nc via muGrid,
# reads it back through the loader, scalarizes, ray-traces offscreen and
# cross-checks every available backend against the CPU reference.
# Build with: cmake --build build --target muEye_check
# ---------------------------------------------------------------------------
add_executable(muEye_check tools/offscreen_check.cc)
set_target_properties(muEye_check PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_link_libraries(muEye_check PRIVATE mueye_core)

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
