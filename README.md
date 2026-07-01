# muEye

A real-time **ray-tracing viewer for [muGrid](https://github.com/muSpectre/muGrid)
data**. muEye opens muGrid NetCDF files and renders 3D scalar/tensor fields as a
volume with a **CPU ray tracer** (and, in a later pass, a GPU one). The UI is built
with [Dear ImGui](https://github.com/ocornut/imgui).

muEye reuses muGrid internally:

- **I/O** — `muGrid::FileIONetCDF` reads frames/fields; muEye introspects the file's
  grid (`nx`/`ny`/`nz`), frame count and variables via the netcdf-c API first, since
  muGrid's read API needs a pre-shaped `GlobalFieldCollection`.
- **Data representation** — `muGrid::GlobalFieldCollection` / `Field`; field data
  (double precision) is downcast once to `float` on load (the renderer is single
  precision throughout).
- **GPU device handling** (planned) — `muGrid::Device` + `memory/gpu_runtime.hh`.

> **Standalone repo.** muEye is intended to live in its own repository, as a sibling
> of `muGrid`. It does **not** modify muGrid and is not part of muGrid's build; it only
> consumes muGrid as a subproject.

## Features

- Direct volume rendering (DVR) with an editable transfer function (Viridis / Grayscale
  / Cool-Warm colormaps + opacity ramp).
- Isosurface ray-casting with on-the-fly gradient (Phong) shading.
- Orbit camera (left-drag orbit, right/middle-drag pan, wheel zoom).
- Field / frame / component selection, plus derived scalars: vector **magnitude** and,
  for 3×3 tensor fields, **von Mises** and **trace**.
- **Pluggable rendering backends** behind one interface (`render/Renderer.hh`),
  selectable at runtime in the Device panel:
  - **CPU** — multi-threaded (OpenMP, or a `std::thread` fallback so it is parallel
    even on Apple clang).
  - **Metal** — Apple-GPU compute shader (default on macOS).
  - **CUDA / HIP** — single-source kernel sharing `render_core.hh` with the CPU path
    (opt-in; requires the respective toolchain).
- Adjustable render downscale for interactivity.
- Uses the host platform's **native UI font** (San Francisco on macOS, Segoe UI on
  Windows, Ubuntu/Cantarell/Noto→DejaVu Sans on Linux), HiDPI-aware, with a graceful
  fallback to Dear ImGui's built-in font.

## Build

Requirements: a C++20 compiler, CMake ≥ 3.18, a serial **NetCDF** C library, and
OpenGL. GLFW and Dear ImGui are fetched automatically. The muGrid source tree must be
available (default location `../muGrid`).

```bash
# from the muEye/ directory, with muGrid as a sibling (../muGrid)
cmake -S . -B build -DMUEYE_MUGRID_SOURCE_DIR=../muGrid
cmake --build build -j8
```

On macOS install NetCDF e.g. with `brew install netcdf`. On Debian/Ubuntu:
`sudo apt install libnetcdf-dev libglfw3-dev` (GLFW is fetched if absent).

## Run

```bash
# generate a test volume (needs muGrid's Python package; use the workspace venv)
source ../venv/bin/activate
python scripts/make_test_volume.py demo.nc

# launch the viewer (optionally auto-open a file)
./build/muEye demo.nc
```

Then: type a path and **Load** (or pass it on the command line), pick a field/frame,
toggle **DVR**/**Isosurface**, drag to orbit, scroll to zoom.

## Rendering backends

All backends implement `mueye::Renderer` (`src/render/Renderer.hh`), which separates
*data upload* (`set_volume` / `set_transfer_function`, done only when the data changes)
from *per-frame* `render()` — so GPU backends keep the volume resident in device memory.
A backend may also override `render_to_gl()` to draw straight into the display's GL
texture (the CUDA/HIP backend does, via a shared PBO); backends that don't fall back to
`render()` + a host texture upload. `RendererFactory` discovers which backends are
compiled in **and** have a device present; the Device panel lists them and switches at
runtime.

DVR and isosurface are compiled as **separate single-path kernels** and the matching one
is dispatched per frame: the mode is uniform across the launch (so it never causes warp
divergence), but keeping the two paths in separate kernels trims register/instruction
footprint and helps GPU occupancy.

| Backend     | When built                          | Notes |
|-------------|-------------------------------------|-------|
| CPU         | always                              | OpenMP if available, else `std::thread` |
| Metal       | Apple, `MUEYE_ENABLE_METAL` (ON)    | compute shader compiled at runtime (no offline `metal` toolchain needed) |
| CUDA / HIP  | `-DMUEYE_ENABLE_CUDA=ON` / `=ON`    | single-source kernel sharing `render_core.hh`; reuses muGrid `gpu_runtime.hh` |

The CPU, Metal and CUDA/HIP backends all call the **same** ray-march algorithm —
`render_core.hh`'s `__host__ __device__` `trace_ray()` (the Metal shader mirrors it in
MSL) — so their images match. `trace_ray()` is templated on a *volume sampler*: the CPU
does trilinear filtering in software (`ArraySampler`) while the GPU backends sample a
hardware 3-D texture (linear filtering + texture cache), which is much faster for the
3-D-local access pattern of ray marching. `tools/offscreen_check.cc` renders with every
available backend and asserts they agree with the CPU reference; the texture unit's
fixed-point interpolation weights differ from software floats only in the last bit or two
(mean |Δ| ≈ 0.007/255 on Apple GPUs, well under the check's tolerance).

```bash
# CUDA build on an NVIDIA machine:
cmake -S . -B build -DMUEYE_MUGRID_SOURCE_DIR=../muGrid -DMUEYE_ENABLE_CUDA=ON
# HIP build on an AMD/ROCm machine:
cmake -S . -B build -DMUEYE_MUGRID_SOURCE_DIR=../muGrid -DMUEYE_ENABLE_HIP=ON
```

## Layout

```
src/
  main.cc              GLFW + OpenGL3 + ImGui bootstrap and main loop
  App.{hh,cc}          application state; load + render orchestration
  ui/panels.cc         ImGui windows (file/dataset/render/transfer/device/stats/viewport)
  ui/OrbitCamera.*     mouse-driven orbit camera
  ui/TransferFunction.* colormap + opacity LUT
  io/Volume.*          dense float volume + scalarization
  io/VolumeLoader.*    netcdf-c introspection + muGrid-backed field read
  render/render_core.hh shared host/device ray-march core (DVR + isosurface)
  render/Renderer.hh   backend interface (Backend enum) + RGBA8 framebuffer
  render/RendererFactory.* backend discovery + construction
  render/parallel_for.hh   OpenMP / std::thread parallel-for
  render/CpuRenderer.*  multi-threaded CPU backend
  render/MetalRenderer.{hh,mm}  Metal compute backend (Apple)
  render/GpuRenderer.{hh,cc}    CUDA / HIP backend (single-source kernel)
  gl/GlTexture.*       framebuffer -> GL texture for ImGui::Image
scripts/make_test_volume.py   writes a 64^3 demo .nc via muGrid
tools/offscreen_check.cc      headless pipeline + cross-backend verification
```

## Known limitations (first draft)

- Fields on a sub-point (quadrature) subdivision are read, but derived scalars operate
  on sub-point 0 only.
- The transfer function is a colormap preset + opacity ramp (no node editor yet).
- The CUDA/HIP backend displays via a CUDA↔GL PBO (no device→host round trip). The Metal
  backend still copies its result through the host before the GL upload: a true Metal→GL
  zero-copy needs an IOSurface exposed to GL as `GL_TEXTURE_RECTANGLE`, which Dear ImGui's
  GL3 backend can't sample — and on Apple Silicon's unified memory that copy is a
  same-RAM memcpy, so the payoff is small.

## License

muEye is released under the [MIT License](LICENSE). Note that it links muGrid
(LGPL-3.0) and fetches GLFW (zlib/libpng) and Dear ImGui (MIT) at build time;
those components retain their own licenses.
