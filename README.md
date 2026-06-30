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
- Multi-threaded CPU renderer (OpenMP); adjustable render downscale for interactivity.

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

## GPU backend (deferred)

The CMake options `-DMUEYE_ENABLE_CUDA=ON` / `-DMUEYE_ENABLE_HIP=ON` are declared but
the GPU ray tracer is **not implemented in this pass** (the development machine is
CPU-only macOS, where CUDA/HIP cannot be built or run). The ray-march core
(`src/render/render_core.hh`) is written as a single `__host__ __device__`-decorated
unit precisely so a `src/render/render_gpu.cc` backend can be added later — compiled
with `LANGUAGE CUDA`/`HIP`, reusing muGrid's `Device` and `gpu_runtime.hh` for device
selection and memory — without touching the CPU path. The CPU and GPU images should
then match, which is the cross-check that the shared core is consistent.

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
  render/Renderer.hh   renderer interface + RGBA8 framebuffer
  render/CpuRenderer.* OpenMP CPU ray tracer
  gl/GlTexture.*       framebuffer -> GL texture for ImGui::Image
scripts/make_test_volume.py   writes a 64^3 demo .nc via muGrid
```

## Known limitations (first draft)

- Fields on a sub-point (quadrature) subdivision are read, but derived scalars operate
  on sub-point 0 only.
- The transfer function is a colormap preset + opacity ramp (no node editor yet).
- GPU↔GL display, when the GPU backend lands, will go via a host round-trip first; PBO /
  CUDA-GL interop is a later optimization.

## License

muEye is released under the [MIT License](LICENSE). Note that it links muGrid
(LGPL-3.0) and fetches GLFW (zlib/libpng) and Dear ImGui (MIT) at build time;
those components retain their own licenses.
