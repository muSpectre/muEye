# muEye

A real-time **ray-tracing viewer for [muGrid](https://github.com/muSpectre/muGrid)
data**. muEye opens muGrid NetCDF files and renders 3D scalar/tensor fields as a
volume with a **CPU ray tracer** (and, in a later pass, a GPU one). **2D fields**
(files with only `nx`/`ny`) are supported too — they render on a plane. The UI is built
with [Dear ImGui](https://github.com/ocornut/imgui).

muEye reuses muGrid internally:

- **I/O** — `muGrid::FileIONetCDF` reads frames/fields; muEye introspects the file's
  grid (`nx`/`ny`/`nz`), frame count and variables via the netcdf-c API first, since
  muGrid's read API needs a pre-shaped `GlobalFieldCollection`. Single components and
  non-double variables are fetched directly through netcdf-c (only the selected
  component, as `float`); `muEye_check` verifies both readers agree bit for bit.
- **Data representation** — `muGrid::GlobalFieldCollection` / `Field`; field data
  (double precision) is downcast once to `float` on load (the renderer is single
  precision throughout).
- **GPU device handling** (planned) — `muGrid::Device` + `memory/gpu_runtime.hh`.

> **Standalone repo.** muEye is intended to live in its own repository, as a sibling
> of `muGrid`. It does **not** modify muGrid and is not part of muGrid's build; it only
> consumes muGrid as a subproject.

## Features

- Direct volume rendering (DVR) with an editable transfer function: Viridis / Grayscale
  / Cool-Warm colormaps, an opacity ramp with gamma and a **cutoff** (values below it
  are fully transparent), and a choice of **ascending / descending / symmetric** ramp
  (symmetric = transparent at the centre of the range, for diverging maps). The preview
  shows colour *and* opacity.
- Isosurface ray-casting with on-the-fly gradient (Phong) shading.
- **Empty-space skipping**: a per-brick min/max summary of the volume lets the ray
  marcher jump over regions that are transparent (DVR) or cannot contain the iso level;
  the result is bit-identical to the plain march (`muEye_check` asserts it), 2–3× faster
  on sparse volumes (CPU backend; the GPU backends do not use it yet).
- **Adaptive quality**: while the mouse is held the viewport renders at a coarser
  resolution chosen from the last full-quality render time, then refines on release
  (Render panel → "Adaptive quality while interacting").
- Orbit camera (left-drag orbit, right/middle-drag pan, wheel zoom; drags continue when
  the cursor leaves the viewport). 2D fields open face-on. **R** resets the view.
- Field / frame / component selection, plus derived scalars: vector **magnitude** and,
  for 2×2 and 3×3 tensor fields, **von Mises** and **trace**. Tensor components are
  labelled `(row,col)`; muGrid flattens them column-major. Only the reductions that
  apply to the selected field are offered.
- **Lock range** (Render panel): map colours and the iso slider to a fixed value range
  instead of each frame's own min/max, so frames of a time series are comparable.
- Toggleable **box outline** around the rendered volume (Render panel → "Show box").
- **Non-orthogonal (Bravais) cells**: render the volume in a sheared unit cell given by a
  macroscopic deformation gradient **F** (cell edge vectors = columns of `C = F · box`).
  F is read from the file's `deformation_gradient` global attribute (or `average_strain`
  ε, as `F = I + ε`) and can be edited live in the **Cell** panel. Identity F is the
  usual orthogonal box. See *Feeding cell geometry from muGrid* below.
- **Deformed geometry from a displacement field**: pick any vector field with the grid's
  spatial dimension (3 components in 3D, 2 in 2D) as a **Displacement** in the Dataset
  panel to render the *deformed* configuration `x = C·s + u`. Default is undeformed;
  a **Warp scale** gain tunes the visualized displacement (the file carries no physical
  length, so scale is user-set). The ray marcher inverts the warp per sample
  (fixed-point), so it works for DVR and isosurface on every backend. Periodic tiling is
  disabled while warping.
- **Periodic images**: tile the volume periodically with a per-axis replica count
  (Render panel → "Periodic images" + "Replicas"); the ray marcher wraps sampling back
  into the unit cell, so memory use is independent of the replica count.
- **Periodic shift**: translate the structure through the periodic boundaries by whole
  voxels per axis (Cell panel → "Periodic shift"; "Half cell" moves the cell corner to
  the centre). The loaded data (and any displacement field) is rolled in place, so a
  structure straddling the boundary is shown contiguous, interpolated across the old
  seam. Viewer-only: the file is never modified; the shift persists across frames and
  fields and resets when a new file is opened.
- **Pluggable rendering backends** behind one interface (`render/Renderer.hh`),
  selectable at runtime in the Device panel:
  - **CPU** — multi-threaded (OpenMP, or a `std::thread` fallback so it is parallel
    even on Apple clang).
  - **Metal** — Apple-GPU compute shader (default on macOS).
  - **CUDA / HIP** — single-source kernel sharing `render_core.hh` with the CPU path
    (opt-in; requires the respective toolchain).
- Adjustable render downscale for interactivity.
- **PNG snapshots** of the rendered scene (Render panel → "Save PNG" / "Save as...",
  Ctrl+S), written at full viewport resolution regardless of the downscale setting and,
  by default, next to the data file. Existing files are never overwritten without asking.
- Files open from the path box, **Browse...** (Ctrl+O), the command line, or by
  **dragging them onto the window**. A status bar along the bottom shows the result of
  the last action and the active backend.
- **Keyboard shortcuts**: Left/Right and Home/End step frames, R resets the view, B
  toggles the box, P periodic images, I switches DVR/isosurface (listed under Stats →
  "Keyboard shortcuts").
- Uses the host platform's **native UI font** (San Francisco on macOS, Segoe UI on
  Windows, Ubuntu/Cantarell/Noto→DejaVu Sans on Linux), HiDPI-aware, with a graceful
  fallback to Dear ImGui's built-in font. The window layout is remembered per user
  (`~/.config/muEye/imgui.ini`, `~/Library/Application Support/muEye/` or
  `%APPDATA%\muEye\`).

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

## Download

Prebuilt, self-contained binaries (NetCDF/HDF5 and all other non-system libraries
bundled) are attached to each [GitHub release](https://github.com/muSpectre/muEye/releases);
every CI run also uploads them as workflow artifacts.

- **Linux** (`.AppImage`, x86_64 / arm64): `chmod +x muEye-*.AppImage` and run it. Built
  on Ubuntu 24.04, so it needs glibc ≥ 2.39 plus the host's OpenGL/X11 drivers.
- **macOS** (`.zip` with `muEye.app`, Apple Silicon): the app is ad-hoc signed, not
  notarized, so Gatekeeper blocks the first launch. Allow it under *System Settings →
  Privacy & Security → Open Anyway*, or run `xattr -dr com.apple.quarantine muEye.app`.
- **Windows** (`.zip`, x86_64 / arm64): unzip and run `muEye.exe`. The MSVC runtime is
  included, so no VC++ redistributable is needed.

The x86_64 Linux and Windows packages include the **CUDA** backend (Volta and newer;
needs an NVIDIA driver recent enough for CUDA 12.6). Without an NVIDIA GPU they run on
the CPU backend. HIP is not packaged: build it from source against your ROCm install.

To cut a release, push a tag `v<version>` (keep it in sync with `project(muEye VERSION ...)`
in `CMakeLists.txt`); CI builds every platform and publishes the release.

## Run

```bash
# generate a test volume (needs muGrid's Python package, e.g. the workspace venv
# ../muGrid/venv with the muGrid build tree on PYTHONPATH)
source ../muGrid/venv/bin/activate
python scripts/make_test_volume.py demo.nc            # 64^3 cube, 2 frames
python scripts/make_test_volume.py --warp --shear 0.2 warped.nc 32   # + displacement, sheared cell

# launch the viewer (optionally auto-open a file)
# macOS builds a .app bundle (so the Dock/Finder icon is the muSpectre logo):
./build/muEye.app/Contents/MacOS/muEye demo.nc   # or: open ./build/muEye.app
# Linux / Windows produce a bare executable:
./build/muEye demo.nc
```

Then: pick a file with **Browse...** (native dialog), type a path and **Load**, or
pass it on the command line; pick a field/frame, toggle **DVR**/**Isosurface**,
drag to orbit, scroll to zoom.

## Feeding cell geometry from muGrid

muGrid NetCDF files describe geometry only as the integer grid dimensions (`nx/ny/nz`);
they carry no physical cell shape or strain. To render a **deformed / non-orthogonal
cell**, muEye reads an optional `deformation_gradient` global attribute (a flat,
row-major 3×3 in 3D, or 2×2 in 2D). A producer built on muGrid writes it with the
existing `write_global_attribute` API — **no muGrid change is required**:

```python
import numpy as np, muGrid

file = muGrid.FileIONetCDF(path, open_mode="overwrite")
file.register_field_collection(fc)

# Macroscopic deformation gradient F (row-major). F = I ⇒ undeformed.
F = np.eye(3); F[0, 1] = 0.2                       # example simple shear
file.write_global_attribute("deformation_gradient", list(F.ravel()))

# ... append_frame() / write fields as usual ...
```

Notes:
- Write the attribute **once, up front, before appending frames** — muGrid defines global
  attributes at file creation.
- For a small-strain workflow, store the **average strain** ε instead under
  `average_strain`; muEye applies `F = I + ε`.
- muEye's rendered cell is `C = F · diag(box)`, where `box` is the grid shape normalized
  so the longest axis is 1 (or the `domain_lengths` attribute when present).
  `scripts/make_test_volume.py --shear S` writes such a file for testing, and
  `./build/muEye_check --view file.nc [field]` renders it headlessly to
  `<file>.view.png` / `<file>.view_rep.png` next to the data.
- A per-frame `applied_deformation_gradient` variable (as muTopOpt writes) supersedes
  the global attribute frame by frame. Editing F by hand in the Cell panel overrides
  both until "Use file's F" is pressed.

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
  ui/panels.cc         ImGui windows (dataset/cell/render/transfer/device/stats/status/viewport), shortcuts, dialogs
  ui/OrbitCamera.*     mouse-driven orbit camera
  ui/TransferFunction.* colormap + opacity LUT
  io/Volume.*          dense float volume + scalarization + brick min/max summary
  io/VolumeLoader.*    netcdf-c introspection; muGrid-backed and direct netcdf-c field readers
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
- Empty-space skipping runs on the CPU backend only; the Metal and CUDA/HIP kernels
  render the identical image but march every sample (no brick summary is uploaded yet).
- Rendering is synchronous: a very large volume at full resolution still blocks the UI
  for the duration of one frame (adaptive quality keeps interaction fluid, but the
  final refine is a single blocking render).
- The CUDA/HIP backend displays via a CUDA↔GL PBO (no device→host round trip). The Metal
  backend still copies its result through the host before the GL upload: a true Metal→GL
  zero-copy needs an IOSurface exposed to GL as `GL_TEXTURE_RECTANGLE`, which Dear ImGui's
  GL3 backend can't sample — and on Apple Silicon's unified memory that copy is a
  same-RAM memcpy, so the payoff is small.

## License

muEye is released under the [MIT License](LICENSE). Note that it links muGrid
(LGPL-3.0) and fetches GLFW (zlib/libpng) and Dear ImGui (MIT) at build time;
those components retain their own licenses.
