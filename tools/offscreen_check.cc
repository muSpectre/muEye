/**
 * @file   offscreen_check.cc
 *
 * @brief  Headless end-to-end verification of the muEye data + render pipeline.
 *
 * Writes demo muGrid NetCDF files (a 3D Gaussian/cube blob and a 2D disk) using
 * muGrid's C++ API, reads them back through muEye's VolumeLoader, scalarizes,
 * ray-traces a frame with the CPU renderer, and cross-checks every other
 * compiled-in backend against the CPU reference. Prints statistics so the whole
 * pipeline can be validated without a display or Python.
 *
 * This exercises exactly the code paths the GUI uses, minus ImGui/OpenGL — for
 * both 3D volumes and 2D fields (which render on a plane).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "collection/field_collection.hh"
#include "collection/field_collection_global.hh"
#include "core/types.hh"
#include "field/field.hh"
#include "io/PngWriter.hh"
#include "io/Volume.hh"
#include "io/VolumeLoader.hh"
#include "io/file_io_base.hh"
#include "io/file_io_netcdf.hh"
#include "render/CpuRenderer.hh"
#include "render/RendererFactory.hh"
#include "ui/OrbitCamera.hh"
#include "ui/TransferFunction.hh"

namespace {

// A solid axis-aligned cube of half-extent `half` centred at (cx,cy,cz):
// value 1 inside, 0 outside (trilinear sampling softens the faces by a voxel).
void fill_cube(double *d, int n, double cx, double cy, double cz, double half) {
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        double x = (i + 0.5) / n, y = (j + 0.5) / n, z = (k + 0.5) / n;
        double cheby = std::fmax(std::fabs(x - cx),
                                 std::fmax(std::fabs(y - cy), std::fabs(z - cz)));
        d[i + n * (j + n * k)] = cheby <= half ? 1.0 : 0.0;
      }
}

void write_demo(const std::string &path, int n) {
  muGrid::GlobalFieldCollection fc(muGrid::DynGridIndex{n, n, n});
  muGrid::Field &phi = fc.register_real_field("phi", 1);
  double *d = static_cast<double *>(phi.get_void_data_ptr());
  // A 9-component (3x3 tensor) field whose value is distinct for every
  // (voxel, component) pair, used to cross-check per-component reads of the
  // direct netcdf path against muGrid's (component or axis mix-ups cannot
  // cancel out).
  muGrid::Field &sig = fc.register_real_field("sig", 9);
  double *s = static_cast<double *>(sig.get_void_data_ptr());
  {
    std::size_t nb_entries = static_cast<std::size_t>(n) * n * n * 9;
    // Layout-agnostic distinctness: value = flat buffer index. Both readers
    // fetch the same file, so only their mutual agreement matters.
    for (std::size_t e = 0; e < nb_entries; ++e)
      s[e] = static_cast<double>(e);
  }

  muGrid::FileIONetCDF file(path, muGrid::FileIOBase::OpenMode::Overwrite);
  file.register_field_collection(fc);

  fill_cube(d, n, 0.5, 0.5, 0.5, 0.25);
  file.append_frame().write();
  fill_cube(d, n, 0.62, 0.40, 0.55, 0.18);
  file.append_frame().write();
  file.close();
  std::printf("wrote %s (%d^3, fields 'phi' cube + 'sig' 3x3, 2 frames)\n",
              path.c_str(), n);
}

// A 2D grid (no nz dimension): a filled disk of radius 0.25, value 1 inside.
void write_demo_2d(const std::string &path, int n) {
  muGrid::GlobalFieldCollection fc(muGrid::DynGridIndex{n, n});
  muGrid::Field &phi = fc.register_real_field("phi", 1);
  double *d = static_cast<double *>(phi.get_void_data_ptr());

  muGrid::FileIONetCDF file(path, muGrid::FileIOBase::OpenMode::Overwrite);
  file.register_field_collection(fc);

  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      d[i + n * j] = std::sqrt(x * x + y * y) <= 0.25 ? 1.0 : 0.0;
    }
  file.append_frame().write();
  file.close();
  std::printf("wrote %s (%d^2 2D, field 'phi' disk, 1 frame)\n", path.c_str(),
              n);
}

// A smooth synthetic displacement field (4 floats/voxel) for the warp check: a
// linear dilation about the cell centre, u = A*(s - 0.5) per axis (in grid
// units). Linear => the fixed-point inverse converges in a couple of iterations,
// and every backend must recover the same deformed image. z stays ~0 for a 2D
// (nz==1) grid, so it doubles as the 2D test.
mueye::DisplacementField make_disp(int nx, int ny, int nz, float amp) {
  mueye::DisplacementField d;
  d.nx = nx;
  d.ny = ny;
  d.nz = nz;
  d.data.assign(d.size() * 4, 0.0f);
  float mx = 0.0f;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        float sx = (i + 0.5f) / nx - 0.5f;
        float sy = (j + 0.5f) / ny - 0.5f;
        float sz = (k + 0.5f) / nz - 0.5f;
        std::size_t e =
            (i + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k)) * 4;
        float ux = amp * sx, uy = amp * sy, uz = amp * sz;
        d.data[e + 0] = ux;
        d.data[e + 1] = uy;
        d.data[e + 2] = uz;
        float m = std::sqrt(ux * ux + uy * uy + uz * uz);
        if (m > mx) mx = m;
      }
  d.max_mag = mx;
  return d;
}

std::size_t count_nonbg(const mueye::Framebuffer &fb) {
  std::size_t n = 0;
  std::uint8_t bgr = static_cast<std::uint8_t>(0.05f * 255 + 0.5f);
  for (int i = 0; i < fb.width * fb.height; ++i)
    if (std::abs(int(fb.rgba[i * 4]) - int(bgr)) > 6 ||
        fb.rgba[i * 4 + 1] > 16 || fb.rgba[i * 4 + 2] > 24)
      ++n;
  return n;
}

// A Volume holding a cube centred in *normalized grid* (uvw) coordinates:
// value 1 where max(|u-0.5|,|v-0.5|,|w-0.5|) <= 0.25. The value depends only
// on uvw, so grids with permuted extents hold identical data by construction.
mueye::Volume make_cube_volume(int nx, int ny, int nz) {
  mueye::Volume v;
  v.nx = nx;
  v.ny = ny;
  v.nz = nz;
  v.data.assign(v.size(), 0.0f);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        double u = (i + 0.5) / nx, w = (j + 0.5) / ny, s = (k + 0.5) / nz;
        double cheby = std::fmax(std::fabs(u - 0.5),
                                 std::fmax(std::fabs(w - 0.5),
                                           std::fabs(s - 0.5)));
        v.data[i + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k)] =
            cheby <= 0.25 ? 1.0f : 0.0f;
      }
  v.vmin = 0.0f;
  v.vmax = 1.0f;
  return v;
}

/**
 * DVR opacity must not depend on the axis order of the data. Render the same
 * cube on a 48x16x16 grid and on its x<->z permutation (16x16x48), with
 * cameras looking down -y whose right/up bases are permuted to match. Under
 * the swap the two scenes are congruent, so the images must agree pixel-wise
 * (up to float rounding). A per-voxel opacity correction that uses nx instead
 * of the longest axis makes the permuted render ~3x more transparent and
 * fails this check. Also cross-checks every other available backend against
 * the CPU reference on the anisotropic grid, which exercises the hand-written
 * Metal mirror's opacity term.
 * @returns 0 on success, non-zero on any failure.
 */
int check_anisotropic() {
  std::printf("\n=== checking anisotropic grid (opacity vs axis order) ===\n");

  mueye::TransferFunction tf;
  tf.set_colormap(mueye::Colormap::Viridis);

  mueye::Volume va = make_cube_volume(48, 16, 16);
  mueye::Volume vb = make_cube_volume(16, 16, 48);

  auto make_params = [&tf](const mueye::Volume &v) {
    int md = v.nx > v.ny ? v.nx : v.ny;
    if (v.nz > md) md = v.nz;
    float f = 1.0f / static_cast<float>(md);
    mueye::RenderParams p;
    p.nx = v.nx;
    p.ny = v.ny;
    p.nz = v.nz;
    p.box = mueye::Vec3{v.nx * f, v.ny * f, v.nz * f};
    p.inv_cell = mueye::axis_aligned_inv_cell(p.box);
    p.step = 0.5f * f;
    p.data_min = 0.0f;
    p.data_max = 1.0f;
    p.lut_size = tf.size();
    // Low density so per-sample opacity stays well below saturation — a wrong
    // opacity-correction factor then shows up as a large brightness change
    // instead of vanishing into an equally opaque pixel.
    p.density_scale = 0.3f;
    p.iso_value = 0.5f;
    p.mode = mueye::RenderMode::DVR;
    p.bg = mueye::Vec3{0.05f, 0.06f, 0.08f};
    return p;
  };

  // Top-down camera (forward = -y); right/up select which world axes map to
  // the image axes, so a permuted grid can be given a matching view.
  auto make_cam = [](const mueye::RenderParams &p, const mueye::Vec3 &right,
                     const mueye::Vec3 &up) {
    mueye::Camera cam;
    // Close enough that the cube fills a good fraction of the image.
    cam.eye = mueye::Vec3{0.5f * p.box.x, 0.5f * p.box.y + 1.0f,
                          0.5f * p.box.z};
    cam.forward = mueye::Vec3{0.0f, -1.0f, 0.0f};
    cam.right = right;
    cam.up = up;
    cam.tan_half_fov = 0.45f;
    cam.aspect = 1.0f;
    return cam;
  };

  mueye::CpuRenderer cpu;
  cpu.set_transfer_function(tf.data(), tf.size());
  mueye::Framebuffer fa, fb;
  fa.resize(256, 256);
  fb.resize(256, 256);

  mueye::RenderParams pa = make_params(va);
  mueye::Camera cam_a = make_cam(pa, mueye::Vec3{1.0f, 0.0f, 0.0f},
                                 mueye::Vec3{0.0f, 0.0f, 1.0f});
  cpu.set_volume(va.data.data(), va.nx, va.ny, va.nz);
  cpu.render(pa, cam_a, fa);

  mueye::RenderParams pb = make_params(vb);
  mueye::Camera cam_b = make_cam(pb, mueye::Vec3{0.0f, 0.0f, 1.0f},
                                 mueye::Vec3{1.0f, 0.0f, 0.0f});
  cpu.set_volume(vb.data.data(), vb.nx, vb.ny, vb.nz);
  cpu.render(pb, cam_b, fb);

  // Sanity: the render must actually contain the cube.
  std::size_t nonbg = count_nonbg(fa);
  double frac = double(nonbg) / (fa.width * fa.height);
  std::printf("CPU DVR 48x16x16: %zu non-background px (%.1f%%)\n", nonbg,
              100.0 * frac);
  if (frac < 0.02) {
    std::fprintf(stderr, "anisotropic render produced almost nothing.\n");
    return 1;
  }

  double sum = 0.0;
  for (std::size_t i = 0; i < fa.rgba.size(); ++i)
    sum += std::abs(int(fa.rgba[i]) - int(fb.rgba[i]));
  double mean_abs_diff = sum / fa.rgba.size();
  std::printf("axis-permuted render (16x16x48): mean|Δ| = %.3f / 255\n",
              mean_abs_diff);
  int failures = 0;
  if (mean_abs_diff > 0.5) {
    std::fprintf(stderr,
                 "DVR opacity depends on the data's axis order (permuted "
                 "grid renders differently).\n");
    ++failures;
  }

  // Every other available backend must agree with the CPU reference on the
  // anisotropic grid too.
  cpu.set_volume(va.data.data(), va.nx, va.ny, va.nz);
  cpu.render(pa, cam_a, fa);
  for (const mueye::BackendInfo &bi : mueye::enumerate_backends()) {
    if (bi.backend == mueye::Backend::CPU) continue;
    if (!bi.available) {
      std::printf("backend %-14s : unavailable (%s)\n", bi.name, bi.note);
      continue;
    }
    auto r = mueye::create_renderer(bi.backend);
    if (!r) {
      std::printf("backend %-14s : create failed\n", bi.name);
      continue;
    }
    mueye::Framebuffer fg;
    fg.resize(256, 256);
    r->set_volume(va.data.data(), va.nx, va.ny, va.nz);
    r->set_transfer_function(tf.data(), tf.size());
    r->render(pa, cam_a, fg);
    sum = 0.0;
    for (std::size_t i = 0; i < fg.rgba.size(); ++i)
      sum += std::abs(int(fg.rgba[i]) - int(fa.rgba[i]));
    double backend_diff = sum / fg.rgba.size();
    std::printf("backend %-14s : anisotropic DVR mean|Δ| vs CPU = %.3f / 255\n",
                bi.name, backend_diff);
    if (backend_diff > 4.0) {
      std::fprintf(stderr, "backend %s diverged on the anisotropic grid.\n",
                   bi.name);
      ++failures;
    }
  }
  return failures;
}

/**
 * Scalarization of tensor fields: trace and von Mises for a 2x2 (2D) and a 3x3
 * (3D) tensor, on a one-voxel "grid", against hand-computed values. The
 * off-diagonal terms are deliberately asymmetric so the check also pins down
 * that the reductions symmetrize them (and are thus independent of the
 * row-/column-major flattening of the file).
 * @returns 0 on success, non-zero on any failure.
 */
int check_reductions() {
  std::printf("\n=== checking tensor reductions (2x2 and 3x3) ===\n");
  int failures = 0;
  auto expect = [&](const char *what, double got, double want) {
    bool ok = std::fabs(got - want) <= 1e-5 * std::fmax(1.0, std::fabs(want));
    std::printf("  %-22s = %-12.6g (expected %.6g) %s\n", what, got, want,
                ok ? "ok" : "MISMATCH");
    if (!ok) ++failures;
  };
  auto reduce1 = [](const double *comps, int nb, mueye::Scalarize mode) {
    mueye::Volume v;
    v.from_field(comps, 1, 1, 1, nb, 1, 1, 1, mode, 0);
    return static_cast<double>(v.data[0]);
  };
  // 3x3, flat index r + 3c (column-major) or 3r + c (row-major); both give the
  // same diagonal and the same off-diagonal pairs {1,3},{2,6},{5,7}.
  const double t3[9] = {1.0, 0.5, 0.2, 0.1, 2.0, 0.4, 0.6, 0.0, 3.0};
  expect("trace 3x3", reduce1(t3, 9, mueye::Scalarize::Trace), 6.0);
  {
    double sxy = 0.5 * (0.5 + 0.1), sxz = 0.5 * (0.2 + 0.6), syz = 0.5 * (0.4 + 0.0);
    double j2 = 0.5 * (1.0 + 1.0 + 4.0) + 3.0 * (sxy * sxy + syz * syz + sxz * sxz);
    expect("von Mises 3x3", reduce1(t3, 9, mueye::Scalarize::VonMises),
           std::sqrt(j2));
  }
  // 2x2: diagonal at 0 and 3, off-diagonal pair {1,2}.
  const double t2[4] = {2.0, 0.3, 0.1, -1.0};
  expect("trace 2x2", reduce1(t2, 4, mueye::Scalarize::Trace), 1.0);
  {
    double sxy = 0.5 * (0.3 + 0.1);
    double j2 = 4.0 - (2.0 * -1.0) + 1.0 + 3.0 * sxy * sxy;
    expect("von Mises 2x2", reduce1(t2, 4, mueye::Scalarize::VonMises),
           std::sqrt(j2));
  }
  expect("magnitude 2x2", reduce1(t2, 4, mueye::Scalarize::Magnitude),
         std::sqrt(4.0 + 0.09 + 0.01 + 1.0));
  return failures;
}

bool write_ppm(const std::string &path, const mueye::Framebuffer &fb) {
  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  std::fprintf(f, "P6\n%d %d\n255\n", fb.width, fb.height);
  for (int y = 0; y < fb.height; ++y)
    for (int x = 0; x < fb.width; ++x) {
      std::size_t idx = (static_cast<std::size_t>(y) * fb.width + x) * 4;
      std::fputc(fb.rgba[idx + 0], f);
      std::fputc(fb.rgba[idx + 1], f);
      std::fputc(fb.rgba[idx + 2], f);
    }
  std::fclose(f);
  return true;
}

/**
 * Introspect @p path, load field 0, render a CPU DVR + isosurface frame, and
 * cross-check every other available backend against the CPU reference.
 * @returns 0 on success, non-zero on any failure.
 */
int check_file(const std::string &path, const char *ppm_out) {
  std::printf("\n=== checking %s ===\n", path.c_str());

  mueye::VolumeLoader loader;
  mueye::FileMeta meta = loader.open(path);
  if (!meta.valid) {
    std::fprintf(stderr, "introspection failed: %s\n", meta.error.c_str());
    return 1;
  }
  std::printf("introspected: %d x %d x %d (spatial_dim=%d), %d frame(s), "
              "%zu field(s)\n",
              meta.nx, meta.ny, meta.nz, meta.spatial_dim, meta.nb_frames,
              meta.fields.size());
  if (meta.fields.empty()) return 1;

  mueye::Volume vol;
  std::string err = loader.load(path, meta, meta.fields[0], 0,
                                mueye::Scalarize::Component, 0, vol);
  if (!err.empty()) {
    std::fprintf(stderr, "load failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("volume loaded: %dx%dx%d range [%.5f, %.5f]\n", vol.nx, vol.ny,
              vol.nz, vol.vmin, vol.vmax);
  if (!(vol.vmax > 0.5f && vol.vmin < 0.5f)) {
    std::fprintf(stderr, "unexpected value range; shape not recovered?\n");
    return 1;
  }

  // Cross-check the direct netcdf-c read path (used for non-double variables,
  // e.g. muFFTTO's float32 output) against the muGrid read path on the same
  // double variable: both must produce the identical volume. The last frame of
  // the 3D demo holds an off-centre cube, so an axis mix-up cannot cancel out.
  {
    int fcheck = meta.nb_frames - 1;
    mueye::Volume vg, vd;
    mueye::FieldInfo direct = meta.fields[0];
    direct.is_double = false;  // force the netcdf-c path
    std::string e1 = loader.load(path, meta, meta.fields[0], fcheck,
                                 mueye::Scalarize::Component, 0, vg);
    std::string e2 = loader.load(path, meta, direct, fcheck,
                                 mueye::Scalarize::Component, 0, vd);
    if (!e1.empty() || !e2.empty() || vg.data.size() != vd.data.size() ||
        vg.data.empty()) {
      std::fprintf(stderr, "direct-read cross-check failed to load (%s%s)\n",
                   e1.c_str(), e2.c_str());
      return 1;
    }
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < vg.data.size(); ++i) {
      float d = std::fabs(vg.data[i] - vd.data[i]);
      if (d > max_diff) max_diff = d;
    }
    std::printf("direct netcdf read vs muGrid read: max|Δ| = %g\n",
                double(max_diff));
    if (max_diff > 0.0f) {
      std::fprintf(stderr, "direct read path disagrees with muGrid read.\n");
      return 1;
    }
  }

  // Same cross-check for multi-component fields: every component plus the
  // magnitude reduction must agree between the muGrid path and the direct
  // netcdf path (which fetches only the selected component in Component mode).
  for (const mueye::FieldInfo &finfo : meta.fields) {
    if (finfo.nb_components <= 1) continue;
    mueye::FieldInfo direct = finfo;
    direct.is_double = false;  // force the netcdf-c path
    for (int c = 0; c <= finfo.nb_components; ++c) {
      // c == nb_components is the magnitude pass (component index unused).
      bool magnitude = c == finfo.nb_components;
      mueye::Scalarize sm =
          magnitude ? mueye::Scalarize::Magnitude : mueye::Scalarize::Component;
      mueye::Volume vg, vd;
      std::string e1 = loader.load(path, meta, finfo, 0, sm, c, vg);
      std::string e2 = loader.load(path, meta, direct, 0, sm, c, vd);
      if (!e1.empty() || !e2.empty() || vg.data.size() != vd.data.size() ||
          vg.data.empty()) {
        std::fprintf(stderr,
                     "multi-component cross-check failed to load '%s' (%s%s)\n",
                     finfo.name.c_str(), e1.c_str(), e2.c_str());
        return 1;
      }
      float max_diff = 0.0f;
      for (std::size_t i = 0; i < vg.data.size(); ++i) {
        float dv = std::fabs(vg.data[i] - vd.data[i]);
        if (dv > max_diff) max_diff = dv;
      }
      if (max_diff > 0.0f) {
        std::fprintf(stderr,
                     "direct read of '%s' (%s %d) disagrees with muGrid read "
                     "(max|Δ| = %g).\n",
                     finfo.name.c_str(), magnitude ? "magnitude" : "component",
                     c, double(max_diff));
        return 1;
      }
    }
    std::printf("field '%s': %d components + magnitude, direct vs muGrid "
                "reads agree\n",
                finfo.name.c_str(), finfo.nb_components);
  }

  mueye::TransferFunction tf;
  tf.set_colormap(mueye::Colormap::Viridis);
  tf.set_opacity_scale(1.0f);

  // Frame the box centre the same way App does (planar/non-cubic aware).
  int md = vol.nx > vol.ny ? vol.nx : vol.ny;
  if (vol.nz > md) md = vol.nz;
  if (md < 1) md = 1;
  float f = 1.0f / static_cast<float>(md);
  mueye::OrbitCamera cam;
  cam.frame_box(mueye::Vec3{vol.nx * f, vol.ny * f, vol.nz * f});
  mueye::Camera camv = cam.to_camera(1.0f);

  mueye::RenderParams p;
  p.nx = vol.nx;
  p.ny = vol.ny;
  p.nz = vol.nz;
  p.box = mueye::Vec3{vol.nx * f, vol.ny * f, vol.nz * f};
  p.inv_cell = mueye::axis_aligned_inv_cell(p.box);
  p.step = 0.5f * f;
  p.data_min = vol.vmin;
  p.data_max = vol.vmax;
  p.lut_size = tf.size();
  p.density_scale = 1.0f;
  p.iso_value = 0.5f * (vol.vmin + vol.vmax);
  p.mode = mueye::RenderMode::DVR;
  p.bg = mueye::Vec3{0.05f, 0.06f, 0.08f};

  auto render_with = [&](mueye::Renderer &r, mueye::Framebuffer &fb,
                         mueye::RenderMode mode) {
    p.mode = mode;
    r.set_volume(vol.data.data(), vol.nx, vol.ny, vol.nz);
    r.set_transfer_function(tf.data(), tf.size());
    r.render(p, camv, fb);
  };

  // CPU reference render (DVR + isosurface).
  mueye::CpuRenderer cpu;
  mueye::Framebuffer fb_cpu;
  fb_cpu.resize(256, 256);
  render_with(cpu, fb_cpu, mueye::RenderMode::DVR);
  std::size_t nonbg = count_nonbg(fb_cpu);
  double frac = double(nonbg) / (fb_cpu.width * fb_cpu.height);
  std::printf("CPU DVR 256x256: %zu non-background px (%.1f%%)\n", nonbg,
              100.0 * frac);
  if (ppm_out) {
    if (write_ppm(ppm_out, fb_cpu)) std::printf("wrote %s\n", ppm_out);
    // Also exercise the PNG writer the GUI's "Save PNG" button uses.
    std::string png_out = std::string(ppm_out) + ".png";
    if (!mueye::write_png(png_out, fb_cpu)) {
      std::fprintf(stderr, "PNG write failed: %s\n", png_out.c_str());
      return 1;
    }
    std::printf("wrote %s\n", png_out.c_str());
  }
  if (frac < 0.02) {
    std::fprintf(stderr, "render produced almost nothing; check pipeline.\n");
    return 1;
  }
  render_with(cpu, fb_cpu, mueye::RenderMode::Isosurface);
  std::size_t iso_hits = 0;
  for (int i = 0; i < fb_cpu.width * fb_cpu.height; ++i)
    if (fb_cpu.rgba[i * 4] > 60) ++iso_hits;
  std::printf("CPU isosurface: %zu lit px\n", iso_hits);

  // Render a CPU DVR reference with the current params, then require every
  // other available backend to agree with it.
  auto compare_backends = [&](const char *what) {
    render_with(cpu, fb_cpu, mueye::RenderMode::DVR);
    int bad = 0;
    for (const mueye::BackendInfo &bi : mueye::enumerate_backends()) {
      if (bi.backend == mueye::Backend::CPU) continue;
      if (!bi.available) {
        std::printf("backend %-14s : unavailable (%s)\n", bi.name, bi.note);
        continue;
      }
      auto r = mueye::create_renderer(bi.backend);
      if (!r) {
        std::printf("backend %-14s : create failed\n", bi.name);
        continue;
      }
      mueye::Framebuffer fb;
      fb.resize(256, 256);
      render_with(*r, fb, mueye::RenderMode::DVR);

      double sum = 0.0;
      for (std::size_t i = 0; i < fb.rgba.size(); ++i)
        sum += std::abs(int(fb.rgba[i]) - int(fb_cpu.rgba[i]));
      double mean_abs_diff = sum / fb.rgba.size();
      std::printf("backend %-14s : %s mean|Δ| vs CPU = %.3f / 255\n", bi.name,
                  what, mean_abs_diff);
      if (mean_abs_diff > 4.0) ++bad;
    }
    return bad;
  };

  int mismatches = compare_backends("DVR");

  // Periodic tiling: 2x2x2 replicas, camera reframed on the enlarged box.
  p.rep_x = 2;
  p.rep_y = 2;
  p.rep_z = 2;
  cam.frame_box(
      mueye::Vec3{2 * vol.nx * f, 2 * vol.ny * f, 2 * vol.nz * f});
  camv = cam.to_camera(1.0f);
  render_with(cpu, fb_cpu, mueye::RenderMode::DVR);
  std::size_t nonbg_rep = count_nonbg(fb_cpu);
  double frac_rep = double(nonbg_rep) / (fb_cpu.width * fb_cpu.height);
  std::printf("CPU DVR 2x2x2 replicas: %zu non-background px (%.1f%%)\n",
              nonbg_rep, 100.0 * frac_rep);
  if (ppm_out &&
      mueye::write_png(std::string(ppm_out) + ".rep.png", fb_cpu))
    std::printf("wrote %s.rep.png\n", ppm_out);
  if (frac_rep < 0.02) {
    std::fprintf(stderr, "replicated render produced almost nothing.\n");
    return 1;
  }
  mismatches += compare_backends("DVR 2x2x2");

  // Sheared (Bravais) cell: a non-orthogonal inv_cell must render identically
  // across backends too (exercises the Metal mirror's cell-matrix math). Reuse
  // the single-cell params but replace the axis-aligned map with C = F*box.
  p.rep_x = p.rep_y = p.rep_z = 1;
  mueye::Mat3 Fsh{{1.0f, 0.4f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  mueye::Mat3 Csh = mueye::mat3_matmul(Fsh, mueye::mat3_diag(p.box));
  p.inv_cell = mueye::mat3_inverse(Csh);
  {
    mueye::Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < 8; ++i) {
      mueye::Vec3 fr{(i & 1) ? 1.0f : 0.0f, (i & 2) ? 1.0f : 0.0f,
                     (i & 4) ? 1.0f : 0.0f};
      mueye::Vec3 w = mueye::mat3_mul(Csh, fr);
      lo.x = std::fmin(lo.x, w.x); hi.x = std::fmax(hi.x, w.x);
      lo.y = std::fmin(lo.y, w.y); hi.y = std::fmax(hi.y, w.y);
      lo.z = std::fmin(lo.z, w.z); hi.z = std::fmax(hi.z, w.z);
    }
    mueye::Vec3 center = (lo + hi) * 0.5f;
    float extent = std::fmax(hi.x - lo.x, std::fmax(hi.y - lo.y, hi.z - lo.z));
    cam.frame_aabb(center, extent);
    camv = cam.to_camera(1.0f);
  }
  render_with(cpu, fb_cpu, mueye::RenderMode::DVR);
  std::size_t nonbg_sh = count_nonbg(fb_cpu);
  std::printf("CPU DVR sheared cell (F_xy=0.4): %zu non-background px\n",
              nonbg_sh);
  if (ppm_out && mueye::write_png(std::string(ppm_out) + ".shear.png", fb_cpu))
    std::printf("wrote %s.shear.png\n", ppm_out);
  // A modest absolute floor (a sheared thin 2D slab is legitimately sparse);
  // the real regression signal is the cross-backend agreement below.
  if (nonbg_sh < 200) {
    std::fprintf(stderr, "sheared render produced almost nothing.\n");
    return 1;
  }
  mismatches += compare_backends("DVR sheared");

  // Deformed geometry: a synthetic displacement warps the volume; every backend
  // must agree with the CPU reference on the inverse-warp (fixed-point) path.
  mueye::DisplacementField df = make_disp(vol.nx, vol.ny, vol.nz, 2.0f);
  p.inv_cell = mueye::axis_aligned_inv_cell(p.box);  // undo the shear above
  p.warp_enabled = 1;
  p.warp_scale = 8.0f;
  p.warp_iters = 8;
  p.rep_x = p.rep_y = p.rep_z = 1;
  {
    int md2 = vol.nx > vol.ny ? vol.nx : vol.ny;
    if (vol.nz > md2) md2 = vol.nz;
    float ff = 1.0f / static_cast<float>(md2 > 0 ? md2 : 1);
    float margin = p.warp_scale * ff * df.max_mag;
    p.warp_lo = mueye::Vec3{-margin, -margin, -margin};
    p.warp_hi = mueye::Vec3{p.box.x + margin, p.box.y + margin, p.box.z + margin};
    mueye::Vec3 center{0.5f * p.box.x, 0.5f * p.box.y, 0.5f * p.box.z};
    cam.frame_aabb(center, p.box.x + 2 * margin);
    camv = cam.to_camera(1.0f);
  }
  // compare_backends renders through render_with, which does not set the
  // displacement, so drive the comparison directly here.
  {
    cpu.set_displacement(df.data.data(), df.nx, df.ny, df.nz);
    p.mode = mueye::RenderMode::DVR;
    cpu.set_volume(vol.data.data(), vol.nx, vol.ny, vol.nz);
    cpu.set_transfer_function(tf.data(), tf.size());
    cpu.render(p, camv, fb_cpu);
    std::size_t nonbg_w = count_nonbg(fb_cpu);
    std::printf("CPU DVR warped (displacement): %zu non-background px\n",
                nonbg_w);
    if (ppm_out && mueye::write_png(std::string(ppm_out) + ".warp.png", fb_cpu))
      std::printf("wrote %s.warp.png\n", ppm_out);
    if (nonbg_w < 200) {
      std::fprintf(stderr, "warped render produced almost nothing.\n");
      return 1;
    }
    for (const mueye::BackendInfo &bi : mueye::enumerate_backends()) {
      if (bi.backend == mueye::Backend::CPU || !bi.available) continue;
      auto r = mueye::create_renderer(bi.backend);
      if (!r) continue;
      mueye::Framebuffer fb;
      fb.resize(256, 256);
      r->set_volume(vol.data.data(), vol.nx, vol.ny, vol.nz);
      r->set_transfer_function(tf.data(), tf.size());
      r->set_displacement(df.data.data(), df.nx, df.ny, df.nz);
      r->render(p, camv, fb);
      double sum = 0.0;
      for (std::size_t i = 0; i < fb.rgba.size(); ++i)
        sum += std::abs(int(fb.rgba[i]) - int(fb_cpu.rgba[i]));
      double mad = sum / fb.rgba.size();
      std::printf("backend %-14s : DVR warped mean|Δ| vs CPU = %.3f / 255\n",
                  bi.name, mad);
      if (mad > 4.0) ++mismatches;
    }
  }

  if (mismatches) {
    std::fprintf(stderr, "%d backend(s) diverged from the CPU reference.\n",
                 mismatches);
    return 1;
  }
  return 0;
}

/**
 * View mode (`muEye_check --view file.nc [field]`): render an existing file
 * with the GUI's default appearance (DVR, default transfer function, white
 * background) to view.png / view_rep.png (2x2x2 periodic replicas). Purely a
 * preview/diagnosis aid; the file is only read, never written.
 */
int view_file(const std::string &path, const char *field_name) {
  mueye::VolumeLoader loader;
  mueye::FileMeta meta = loader.open(path);
  if (!meta.valid) {
    std::fprintf(stderr, "introspection failed: %s\n", meta.error.c_str());
    return 1;
  }
  std::printf("%s: %d x %d x %d (spatial_dim=%d), %d frame(s)\n", path.c_str(),
              meta.nx, meta.ny, meta.nz, meta.spatial_dim, meta.nb_frames);
  int fi = 0;
  for (std::size_t i = 0; i < meta.fields.size(); ++i) {
    const mueye::FieldInfo &f = meta.fields[i];
    std::printf("  field %zu: %s (%d comp)\n", i, f.name.c_str(),
                f.nb_components);
    if (field_name && f.name == field_name) fi = static_cast<int>(i);
  }

  mueye::Volume vol;
  std::string err = loader.load(path, meta, meta.fields[fi], 0,
                                mueye::Scalarize::Component, 0, vol);
  if (!err.empty()) {
    std::fprintf(stderr, "load failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("rendering field '%s': %dx%dx%d range [%g, %g]\n",
              meta.fields[fi].name.c_str(), vol.nx, vol.ny, vol.nz, vol.vmin,
              vol.vmax);

  mueye::TransferFunction tf;  // GUI defaults
  int md = vol.nx > vol.ny ? vol.nx : vol.ny;
  if (vol.nz > md) md = vol.nz;
  if (md < 1) md = 1;
  float f = 1.0f / static_cast<float>(md);
  mueye::Vec3 box{vol.nx * f, vol.ny * f, vol.nz * f};

  // Honour any deformation gradient in the file, exactly as the GUI does.
  mueye::Mat3 F{{static_cast<float>(meta.F[0]), static_cast<float>(meta.F[1]),
                 static_cast<float>(meta.F[2]), static_cast<float>(meta.F[3]),
                 static_cast<float>(meta.F[4]), static_cast<float>(meta.F[5]),
                 static_cast<float>(meta.F[6]), static_cast<float>(meta.F[7]),
                 static_cast<float>(meta.F[8])}};
  mueye::Mat3 cell = mueye::mat3_matmul(F, mueye::mat3_diag(box));
  if (meta.has_deformation) std::printf("applying deformation gradient F\n");

  mueye::RenderParams p;
  p.nx = vol.nx;
  p.ny = vol.ny;
  p.nz = vol.nz;
  p.box = box;
  p.inv_cell = mueye::mat3_inverse(cell);
  p.step = 0.5f * f;
  p.data_min = vol.vmin;
  p.data_max = vol.vmax;
  p.lut_size = tf.size();
  p.density_scale = 1.0f;
  p.iso_value = 0.5f * (vol.vmin + vol.vmax);
  p.mode = mueye::RenderMode::DVR;
  p.bg = mueye::Vec3{1.0f, 1.0f, 1.0f};  // GUI default: white

  mueye::CpuRenderer cpu;
  cpu.set_volume(vol.data.data(), vol.nx, vol.ny, vol.nz);
  cpu.set_transfer_function(tf.data(), tf.size());
  mueye::Framebuffer fb;
  fb.resize(512, 512);

  // If the file has a displacement-eligible field (spatial_dim components) other
  // than the one being coloured, render the deformed geometry (exercises the
  // real VolumeLoader::load_displacement path). Auto-scale to a visible warp.
  mueye::DisplacementField df;
  for (std::size_t i = 0; i < meta.fields.size(); ++i) {
    if (static_cast<int>(i) == fi) continue;
    if (meta.fields[i].nb_components != meta.spatial_dim) continue;
    if (!loader.load_displacement(path, meta, meta.fields[i], 0, df).empty() ||
        df.empty())
      continue;
    cpu.set_displacement(df.data.data(), df.nx, df.ny, df.nz);
    p.warp_enabled = 1;
    p.warp_iters = 12;
    p.warp_scale = df.max_mag > 0.0f ? 0.3f / (f * df.max_mag) : 1.0f;
    float margin = p.warp_scale * f * df.max_mag;
    p.warp_lo = mueye::Vec3{-margin, -margin, -margin};
    p.warp_hi = mueye::Vec3{box.x + margin, box.y + margin, box.z + margin};
    std::printf("applying displacement field '%s' (warp_scale=%.2f)\n",
                meta.fields[i].name.c_str(), p.warp_scale);
    break;
  }

  mueye::OrbitCamera cam;
  cam.frame_box(box);
  cpu.render(p, cam.to_camera(1.0f), fb);
  if (!mueye::write_png("view.png", fb)) return 1;
  std::printf("wrote view.png (1x1x1)\n");

  // The replica preview does not warp (periodic tiling is disabled while
  // warping); show the undeformed tiled cell.
  p.warp_enabled = 0;
  p.rep_x = p.rep_y = p.rep_z = 2;
  cam.retarget_box(mueye::Vec3{2 * box.x, 2 * box.y, 2 * box.z});
  cpu.render(p, cam.to_camera(1.0f), fb);
  if (!mueye::write_png("view_rep.png", fb)) return 1;
  std::printf("wrote view_rep.png (2x2x2 replicas)\n");
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc > 2 && std::string(argv[1]) == "--view") {
    return view_file(argv[2], argc > 3 ? argv[3] : nullptr);
  }
  const std::string path = argc > 1 ? argv[1] : "demo.nc";
  const int n = argc > 2 ? std::atoi(argv[2]) : 64;

  // Write demo files with muGrid's C++ API (a 3D volume and a 2D field).
  const std::string path2d = path + ".2d.nc";
  try {
    write_demo(path, n);
    write_demo_2d(path2d, n);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "write_demo failed: %s\n", e.what());
    return 1;
  }

  if (check_reductions() != 0) return 1;
  if (check_file(path, "offscreen.ppm") != 0) return 1;
  if (check_file(path2d, "offscreen_2d.ppm") != 0) return 1;
  if (check_anisotropic() != 0) return 1;

  std::printf(
      "\nOK: muGrid I/O -> scalarize -> ray-trace verified (2D + 3D + "
      "anisotropic); backends agree.\n");
  return 0;
}
