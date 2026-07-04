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

  muGrid::FileIONetCDF file(path, muGrid::FileIOBase::OpenMode::Overwrite);
  file.register_field_collection(fc);

  fill_cube(d, n, 0.5, 0.5, 0.5, 0.25);
  file.append_frame().write();
  fill_cube(d, n, 0.62, 0.40, 0.55, 0.18);
  file.append_frame().write();
  file.close();
  std::printf("wrote %s (%d^3, field 'phi' cube, 2 frames)\n", path.c_str(), n);
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

std::size_t count_nonbg(const mueye::Framebuffer &fb) {
  std::size_t n = 0;
  std::uint8_t bgr = static_cast<std::uint8_t>(0.05f * 255 + 0.5f);
  for (int i = 0; i < fb.width * fb.height; ++i)
    if (std::abs(int(fb.rgba[i * 4]) - int(bgr)) > 6 ||
        fb.rgba[i * 4 + 1] > 16 || fb.rgba[i * 4 + 2] > 24)
      ++n;
  return n;
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

  mueye::RenderParams p;
  p.nx = vol.nx;
  p.ny = vol.ny;
  p.nz = vol.nz;
  p.box = box;
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

  mueye::OrbitCamera cam;
  cam.frame_box(box);
  cpu.render(p, cam.to_camera(1.0f), fb);
  if (!mueye::write_png("view.png", fb)) return 1;
  std::printf("wrote view.png (1x1x1)\n");

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

  if (check_file(path, "offscreen.ppm") != 0) return 1;
  if (check_file(path2d, "offscreen_2d.ppm") != 0) return 1;

  std::printf(
      "\nOK: muGrid I/O -> scalarize -> ray-trace verified (2D + 3D); "
      "backends agree.\n");
  return 0;
}
