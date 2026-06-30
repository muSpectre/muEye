/**
 * @file   offscreen_check.cc
 *
 * @brief  Headless end-to-end verification of the muEye data + render pipeline.
 *
 * Writes a demo muGrid NetCDF file (a Gaussian blob, two frames) using muGrid's
 * C++ API, reads it back through muEye's VolumeLoader, scalarizes it, ray-traces
 * a frame with the CPU renderer, and writes the result to a PPM image. Prints
 * statistics so the pipeline can be validated without a display or Python.
 *
 * This exercises exactly the code paths the GUI uses, minus ImGui/OpenGL.
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
#include "io/Volume.hh"
#include "io/VolumeLoader.hh"
#include "io/file_io_base.hh"
#include "io/file_io_netcdf.hh"
#include "render/CpuRenderer.hh"
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

}  // namespace

int main(int argc, char **argv) {
  const std::string path = argc > 1 ? argv[1] : "demo.nc";
  const int n = argc > 2 ? std::atoi(argv[2]) : 64;

  // 1. Write a demo file with muGrid's C++ API.
  try {
    write_demo(path, n);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "write_demo failed: %s\n", e.what());
    return 1;
  }

  // 2. Introspect it back through muEye's loader.
  mueye::VolumeLoader loader;
  mueye::FileMeta meta = loader.open(path);
  if (!meta.valid) {
    std::fprintf(stderr, "introspection failed: %s\n", meta.error.c_str());
    return 1;
  }
  std::printf("introspected: %d x %d x %d, %d frame(s), %zu field(s)\n", meta.nx,
              meta.ny, meta.nz, meta.nb_frames, meta.fields.size());
  for (auto &fi : meta.fields)
    std::printf("  field '%s'  components=%d  sub_pts=%d\n", fi.name.c_str(),
                fi.nb_components, fi.nb_sub_pts);

  if (meta.fields.empty()) return 1;

  // 3. Load frame 0 and scalarize.
  mueye::Volume vol;
  std::string err = loader.load(path, meta, meta.fields[0], 0,
                                mueye::Scalarize::Component, 0, vol);
  if (!err.empty()) {
    std::fprintf(stderr, "load failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("volume loaded: %dx%dx%d range [%.5f, %.5f]\n", vol.nx, vol.ny,
              vol.nz, vol.vmin, vol.vmax);

  // Expect a solid cube: min ~0, max ~1.
  if (!(vol.vmax > 0.5f && vol.vmin < 0.5f)) {
    std::fprintf(stderr, "unexpected value range; cube not recovered?\n");
    return 1;
  }

  // 4. Ray-trace a frame with the CPU renderer (DVR).
  mueye::TransferFunction tf;
  tf.set_colormap(mueye::Colormap::Viridis);
  tf.set_opacity_scale(1.0f);

  mueye::OrbitCamera cam;
  mueye::Framebuffer fb;
  fb.resize(256, 256);

  mueye::RenderRequest req;
  req.volume = vol.data.data();
  req.lut = tf.data();
  mueye::RenderParams p;
  p.nx = vol.nx;
  p.ny = vol.ny;
  p.nz = vol.nz;
  int md = vol.nx > vol.ny ? vol.nx : vol.ny;
  if (vol.nz > md) md = vol.nz;
  p.step = 0.5f / md;
  p.data_min = vol.vmin;
  p.data_max = vol.vmax;
  p.lut_size = tf.size();
  p.density_scale = 1.0f;
  p.iso_value = 0.5f * (vol.vmin + vol.vmax);
  p.mode = mueye::RenderMode::DVR;
  p.bg = mueye::Vec3{0.05f, 0.06f, 0.08f};
  req.params = p;
  req.camera = cam.to_camera(1.0f);

  mueye::CpuRenderer renderer;
  renderer.render(req, fb);

  // 5. Count non-background pixels (the blob should be visible).
  std::size_t nonbg = 0;
  std::uint8_t bgr = static_cast<std::uint8_t>(0.05f * 255 + 0.5f);
  for (int i = 0; i < fb.width * fb.height; ++i) {
    std::uint8_t r = fb.rgba[i * 4 + 0];
    std::uint8_t g = fb.rgba[i * 4 + 1];
    std::uint8_t b = fb.rgba[i * 4 + 2];
    if (std::abs(int(r) - int(bgr)) > 6 || g > 16 || b > 24) ++nonbg;
  }
  double frac = double(nonbg) / (fb.width * fb.height);
  std::printf("rendered 256x256 DVR: %zu non-background px (%.1f%%)\n", nonbg,
              100.0 * frac);

  if (write_ppm("offscreen.ppm", fb))
    std::printf("wrote offscreen.ppm\n");

  if (frac < 0.02) {
    std::fprintf(stderr, "render produced almost nothing; check pipeline.\n");
    return 1;
  }

  // 6. Also exercise the isosurface path.
  req.params.mode = mueye::RenderMode::Isosurface;
  renderer.render(req, fb);
  std::size_t iso_hits = 0;
  for (int i = 0; i < fb.width * fb.height; ++i)
    if (fb.rgba[i * 4 + 0] > 60) ++iso_hits;
  std::printf("rendered isosurface: %zu lit px\n", iso_hits);

  std::printf("OK: muGrid I/O -> scalarize -> ray-trace pipeline verified.\n");
  return 0;
}
