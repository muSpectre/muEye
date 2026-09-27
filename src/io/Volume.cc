/**
 * @file   Volume.cc
 *
 * @brief  Implementation of Volume scalarization.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "io/Volume.hh"

#include <algorithm>
#include <cmath>
#include <limits>

#include "render/parallel_for.hh"

namespace mueye {

const char *to_string(Scalarize s) {
  switch (s) {
    case Scalarize::Component:
      return "Component";
    case Scalarize::Magnitude:
      return "Magnitude";
    case Scalarize::VonMises:
      return "von Mises (tensor)";
    case Scalarize::Trace:
      return "Trace (tensor)";
  }
  return "?";
}

int tensor_dim(int nb_components) {
  if (nb_components == 4) return 2;
  if (nb_components == 9) return 3;
  return 0;
}

namespace {

// Reduce the nb_components values at one voxel (component stride sc) to a
// scalar. T is the storage type of the source buffer; arithmetic is double.
template <class T>
double reduce(const T *base, int nb_components, Scalarize mode, int component,
              std::ptrdiff_t sc) {
  switch (mode) {
    case Scalarize::Component: {
      int c = component;
      if (c < 0) c = 0;
      if (c >= nb_components) c = nb_components - 1;
      return base[c * sc];
    }
    case Scalarize::Magnitude: {
      double s = 0.0;
      for (int c = 0; c < nb_components; ++c) {
        double x = base[c * sc];  // promote before squaring (T may be float)
        s += x * x;
      }
      return std::sqrt(s);
    }
    case Scalarize::Trace: {
      // A d x d tensor (4 components in 2D, 9 in 3D) has its diagonal at flat
      // index r*(d+1) whichever way (row- or column-major) it is flattened.
      int d = tensor_dim(nb_components);
      if (d > 0) {
        double s = 0.0;
        for (int r = 0; r < d; ++r) s += base[r * (d + 1) * sc];
        return s;
      }
      // Not a square tensor: sum of available components.
      double s = 0.0;
      for (int c = 0; c < nb_components; ++c) s += base[c * sc];
      return s;
    }
    case Scalarize::VonMises: {
      int d = tensor_dim(nb_components);
      if (d == 3) {
        // Off-diagonals are symmetrized, 0.5*(m[r,c] + m[c,r]): for a stress
        // tensor this is a no-op and it makes the result independent of
        // whether the file flattens components row- or column-major (muGrid
        // stores them column-major).
        double sxx = base[0], syy = base[4 * sc], szz = base[8 * sc];
        double sxy = 0.5 * (base[1 * sc] + base[3 * sc]);
        double sxz = 0.5 * (base[2 * sc] + base[6 * sc]);
        double syz = 0.5 * (base[5 * sc] + base[7 * sc]);
        double a = sxx - syy, b = syy - szz, c = szz - sxx;
        double j2 = 0.5 * (a * a + b * b + c * c) +
                    3.0 * (sxy * sxy + syz * syz + sxz * sxz);
        return std::sqrt(j2);
      }
      if (d == 2) {
        // 2x2 in-plane tensor; the out-of-plane component is taken as zero
        // (plane-stress convention): sqrt(sxx^2 - sxx*syy + syy^2 + 3 sxy^2).
        double sxx = base[0], syy = base[3 * sc];
        double sxy = 0.5 * (base[1 * sc] + base[2 * sc]);
        double j2 = sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy;
        return std::sqrt(j2);
      }
      // Not a square tensor: fall back to the magnitude.
      double s = 0.0;
      for (int c = 0; c < nb_components; ++c) {
        double x = base[c * sc];
        s += x * x;
      }
      return std::sqrt(s);
    }
  }
  return 0.0;
}

}  // namespace

template <class T>
void Volume::from_field(const T *src, int nx_, int ny_, int nz_,
                        int nb_components, std::ptrdiff_t stride_x,
                        std::ptrdiff_t stride_y, std::ptrdiff_t stride_z,
                        Scalarize mode, int component,
                        std::ptrdiff_t stride_c) {
  nx = nx_;
  ny = ny_;
  nz = nz_;
  data.assign(size(), 0.0f);

  // Scalarize in parallel over (j,k) rows, each row reducing into its own
  // min/max/NaN-count slot; the slots are combined afterwards. Rows rather
  // than k-slices so a 2D field (nz == 1) parallelizes too. This is what sits
  // between a frame-slider tick and the picture, and at 512^3 the serial loop
  // took long enough to push scrubbing over the live-reload threshold.
  const int nb_rows = ny * nz;
  std::vector<double> row_lo(static_cast<std::size_t>(std::max(nb_rows, 1)),
                             std::numeric_limits<double>::infinity());
  std::vector<double> row_hi(row_lo.size(),
                             -std::numeric_limits<double>::infinity());
  std::vector<std::size_t> row_bad(row_lo.size(), 0);

  parallel_for(nb_rows, 0, [&](int row) {
    const int j = row % ny;
    const int k = row / ny;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    std::size_t bad = 0;
    float *dst = data.data() + nx * (j + static_cast<std::size_t>(ny) * k);
    const T *row_base = src + j * stride_y + k * stride_z;
    for (int i = 0; i < nx; ++i) {
      double v = reduce(row_base + i * stride_x, nb_components, mode, component,
                        stride_c);
      // NaN/Inf never enter the range: they would poison the LUT lookup
      // (an (int) cast of NaN is undefined) and the final 8-bit conversion.
      if (std::isfinite(v)) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      } else {
        ++bad;
      }
      dst[i] = static_cast<float>(v);
    }
    row_lo[row] = lo;
    row_hi[row] = hi;
    row_bad[row] = bad;
  });

  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  std::size_t nb_bad = 0;
  for (int row = 0; row < nb_rows; ++row) {
    lo = std::min(lo, row_lo[row]);
    hi = std::max(hi, row_hi[row]);
    nb_bad += row_bad[row];
  }

  if (!(lo <= hi)) {  // empty or no finite value at all
    lo = 0.0;
    hi = 1.0;
  }
  vmin = static_cast<float>(lo);
  vmax = static_cast<float>(hi);
  nb_nonfinite = nb_bad;

  // Replace non-finite voxels by the minimum, which the transfer function
  // maps to LUT entry 0 (transparent for the default opacity ramp) and which
  // an isosurface can never cross, so they neither render as garbage nor
  // fake a surface. The count is kept so the UI can say they were there.
  if (nb_bad > 0) {
    for (float &f : data)
      if (!std::isfinite(f)) f = vmin;
  }
}

template void Volume::from_field<double>(const double *, int, int, int, int,
                                         std::ptrdiff_t, std::ptrdiff_t,
                                         std::ptrdiff_t, Scalarize, int,
                                         std::ptrdiff_t);
template void Volume::from_field<float>(const float *, int, int, int, int,
                                        std::ptrdiff_t, std::ptrdiff_t,
                                        std::ptrdiff_t, Scalarize, int,
                                        std::ptrdiff_t);

}  // namespace mueye
