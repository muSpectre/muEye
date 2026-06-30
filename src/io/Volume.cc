/**
 * @file   Volume.cc
 *
 * @brief  Implementation of Volume scalarization.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "io/Volume.hh"

#include <cmath>
#include <limits>

namespace mueye {

const char *to_string(Scalarize s) {
  switch (s) {
    case Scalarize::Component:
      return "Component";
    case Scalarize::Magnitude:
      return "Magnitude";
    case Scalarize::VonMises:
      return "von Mises (3x3)";
    case Scalarize::Trace:
      return "Trace (3x3)";
  }
  return "?";
}

namespace {

// Reduce the nb_components values at one voxel (component stride 1) to a scalar.
double reduce(const double *base, int nb_components, Scalarize mode,
              int component) {
  switch (mode) {
    case Scalarize::Component: {
      int c = component;
      if (c < 0) c = 0;
      if (c >= nb_components) c = nb_components - 1;
      return base[c];
    }
    case Scalarize::Magnitude: {
      double s = 0.0;
      for (int c = 0; c < nb_components; ++c) s += base[c] * base[c];
      return std::sqrt(s);
    }
    case Scalarize::Trace: {
      // Interpret the 9 components as a row-major 3x3 tensor: diag = 0,4,8.
      if (nb_components >= 9) return base[0] + base[4] + base[8];
      // Fallback: sum of available components.
      double s = 0.0;
      for (int c = 0; c < nb_components; ++c) s += base[c];
      return s;
    }
    case Scalarize::VonMises: {
      if (nb_components >= 9) {
        // sigma indices (row-major 3x3): 0 1 2 / 3 4 5 / 6 7 8
        double sxx = base[0], syy = base[4], szz = base[8];
        double sxy = base[1], syz = base[5], sxz = base[2];
        double a = sxx - syy, b = syy - szz, c = szz - sxx;
        double j2 = 0.5 * (a * a + b * b + c * c) +
                    3.0 * (sxy * sxy + syz * syz + sxz * sxz);
        return std::sqrt(j2);
      }
      // Fallback to magnitude for non-3x3 fields.
      double s = 0.0;
      for (int c = 0; c < nb_components; ++c) s += base[c] * base[c];
      return std::sqrt(s);
    }
  }
  return 0.0;
}

}  // namespace

void Volume::from_field(const double *src, int nx_, int ny_, int nz_,
                        int nb_components, std::ptrdiff_t stride_x,
                        std::ptrdiff_t stride_y, std::ptrdiff_t stride_z,
                        Scalarize mode, int component) {
  nx = nx_;
  ny = ny_;
  nz = nz_;
  data.assign(size(), 0.0f);

  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const double *base =
            src + i * stride_x + j * stride_y + k * stride_z;
        double v = reduce(base, nb_components, mode, component);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        data[static_cast<std::size_t>(i) + nx * (j + static_cast<std::size_t>(ny) * k)] =
            static_cast<float>(v);
      }
    }
  }

  if (!(lo <= hi)) {  // empty or all-NaN
    lo = 0.0;
    hi = 1.0;
  }
  vmin = static_cast<float>(lo);
  vmax = static_cast<float>(hi);
}

}  // namespace mueye
