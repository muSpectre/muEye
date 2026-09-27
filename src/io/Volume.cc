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
// scalar.
double reduce(const double *base, int nb_components, Scalarize mode,
              int component, std::ptrdiff_t sc) {
  switch (mode) {
    case Scalarize::Component: {
      int c = component;
      if (c < 0) c = 0;
      if (c >= nb_components) c = nb_components - 1;
      return base[c * sc];
    }
    case Scalarize::Magnitude: {
      double s = 0.0;
      for (int c = 0; c < nb_components; ++c) s += base[c * sc] * base[c * sc];
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
      for (int c = 0; c < nb_components; ++c) s += base[c * sc] * base[c * sc];
      return std::sqrt(s);
    }
  }
  return 0.0;
}

}  // namespace

void Volume::from_field(const double *src, int nx_, int ny_, int nz_,
                        int nb_components, std::ptrdiff_t stride_x,
                        std::ptrdiff_t stride_y, std::ptrdiff_t stride_z,
                        Scalarize mode, int component,
                        std::ptrdiff_t stride_c) {
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
        double v = reduce(base, nb_components, mode, component, stride_c);
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
