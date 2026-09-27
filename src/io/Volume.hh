/**
 * @file   Volume.hh
 *
 * @brief  Dense single-precision scalar volume plus scalarization of muGrid
 *         multi-component fields.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_VOLUME_HH_
#define MUEYE_VOLUME_HH_

#include <cstddef>
#include <string>
#include <vector>

namespace mueye {

/** How to reduce a (possibly multi-component) muGrid field to a single scalar
 *  per voxel. */
enum class Scalarize : int {
  Component = 0,  //!< pick one raw component (flat index; muGrid flattens a
                  //!< d x d tensor column-major, flat = row + d*col)
  Magnitude = 1,  //!< Euclidean norm over all components (vector magnitude)
  VonMises = 2,   //!< von Mises equivalent of a 2x2 (4-comp.) or 3x3 (9-comp.) tensor
  Trace = 3       //!< trace of a 2x2 (4-comp.) or 3x3 (9-comp.) tensor
};

const char *to_string(Scalarize s);

/** Side length d of the square tensor a field with @p nb_components holds
 *  (2 for 4 components, 3 for 9), or 0 if it is not a square tensor. */
int tensor_dim(int nb_components);

/** A dense scalar field on a regular grid, stored column-major
 *  (idx = i + nx*(j + ny*k)) in single precision. */
struct Volume {
  int nx{0}, ny{0}, nz{0};
  float vmin{0.0f}, vmax{1.0f};  //!< range over the *finite* voxels
  std::size_t nb_nonfinite{0};   //!< NaN/Inf voxels found (replaced by vmin)
  std::vector<float> data;

  bool empty() const { return data.empty(); }
  std::size_t size() const {
    return static_cast<std::size_t>(nx) * ny * nz;
  }

  /**
   * Fill this volume from a raw field buffer of doubles or floats (the
   * reductions are computed in double either way).
   *
   * @param src           pointer to the field's data (host memory)
   * @param nx,ny,nz      grid dimensions
   * @param nb_components number of components stored per voxel
   * @param stride_x/y/z  element strides between adjacent voxels along each axis
   *                      (from Field::get_strides(IterUnit::Pixel))
   * @param mode          scalarization mode
   * @param component     component index used by Scalarize::Component
   * @param stride_c      element stride between components of one voxel: 1 for
   *                      muGrid's AoS layout (default), large for the planar
   *                      layout of a raw NetCDF hyperslab
   */
  template <class T>
  void from_field(const T *src, int nx, int ny, int nz, int nb_components,
                  std::ptrdiff_t stride_x, std::ptrdiff_t stride_y,
                  std::ptrdiff_t stride_z, Scalarize mode, int component,
                  std::ptrdiff_t stride_c = 1);
};

extern template void Volume::from_field<double>(const double *, int, int, int,
                                                int, std::ptrdiff_t,
                                                std::ptrdiff_t, std::ptrdiff_t,
                                                Scalarize, int, std::ptrdiff_t);
extern template void Volume::from_field<float>(const float *, int, int, int, int,
                                               std::ptrdiff_t, std::ptrdiff_t,
                                               std::ptrdiff_t, Scalarize, int,
                                               std::ptrdiff_t);

/** A dense vector (displacement) field used for the deformed-geometry warp,
 *  stored as 4 floats per voxel (x, y, z, unused) column-major so it maps
 *  directly onto a GPU float4 / RGBA32F 3-D texture. */
struct DisplacementField {
  int nx{0}, ny{0}, nz{0};
  float max_mag{0.0f};      //!< max |u| over the grid (for the warp AABB margin)
  std::vector<float> data;  //!< 4 floats per voxel

  bool empty() const { return data.empty(); }
  std::size_t size() const {
    return static_cast<std::size_t>(nx) * ny * nz;
  }
};

}  // namespace mueye

#endif  // MUEYE_VOLUME_HH_
