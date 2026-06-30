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
  Component = 0,  //!< pick one raw component
  Magnitude = 1,  //!< Euclidean norm over all components (vector magnitude)
  VonMises = 2,   //!< von Mises equivalent of a 3x3 (9-component) tensor
  Trace = 3       //!< trace of a 3x3 (9-component) tensor
};

const char *to_string(Scalarize s);

/** A dense scalar field on a regular grid, stored column-major
 *  (idx = i + nx*(j + ny*k)) in single precision. */
struct Volume {
  int nx{0}, ny{0}, nz{0};
  float vmin{0.0f}, vmax{1.0f};
  std::vector<float> data;

  bool empty() const { return data.empty(); }
  std::size_t size() const {
    return static_cast<std::size_t>(nx) * ny * nz;
  }

  /**
   * Fill this volume from a raw muGrid field buffer.
   *
   * @param src           pointer to the field's double data (host memory)
   * @param nx,ny,nz      grid dimensions
   * @param nb_components number of components stored per voxel
   * @param stride_x/y/z  element strides between adjacent voxels along each axis
   *                      (from Field::get_strides(IterUnit::Pixel)); the
   *                      component stride is assumed to be 1 (muGrid AoS layout)
   * @param mode          scalarization mode
   * @param component     component index used by Scalarize::Component
   */
  void from_field(const double *src, int nx, int ny, int nz, int nb_components,
                  std::ptrdiff_t stride_x, std::ptrdiff_t stride_y,
                  std::ptrdiff_t stride_z, Scalarize mode, int component);
};

}  // namespace mueye

#endif  // MUEYE_VOLUME_HH_
