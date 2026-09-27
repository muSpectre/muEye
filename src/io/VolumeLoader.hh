/**
 * @file   VolumeLoader.hh
 *
 * @brief  Introspect a muGrid NetCDF file and load frames/fields into a Volume.
 *
 * The loader uses the netcdf-c API directly to discover the grid dimensions,
 * frame count and field variables of an arbitrary file (muGrid's read API needs
 * a pre-shaped FieldCollection, so we must know the layout first). The actual
 * data read is then delegated to muGrid::FileIONetCDF.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_VOLUME_LOADER_HH_
#define MUEYE_VOLUME_LOADER_HH_

#include <array>
#include <string>
#include <vector>

#include "io/Volume.hh"

namespace mueye {

/** Metadata describing one renderable field variable in the file. */
struct FieldInfo {
  std::string name;
  int nb_components{1};      //!< tensor components per sub-point
  std::string sub_tag;       //!< sub-division tag, empty => pixel subdivision
  int nb_sub_pts{1};         //!< sub-points per pixel
  bool has_tensor_dim{false};  //!< file has a tensor_dim__ axis for this field;
                               //!< if false the field is a true scalar and must
                               //!< be read back with an empty component shape
  bool is_double{true};  //!< variable is stored as NC_DOUBLE. muGrid's read
                         //!< path transfers raw bytes into its Real (double)
                         //!< fields without type conversion, so only double
                         //!< variables can go through it; float variables
                         //!< (e.g. muFFTTO output) are read directly via
                         //!< netcdf-c instead.
};

/** Metadata describing the whole file. */
struct FileMeta {
  int nx{0}, ny{0}, nz{1};
  int spatial_dim{3};  //!< 2 for a genuine 2D grid (no nz), 3 otherwise
  int nb_frames{1};
  std::vector<FieldInfo> fields;
  bool valid{false};
  std::string error;  //!< populated when valid == false

  //! Macroscopic deformation gradient F (row-major 3x3; identity == undeformed)
  //! that shears the reference cell into a (possibly non-orthogonal) Bravais
  //! cell. Read from the file's `deformation_gradient` global attribute, or
  //! from `average_strain` (eps) as F = I + eps. muGrid never writes these
  //! today; a producer adds them via FileIONetCDF::write_global_attribute.
  //! A 2D file's 2x2 tensor is embedded in the upper-left, with F_zz = 1.
  double F[9]{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  bool has_deformation{false};  //!< a deformation attribute was found

  //! Physical edge lengths of the unit cell per axis, from the `domain_lengths`
  //! global attribute (muTopOpt writes it). Sets the rendered cell proportions
  //! (the reference box aspect) instead of deriving them from the grid shape.
  double domain_lengths[3]{0.0, 0.0, 0.0};
  bool has_domain_lengths{false};

  //! Per-frame applied deformation gradient (row-major 3x3), from the
  //! `applied_deformation_gradient` frame variable (muTopOpt writes it). If the
  //! variable carries a leading load-case axis, load case 0 is used. Empty when
  //! the variable is absent; when present it supersedes `F`/has_deformation and
  //! is applied per frame.
  std::vector<std::array<double, 9>> applied_F;
};

/** Which reader fetches the voxel data.
 *
 *  - MuGrid: muGrid::FileIONetCDF into a GlobalFieldCollection. Reads *every*
 *    component of the field (AoS), so for a 3x3 double tensor it moves and
 *    holds nine times the data a single component needs. Only works for
 *    NC_DOUBLE variables (muGrid copies bytes without type conversion).
 *  - Direct: netcdf-c hyperslab read with on-the-fly type conversion. Fetches
 *    only the selected component in Component mode.
 *  - Auto: Direct for non-double variables and for Component mode (where it
 *    is both required or much cheaper), MuGrid otherwise. muEye_check verifies
 *    that both readers produce identical volumes.
 */
enum class ReadPath : int { Auto = 0, MuGrid = 1, Direct = 2 };

/** Stateless loader: every open()/load() opens the file, reads and closes it
 *  again, so the viewer never holds a simulation's output file open. The
 *  per-load open/registration overhead is sub-millisecond — deliberately not
 *  cached (simplicity over micro-optimization). */
class VolumeLoader {
 public:
  /** Introspect @p path. On failure returns a FileMeta with valid==false and a
   *  populated error string. */
  FileMeta open(const std::string &path);

  /** Read (field, frame) and scalarize into @p out. @p read_path selects the
   *  reader (see ReadPath); ReadPath::MuGrid on a non-double variable is an
   *  error.
   *  @returns empty string on success, otherwise an error message. */
  std::string load(const std::string &path, const FileMeta &meta,
                   const FieldInfo &field, int frame, Scalarize mode,
                   int component, Volume &out,
                   ReadPath read_path = ReadPath::Auto);

  /** Read (field, frame) as a displacement vector field: the first
   *  meta.spatial_dim components are packed into @p out (4 floats/voxel, z=0 in
   *  2D). @p field must have at least spatial_dim components. One read of the
   *  whole field through the same readers load() uses (muGrid for double
   *  variables, netcdf-c otherwise), so the double/float and 2D/3D cases are
   *  handled identically.
   *  @returns empty string on success, otherwise an error message. */
  std::string load_displacement(const std::string &path, const FileMeta &meta,
                                const FieldInfo &field, int frame,
                                DisplacementField &out);
};

}  // namespace mueye

#endif  // MUEYE_VOLUME_LOADER_HH_
