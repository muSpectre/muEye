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
};

/** Stateless loader: every open()/load() opens the file, reads and closes it
 *  again, so the viewer never holds a simulation's output file open. The
 *  per-load open/registration overhead is sub-millisecond — deliberately not
 *  cached (simplicity over micro-optimization). */
class VolumeLoader {
 public:
  /** Introspect @p path. On failure returns a FileMeta with valid==false and a
   *  populated error string. */
  FileMeta open(const std::string &path);

  /** Read (field, frame) and scalarize into @p out.
   *  @returns empty string on success, otherwise an error message. */
  std::string load(const std::string &path, const FileMeta &meta,
                   const FieldInfo &field, int frame, Scalarize mode,
                   int component, Volume &out);
};

}  // namespace mueye

#endif  // MUEYE_VOLUME_LOADER_HH_
