/**
 * @file   VolumeLoader.cc
 *
 * @brief  Implementation of NetCDF introspection + muGrid-backed field reading.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "io/VolumeLoader.hh"

#include <netcdf.h>

#include <algorithm>
#include <map>
#include <stdexcept>

// muGrid headers (resolved via the muGrid target's build-interface include dirs)
#include "collection/field_collection.hh"
#include "collection/field_collection_global.hh"
#include "core/enums.hh"
#include "core/types.hh"
#include "field/field.hh"
#include "io/file_io_base.hh"
#include "io/file_io_netcdf.hh"

namespace mueye {

namespace {

// Look up a dimension id by name; returns -1 if absent.
int find_dim(int ncid, const char *name) {
  int dimid = -1;
  if (nc_inq_dimid(ncid, name, &dimid) != NC_NOERR) return -1;
  return dimid;
}

std::size_t dim_len(int ncid, int dimid) {
  std::size_t len = 0;
  if (dimid < 0) return 0;
  if (nc_inq_dimlen(ncid, dimid, &len) != NC_NOERR) return 0;
  return len;
}

bool starts_with(const std::string &s, const char *prefix) {
  return s.rfind(prefix, 0) == 0;
}

/**
 * Read one frame of @p field straight through the netcdf-c API, bypassing
 * muGrid. Used for variables not stored as NC_DOUBLE: muGrid's read path
 * (serial nc_get_varm) transfers raw bytes into its Real (double) fields with
 * no type conversion, so an NC_FLOAT variable read through it comes back as
 * reinterpreted garbage. nc_get_vara_double converts on read instead.
 *
 * The hyperslab is fetched in the file's row-major dimension order
 * (frame, [tensor_dim...], [subpt], nx, ny, nz) — muGrid writes with an imap,
 * so the nx/ny/nz axes in the file are the true x/y/z axes — and handed to
 * Volume::from_field with the matching per-axis and per-component strides.
 * Only sub-point 0 is read, like the muGrid path renders.
 */
std::string load_via_netcdf(const std::string &path, const FileMeta &meta,
                            const FieldInfo &field, int frame, Scalarize mode,
                            int component, Volume &out) {
  int ncid = -1;
  int status = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (status != NC_NOERR)
    return std::string("nc_open failed: ") + nc_strerror(status);

  int varid = -1;
  status = nc_inq_varid(ncid, field.name.c_str(), &varid);
  if (status != NC_NOERR) {
    nc_close(ncid);
    return "Variable '" + field.name + "' vanished from the file.";
  }
  int ndims = 0;
  int dimids[NC_MAX_VAR_DIMS];
  if (nc_inq_var(ncid, varid, nullptr, nullptr, &ndims, dimids, nullptr) !=
      NC_NOERR) {
    nc_close(ncid);
    return "Failed to inquire variable '" + field.name + "'.";
  }

  int dx = find_dim(ncid, "nx");
  int dy = find_dim(ncid, "ny");
  int dz = find_dim(ncid, "nz");
  int frame_dim = find_dim(ncid, "frame");

  // One hyperslab covering the selected frame, sub-point 0, all components
  // and the full grid.
  std::vector<std::size_t> start(ndims, 0), count(ndims, 1);
  for (int d = 0; d < ndims; ++d) {
    int did = dimids[d];
    if (did == frame_dim) {
      start[d] = static_cast<std::size_t>(frame);
      continue;  // count stays 1
    }
    char dname[NC_MAX_NAME + 1] = {0};
    nc_inq_dimname(ncid, did, dname);
    if (starts_with(dname, "subpt")) continue;  // sub-point 0 only
    count[d] = dim_len(ncid, did);              // grid or component axis
  }

  // Row-major element strides within the fetched buffer.
  std::vector<std::ptrdiff_t> stride(ndims, 0);
  std::ptrdiff_t total = 1;
  for (int d = ndims - 1; d >= 0; --d) {
    stride[d] = total;
    total *= static_cast<std::ptrdiff_t>(count[d]);
  }

  std::ptrdiff_t sx = 0, sy = 0, sz = 0, sc = 1;
  int nb_comp = 1;
  for (int d = 0; d < ndims; ++d) {
    int did = dimids[d];
    if (did == dx) {
      sx = stride[d];
    } else if (did == dy) {
      sy = stride[d];
    } else if (dz >= 0 && did == dz) {
      sz = stride[d];
    } else if (did != frame_dim && count[d] > 1) {
      // Component (tensor_dim__*) axis. Adjacent axes nest row-major, so the
      // flat component index advances by the stride of the fastest one.
      nb_comp *= static_cast<int>(count[d]);
      sc = stride[d];
    }
  }
  std::ptrdiff_t expect = static_cast<std::ptrdiff_t>(meta.nx) * meta.ny *
                          meta.nz * nb_comp;
  if (sx == 0 || sy == 0 || total != expect) {
    nc_close(ncid);
    return "Unexpected layout of variable '" + field.name + "'.";
  }

  // Typed read: netcdf converts the stored type (float, ...) to double.
  std::vector<double> buf(static_cast<std::size_t>(total));
  status =
      nc_get_vara_double(ncid, varid, start.data(), count.data(), buf.data());
  nc_close(ncid);
  if (status != NC_NOERR)
    return std::string("nc_get_vara_double failed: ") + nc_strerror(status);

  out.from_field(buf.data(), meta.nx, meta.ny, meta.nz, nb_comp, sx, sy, sz,
                 mode, component, sc);
  return "";
}

}  // namespace

FileMeta VolumeLoader::open(const std::string &path) {
  FileMeta meta;
  int ncid = -1;
  int status = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (status != NC_NOERR) {
    meta.error = std::string("nc_open failed: ") + nc_strerror(status);
    return meta;
  }

  // Grid dimensions. muGrid names them nx, ny, nz (empty suffix).
  int dx = find_dim(ncid, "nx");
  int dy = find_dim(ncid, "ny");
  int dz = find_dim(ncid, "nz");
  if (dx < 0 || dy < 0) {
    meta.error = "File has no 'nx'/'ny' dimensions; not a muGrid grid file.";
    nc_close(ncid);
    return meta;
  }
  meta.nx = static_cast<int>(dim_len(ncid, dx));
  meta.ny = static_cast<int>(dim_len(ncid, dy));
  meta.nz = dz >= 0 ? static_cast<int>(dim_len(ncid, dz)) : 1;
  // A file without an 'nz' dimension is a genuine 2D grid; we must read it back
  // through a 2D muGrid collection (a 3D collection would demand an nz axis the
  // file lacks). It still renders as a single-slice (nz==1) volume on a plane.
  meta.spatial_dim = dz >= 0 ? 3 : 2;

  // Frame (unlimited) dimension.
  int unlim = -1;
  nc_inq_unlimdim(ncid, &unlim);
  int frame_dim = find_dim(ncid, "frame");
  if (frame_dim >= 0) {
    meta.nb_frames = std::max<int>(1, static_cast<int>(dim_len(ncid, frame_dim)));
  } else {
    meta.nb_frames = 1;
  }

  // Enumerate variables and keep those defined on the spatial grid.
  int nvars = 0;
  nc_inq_nvars(ncid, &nvars);
  for (int v = 0; v < nvars; ++v) {
    char vname[NC_MAX_NAME + 1] = {0};
    nc_type vtype;
    int vndims = 0;
    int vdimids[NC_MAX_VAR_DIMS];
    if (nc_inq_var(ncid, v, vname, &vtype, &vndims, vdimids, nullptr) !=
        NC_NOERR)
      continue;

    // Must be a real (double/float) field defined on nx, ny (and nz if 3D).
    bool has_x = false, has_y = false, has_z = (dz < 0);
    for (int d = 0; d < vndims; ++d) {
      if (vdimids[d] == dx) has_x = true;
      if (vdimids[d] == dy) has_y = true;
      if (dz >= 0 && vdimids[d] == dz) has_z = true;
    }
    if (!(has_x && has_y && has_z)) continue;
    if (vtype != NC_DOUBLE && vtype != NC_FLOAT) continue;

    FieldInfo fi;
    fi.name = vname;
    fi.nb_components = 1;
    fi.nb_sub_pts = 1;
    fi.is_double = (vtype == NC_DOUBLE);

    // Classify the remaining dimensions: tensor_dim__* -> components,
    // subpt__* -> sub-points, frame/grid -> ignored.
    for (int d = 0; d < vndims; ++d) {
      int did = vdimids[d];
      if (did == dx || did == dy || (dz >= 0 && did == dz) || did == frame_dim)
        continue;
      char dname[NC_MAX_NAME + 1] = {0};
      nc_inq_dimname(ncid, did, dname);
      std::size_t len = dim_len(ncid, did);
      std::string dn(dname);
      if (starts_with(dn, "subpt")) {
        fi.nb_sub_pts = static_cast<int>(len);
        // Derive a tag from subpt__<tag>-<n>: strip prefix and trailing -<n>.
        std::string rest = dn;
        auto pos = rest.find("__");
        if (pos != std::string::npos) rest = rest.substr(pos + 2);
        auto dash = rest.rfind('-');
        if (dash != std::string::npos) rest = rest.substr(0, dash);
        fi.sub_tag = rest.empty() ? "quad" : rest;
      } else {
        // tensor_dim__* (or any other extra axis) multiplies the components.
        if (starts_with(dn, "tensor_dim")) fi.has_tensor_dim = true;
        fi.nb_components *= static_cast<int>(len);
      }
    }
    meta.fields.push_back(std::move(fi));
  }

  nc_close(ncid);

  if (meta.fields.empty()) {
    meta.error = "No renderable grid fields found in file.";
    return meta;
  }
  meta.valid = true;
  return meta;
}

std::string VolumeLoader::load(const std::string &path, const FileMeta &meta,
                               const FieldInfo &field, int frame,
                               Scalarize mode, int component, Volume &out) {
  // Non-double variables (e.g. muFFTTO's float32 output) cannot go through
  // muGrid's byte-copying read path; fetch them directly via netcdf-c.
  if (!field.is_double) {
    return load_via_netcdf(path, meta, field, frame, mode, component, out);
  }
  try {
    // Build a field collection matching the file's spatial dimension. A genuine
    // 2D file (no nz axis) must be read through a 2D collection; a 3D file (or
    // one we treat as 3D) carries all three axes.
    std::vector<muGrid::Index_t> dims =
        meta.spatial_dim == 2 ? std::vector<muGrid::Index_t>{meta.nx, meta.ny}
                              : std::vector<muGrid::Index_t>{meta.nx, meta.ny,
                                                             meta.nz};

    muGrid::DynGridIndex domain(dims);
    muGrid::DynGridIndex locations(static_cast<muGrid::Dim_t>(dims.size()),
                                   muGrid::Index_t{0});

    muGrid::GlobalFieldCollection::SubPtMap_t sub_pts;
    std::string tag = field.sub_tag;
    if (field.nb_sub_pts > 1 && !tag.empty()) {
      sub_pts[tag] = field.nb_sub_pts;
    }

    muGrid::GlobalFieldCollection fc(domain, domain, locations, sub_pts);

    const std::string sub_division =
        (field.nb_sub_pts > 1 && !tag.empty()) ? tag : muGrid::PixelTag;
    // muGrid derives the expected NetCDF dimensions from the field's component
    // shape. A field stored with a tensor_dim__ axis must be registered with a
    // matching component count; a true scalar (no tensor_dim in the file, e.g.
    // muFFTTO's 'density') must be registered with an EMPTY component shape, or
    // muGrid demands a nonexistent tensor_dim__<name>-0 axis and the read fails.
    if (field.has_tensor_dim) {
      fc.register_real_field(field.name, field.nb_components, sub_division);
    } else {
      fc.register_real_field(field.name, muGrid::Shape_t{}, sub_division);
    }

    muGrid::FileIONetCDF file(path, muGrid::FileIOBase::OpenMode::Read);
    file.register_field_collection(fc);
    file.read(frame, {field.name});

    muGrid::Field &f = fc.get_field(field.name);
    const double *src =
        static_cast<const double *>(f.get_void_data_ptr());
    if (src == nullptr) {
      file.close();
      return "Field data pointer is null (is the field on device memory?).";
    }

    // The last `spatial_dim` entries of the pixel strides are the per-voxel
    // strides for the x, y (, z) axes (already scaled by nb_dof_per_pixel in
    // muGrid's AoS layout). A 2D field has no z stride; nz==1 so z never varies.
    muGrid::Shape_t strides = f.get_strides(muGrid::IterUnit::Pixel);
    if (strides.size() < static_cast<std::size_t>(meta.spatial_dim)) {
      file.close();
      return "Unexpected field stride layout (fewer strides than spatial dims).";
    }
    std::ptrdiff_t sx, sy, sz;
    if (meta.spatial_dim == 2) {
      sx = strides[strides.size() - 2];
      sy = strides[strides.size() - 1];
      sz = 0;
    } else {
      sx = strides[strides.size() - 3];
      sy = strides[strides.size() - 2];
      sz = strides[strides.size() - 1];
    }

    int nb_comp = static_cast<int>(f.get_nb_components());
    out.from_field(src, meta.nx, meta.ny, meta.nz, nb_comp, sx, sy, sz, mode,
                   component);

    file.close();
    return "";
  } catch (const std::exception &e) {
    return std::string("muGrid read failed: ") + e.what();
  } catch (...) {
    return "muGrid read failed: unknown error.";
  }
}

}  // namespace mueye
