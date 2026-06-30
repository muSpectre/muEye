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
  try {
    // Build a matching field collection. We always carry three spatial
    // dimensions; a 2D file simply has nz == 1.
    std::vector<muGrid::Index_t> dims{meta.nx, meta.ny, meta.nz};

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
    fc.register_real_field(field.name, field.nb_components, sub_division);

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

    // The last 3 entries of the pixel strides are the per-voxel strides for the
    // x, y, z axes (already scaled by nb_dof_per_pixel in muGrid's AoS layout).
    muGrid::Shape_t strides = f.get_strides(muGrid::IterUnit::Pixel);
    if (strides.size() < 3) {
      file.close();
      return "Unexpected field stride layout (need 3 spatial dimensions).";
    }
    std::ptrdiff_t sx = strides[strides.size() - 3];
    std::ptrdiff_t sy = strides[strides.size() - 2];
    std::ptrdiff_t sz = strides[strides.size() - 1];

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
