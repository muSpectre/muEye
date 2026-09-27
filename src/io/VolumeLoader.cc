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
#include <cmath>
#include <exception>
#include <functional>

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
 * One frame of a field as a strided view of doubles, as delivered by either
 * reader. Only valid inside the callback that receives it (the storage is
 * released when the reader returns). `src` points at the first addressable
 * component of voxel (0,0,0); voxel (i,j,k) component c is at
 * src[i*sx + j*sy + k*sz + c*sc] for 0 <= c < nb_comp. When a reader was asked
 * for a single component, nb_comp == 1 and that component sits at c == 0.
 */
struct RawField {
  const double *src;    //!< double view, or null when the data is float
  const float *src_f;   //!< float view (single-component direct reads), or null
  int nb_comp;
  std::ptrdiff_t sx, sy, sz, sc;
};
using RawFieldFn = std::function<void(const RawField &)>;

/**
 * Read one frame of @p field straight through the netcdf-c API, bypassing
 * muGrid (ReadPath::Direct). Required for variables not stored as NC_DOUBLE:
 * muGrid's read path (serial nc_get_varm) transfers raw bytes into its Real
 * (double) fields with no type conversion, so an NC_FLOAT variable read
 * through it comes back as reinterpreted garbage; nc_get_vara_double converts
 * on read instead. Also the default for Component mode on any variable, since
 * it fetches a single component where muGrid reads the whole field.
 *
 * The hyperslab is fetched in the file's row-major dimension order
 * (frame, [tensor_dim...], [subpt], nx, ny, nz) — muGrid writes with an imap,
 * so the nx/ny/nz axes in the file are the true x/y/z axes — and handed to
 * @p fn as a RawField with the matching per-axis and per-component strides.
 * Only sub-point 0 is read, like the muGrid path renders. With
 * @p only_component >= 0 just that component is fetched (1/9th of the bytes
 * for a 3x3 tensor field); -1 fetches all of them.
 */
std::string read_direct(const std::string &path, const FileMeta &meta,
                        const FieldInfo &field, int frame, int only_component,
                        const RawFieldFn &fn) {
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

  // One hyperslab covering the selected frame, sub-point 0 and the full grid.
  std::vector<std::size_t> start(ndims, 0), count(ndims, 1);
  std::vector<bool> is_comp(ndims, false);
  for (int d = 0; d < ndims; ++d) {
    int did = dimids[d];
    if (did == frame_dim) {
      start[d] = static_cast<std::size_t>(frame);
      continue;  // count stays 1
    }
    if (did == dx || did == dy || (dz >= 0 && did == dz)) {
      count[d] = dim_len(ncid, did);  // grid axis: full extent
      continue;
    }
    char dname[NC_MAX_NAME + 1] = {0};
    nc_inq_dimname(ncid, did, dname);
    if (starts_with(dname, "subpt")) continue;  // sub-point 0 only
    is_comp[d] = true;                          // tensor_dim__* component axis
    count[d] = dim_len(ncid, did);
  }

  // Single-component fetch: the flat component index nests row-major across
  // the (adjacent) tensor axes, so it decomposes fastest-axis-first from the
  // back.
  if (only_component >= 0) {
    std::ptrdiff_t total_comp = 1;
    for (int d = 0; d < ndims; ++d)
      if (is_comp[d]) total_comp *= static_cast<std::ptrdiff_t>(count[d]);
    std::ptrdiff_t rem = only_component;
    if (rem >= total_comp) rem = total_comp - 1;
    for (int d = ndims - 1; d >= 0; --d) {
      if (!is_comp[d]) continue;
      std::ptrdiff_t len = static_cast<std::ptrdiff_t>(count[d]);
      start[d] = static_cast<std::size_t>(rem % len);
      rem /= len;
      count[d] = 1;
    }
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

  // Typed read: netcdf converts the stored type on the fly. A single component
  // is fetched as float straight away: the Volume is float anyway and the
  // rounding is the same as converting the double later, so the result is
  // bit-identical while the temporary buffer is half the size (0.5 GB instead
  // of 1 GB at 512^3). Multi-component fetches stay double so the reductions
  // (magnitude, von Mises, ...) are evaluated in double precision.
  if (only_component >= 0) {
    std::vector<float> buf(static_cast<std::size_t>(total));
    status =
        nc_get_vara_float(ncid, varid, start.data(), count.data(), buf.data());
    nc_close(ncid);
    if (status != NC_NOERR)
      return std::string("nc_get_vara_float failed: ") + nc_strerror(status);
    fn(RawField{nullptr, buf.data(), nb_comp, sx, sy, sz, sc});
    return "";
  }
  std::vector<double> buf(static_cast<std::size_t>(total));
  status =
      nc_get_vara_double(ncid, varid, start.data(), count.data(), buf.data());
  nc_close(ncid);
  if (status != NC_NOERR)
    return std::string("nc_get_vara_double failed: ") + nc_strerror(status);
  fn(RawField{buf.data(), nullptr, nb_comp, sx, sy, sz, sc});
  return "";
}

/**
 * Read one frame of @p field through muGrid::FileIONetCDF into a temporary
 * GlobalFieldCollection (ReadPath::MuGrid) and hand it to @p fn as a
 * RawField. muGrid always reads every component (AoS: component stride 1);
 * with @p only_component >= 0 the view is offset to that component and
 * reports nb_comp == 1, so callers see the same shape from both readers.
 */
std::string read_mugrid(const std::string &path, const FileMeta &meta,
                        const FieldInfo &field, int frame, int only_component,
                        const RawFieldFn &fn) {
  if (!field.is_double) {
    return "Variable '" + field.name +
           "' is not stored as double; muGrid cannot read it (use the direct "
           "reader).";
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
    if (only_component >= 0) {
      // AoS: the components of one voxel are contiguous, so offsetting the
      // base pointer selects the component (clamped like Volume::from_field).
      int c = std::min(only_component, nb_comp - 1);
      src += c;
      nb_comp = 1;
    }
    fn(RawField{src, nullptr, nb_comp, sx, sy, sz, 1});

    file.close();
    return "";
  } catch (const std::exception &e) {
    return std::string("muGrid read failed: ") + e.what();
  } catch (...) {
    return "muGrid read failed: unknown error.";
  }
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
  int frame_dim = find_dim(ncid, "frame");
  if (frame_dim >= 0) {
    meta.nb_frames = std::max<int>(1, static_cast<int>(dim_len(ncid, frame_dim)));
  } else {
    meta.nb_frames = 1;
  }

  // Optional cell geometry: a macroscopic deformation gradient (or average
  // strain) stored as a global attribute. muGrid does not write these today,
  // but a producer can via FileIONetCDF::write_global_attribute; when present
  // muEye renders the sheared (Bravais) cell. Absent => identity (orthogonal).
  {
    auto read_att = [&](const char *name, std::vector<double> &out) -> bool {
      int aid = -1;
      if (nc_inq_attid(ncid, NC_GLOBAL, name, &aid) != NC_NOERR) return false;
      std::size_t len = 0;
      if (nc_inq_attlen(ncid, NC_GLOBAL, name, &len) != NC_NOERR || len == 0)
        return false;
      out.assign(len, 0.0);
      // nc_get_att_double converts whatever numeric type was stored to double.
      return nc_get_att_double(ncid, NC_GLOBAL, name, out.data()) == NC_NOERR;
    };

    std::vector<double> vals;
    bool is_strain = false;
    if (!read_att("deformation_gradient", vals)) {
      if (read_att("average_strain", vals)) is_strain = true;
    }
    // Embed a 3x3 (9) or 2x2 (4) tensor into the row-major 3x3 F, keeping the
    // z row/column as identity for 2D data.
    double e[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool ok = false;
    if (vals.size() == 9) {
      for (int i = 0; i < 9; ++i) e[i] = vals[i];
      ok = true;
    } else if (vals.size() == 4) {
      e[0] = vals[0]; e[1] = vals[1];  // row 0: xx xy
      e[3] = vals[2]; e[4] = vals[3];  // row 1: yx yy
      ok = true;
    }
    if (ok) {
      for (int i = 0; i < 9; ++i)
        meta.F[i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;  // identity
      if (is_strain) {
        for (int i = 0; i < 9; ++i) meta.F[i] += e[i];  // F = I + eps
      } else if (vals.size() == 9) {
        for (int i = 0; i < 9; ++i) meta.F[i] = e[i];
      } else {  // 2x2 deformation gradient: overwrite upper-left, keep F_zz = 1
        meta.F[0] = e[0]; meta.F[1] = e[1];
        meta.F[3] = e[3]; meta.F[4] = e[4];
      }
      meta.has_deformation = true;
    }
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

  // Physical cell size (muTopOpt's `domain_lengths` global attribute): sets the
  // rendered cell proportions instead of deriving them from the grid shape.
  {
    std::size_t len = 0;
    if (nc_inq_attlen(ncid, NC_GLOBAL, "domain_lengths", &len) == NC_NOERR &&
        (len == 2 || len == 3)) {
      double v[3] = {0.0, 0.0, 0.0};
      if (nc_get_att_double(ncid, NC_GLOBAL, "domain_lengths", v) == NC_NOERR) {
        meta.domain_lengths[0] = v[0];
        meta.domain_lengths[1] = v[1];
        meta.domain_lengths[2] = (len == 3) ? v[2] : 0.0;
        meta.has_domain_lengths = true;
      }
    }
  }

  // Per-frame applied deformation gradient (muTopOpt's frame variable). Dims are
  // (frame, [load_case,] d, d); we read load case 0 for every frame. When
  // present it supersedes the deformation_gradient/average_strain attribute.
  {
    int vid = -1;
    if (nc_inq_varid(ncid, "applied_deformation_gradient", &vid) == NC_NOERR) {
      int nd = 0;
      nc_inq_varndims(ncid, vid, &nd);
      if (nd >= 2) {
        std::vector<int> dids(nd);
        nc_inq_vardimid(ncid, vid, dids.data());
        std::size_t d0 = dim_len(ncid, dids[nd - 2]);  // tensor rows
        std::size_t d1 = dim_len(ncid, dids[nd - 1]);  // tensor cols
        bool has_frame_axis = (frame_dim >= 0 && dids[0] == frame_dim);
        int nframes = has_frame_axis ? meta.nb_frames : 1;
        std::vector<std::size_t> start(nd, 0), count(nd, 1);
        count[nd - 2] = d0;
        count[nd - 1] = d1;
        std::vector<double> buf(d0 * d1, 0.0);
        bool ok = true;
        for (int f = 0; f < nframes && ok; ++f) {
          if (has_frame_axis) start[0] = static_cast<std::size_t>(f);
          if (nc_get_vara_double(ncid, vid, start.data(), count.data(),
                                 buf.data()) != NC_NOERR) {
            ok = false;
            break;
          }
          // Embed the (d0 x d1) block into the upper-left of a row-major 3x3,
          // keeping the z row/column as identity for 2D data.
          std::array<double, 9> Fm{1, 0, 0, 0, 1, 0, 0, 0, 1};
          for (std::size_t r = 0; r < d0 && r < 3; ++r)
            for (std::size_t c = 0; c < d1 && c < 3; ++c)
              Fm[r * 3 + c] = buf[r * d1 + c];
          meta.applied_F.push_back(Fm);
        }
        if (!ok) meta.applied_F.clear();
      }
    }
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
                               Scalarize mode, int component, Volume &out,
                               ReadPath read_path) {
  if (read_path == ReadPath::Auto) {
    // Non-double variables (e.g. muFFTTO's float32 output) cannot go through
    // muGrid's byte-copying read path at all. Component mode goes direct as
    // well: the muGrid path reads every component of the field (all nine of a
    // 3x3 tensor, 1.2 GB at 256^3) to display one, whereas the hyperslab read
    // fetches just the selected component.
    read_path = (!field.is_double || mode == Scalarize::Component)
                    ? ReadPath::Direct
                    : ReadPath::MuGrid;
  }
  // In Component mode both readers deliver only the selected component, at
  // index 0 of the view; the other reductions need every component.
  const bool one = mode == Scalarize::Component;
  const int only = one ? std::max(component, 0) : -1;
  auto fill = [&](const RawField &r) {
    if (r.src_f != nullptr)
      out.from_field(r.src_f, meta.nx, meta.ny, meta.nz, r.nb_comp, r.sx, r.sy,
                     r.sz, mode, one ? 0 : component, r.sc);
    else
      out.from_field(r.src, meta.nx, meta.ny, meta.nz, r.nb_comp, r.sx, r.sy,
                     r.sz, mode, one ? 0 : component, r.sc);
  };
  return read_path == ReadPath::Direct
             ? read_direct(path, meta, field, frame, only, fill)
             : read_mugrid(path, meta, field, frame, only, fill);
}

std::string VolumeLoader::load_displacement(const std::string &path,
                                            const FileMeta &meta,
                                            const FieldInfo &field, int frame,
                                            DisplacementField &out) {
  const int nd = meta.spatial_dim;  // 2 or 3 vector components
  if (field.nb_components < nd)
    return "Selected field has too few components for a displacement.";

  out.nx = meta.nx;
  out.ny = meta.ny;
  out.nz = meta.nz;
  const std::size_t n = out.size();
  out.data.assign(n * 4, 0.0f);  // (x, y, z, unused); z stays 0 in 2D

  // One read of the whole field, then scatter the first nd components into
  // the 4-float layout. (Reading per component would open and read the file
  // nd times and, through muGrid, read every component each time.)
  std::string layout_err;
  auto scatter = [&](const RawField &r) {
    if (r.nb_comp < nd || r.src == nullptr) {
      layout_err = "Displacement field delivered fewer components than "
                   "expected.";
      return;
    }
    const int nx = meta.nx, ny = meta.ny, nz = meta.nz;
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
          const double *base = r.src + i * r.sx + j * r.sy + k * r.sz;
          std::size_t e =
              (static_cast<std::size_t>(i) +
               nx * (j + static_cast<std::size_t>(ny) * k)) * 4;
          for (int c = 0; c < nd; ++c)
            out.data[e + c] = static_cast<float>(base[c * r.sc]);
        }
  };
  std::string err =
      field.is_double ? read_mugrid(path, meta, field, frame, -1, scatter)
                      : read_direct(path, meta, field, frame, -1, scatter);
  if (!err.empty()) return err;
  if (!layout_err.empty()) return layout_err;

  // Largest displacement magnitude, for the warp bounding-box margin.
  float mx = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    float x = out.data[i * 4 + 0], y = out.data[i * 4 + 1],
          z = out.data[i * 4 + 2];
    float m = std::sqrt(x * x + y * y + z * z);
    if (m > mx) mx = m;
  }
  out.max_mag = mx;
  return "";
}

}  // namespace mueye
