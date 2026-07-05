/**
 * @file   render_core.hh
 *
 * @brief  Shared host/device ray-marching core for muEye.
 *
 * This header is deliberately self-contained: it pulls in no muGrid, OpenGL or
 * ImGui headers so that it can be compiled both by a plain C++ compiler (the CPU
 * renderer) and, in a later pass, by nvcc/hipcc (the GPU renderer). All entry
 * points are decorated with MUEYE_HD so they become `__host__ __device__` when
 * compiled as CUDA/HIP and plain functions otherwise.
 *
 * Everything here works in single precision (float); muGrid stores Real=double
 * but the viewer downcasts once when a volume is loaded.
 *
 * This file is part of muEye, a viewer for muGrid data. muEye is intended to be
 * extracted into a standalone repository; see README.md.
 */

#ifndef MUEYE_RENDER_CORE_HH_
#define MUEYE_RENDER_CORE_HH_

#if defined(__CUDACC__) || defined(__HIPCC__)
#define MUEYE_HD __host__ __device__
#else
#define MUEYE_HD
#include <cmath>
#endif

// Restrict qualifier: promises the read-only volume/LUT pointers do not alias
// the output, letting the compiler keep loads in registers and (on NVIDIA)
// route them through the read-only data cache.
#if defined(_MSC_VER)
#define MUEYE_RESTRICT __restrict
#else
#define MUEYE_RESTRICT __restrict__
#endif

namespace mueye {

// ---------------------------------------------------------------------------
// Minimal vector math (self-contained so the CPU build needs no CUDA headers).
// ---------------------------------------------------------------------------

struct Vec3 {
  float x, y, z;
};

struct Vec4 {
  float x, y, z, w;
};

MUEYE_HD inline Vec3 make_vec3(float x, float y, float z) { return Vec3{x, y, z}; }

MUEYE_HD inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
MUEYE_HD inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
MUEYE_HD inline Vec3 operator*(const Vec3 &a, float s) {
  return Vec3{a.x * s, a.y * s, a.z * s};
}
MUEYE_HD inline Vec3 operator*(float s, const Vec3 &a) { return a * s; }
MUEYE_HD inline Vec3 operator*(const Vec3 &a, const Vec3 &b) {
  return Vec3{a.x * b.x, a.y * b.y, a.z * b.z};
}

MUEYE_HD inline float dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
MUEYE_HD inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
}
MUEYE_HD inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
MUEYE_HD inline float maxf(float a, float b) { return a > b ? a : b; }
MUEYE_HD inline float minf(float a, float b) { return a < b ? a : b; }

MUEYE_HD inline Vec3 normalize(const Vec3 &a) {
  float n = sqrtf(dot(a, a));
  float inv = n > 0.0f ? 1.0f / n : 0.0f;
  return a * inv;
}

/**
 * A 3x3 matrix stored row-major: m[3*r + c]. Used for the world->fractional
 * "cell" map (see RenderParams::inv_cell) so muEye can render non-orthogonal
 * (sheared / Bravais) unit cells, not just axis-aligned boxes.
 */
struct Mat3 {
  float m[9];
};

/** Matrix-vector product A * v. */
MUEYE_HD inline Vec3 mat3_mul(const Mat3 &A, const Vec3 &v) {
  return Vec3{A.m[0] * v.x + A.m[1] * v.y + A.m[2] * v.z,
              A.m[3] * v.x + A.m[4] * v.y + A.m[5] * v.z,
              A.m[6] * v.x + A.m[7] * v.y + A.m[8] * v.z};
}

/** Transposed matrix-vector product A^T * v (transforms gradients/normals). */
MUEYE_HD inline Vec3 mat3t_mul(const Mat3 &A, const Vec3 &v) {
  return Vec3{A.m[0] * v.x + A.m[3] * v.y + A.m[6] * v.z,
              A.m[1] * v.x + A.m[4] * v.y + A.m[7] * v.z,
              A.m[2] * v.x + A.m[5] * v.y + A.m[8] * v.z};
}

/** Diagonal matrix from a vector. */
MUEYE_HD inline Mat3 mat3_diag(const Vec3 &d) {
  return Mat3{{d.x, 0.0f, 0.0f, 0.0f, d.y, 0.0f, 0.0f, 0.0f, d.z}};
}

/** Row-major 3x3 matrix product A * B. */
MUEYE_HD inline Mat3 mat3_matmul(const Mat3 &A, const Mat3 &B) {
  Mat3 C{};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      C.m[3 * r + c] = A.m[3 * r + 0] * B.m[0 + c] +
                       A.m[3 * r + 1] * B.m[3 + c] +
                       A.m[3 * r + 2] * B.m[6 + c];
  return C;
}

/** Inverse of a row-major 3x3 matrix (cofactor / determinant); returns the
 *  identity for a (near-)singular matrix so the renderer degrades gracefully. */
MUEYE_HD inline Mat3 mat3_inverse(const Mat3 &A) {
  const float *m = A.m;
  float c00 = m[4] * m[8] - m[5] * m[7];
  float c01 = m[5] * m[6] - m[3] * m[8];
  float c02 = m[3] * m[7] - m[4] * m[6];
  float det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  if (det > -1e-20f && det < 1e-20f)
    return Mat3{{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  float inv = 1.0f / det;
  Mat3 R{};
  R.m[0] = c00 * inv;
  R.m[1] = (m[2] * m[7] - m[1] * m[8]) * inv;
  R.m[2] = (m[1] * m[5] - m[2] * m[4]) * inv;
  R.m[3] = c01 * inv;
  R.m[4] = (m[0] * m[8] - m[2] * m[6]) * inv;
  R.m[5] = (m[2] * m[3] - m[0] * m[5]) * inv;
  R.m[6] = c02 * inv;
  R.m[7] = (m[1] * m[6] - m[0] * m[7]) * inv;
  R.m[8] = (m[0] * m[4] - m[1] * m[3]) * inv;
  return R;
}

/** Build the world->fractional map inv_cell for a deformation gradient F
 *  (row-major 3x3) acting on the reference box: cell C = F * diag(box),
 *  inv_cell = C^{-1}. F = identity reproduces axis_aligned_inv_cell(box). */
MUEYE_HD inline Mat3 inv_cell_from_F(const Mat3 &F, const Vec3 &box) {
  return mat3_inverse(mat3_matmul(F, mat3_diag(box)));
}

/** The world->fractional map for an axis-aligned box [0,box]: diag(1/box).
 *  Feeding this as inv_cell reproduces the pre-cell-matrix behaviour exactly
 *  (fractional coord = world / box, elementwise). */
MUEYE_HD inline Mat3 axis_aligned_inv_cell(const Vec3 &box) {
  return mat3_diag(Vec3{1.0f / box.x, 1.0f / box.y, 1.0f / box.z});
}

// ---------------------------------------------------------------------------
// Camera and render parameters.
// ---------------------------------------------------------------------------

/** A pinhole camera described by an orthonormal basis and a field of view. */
struct Camera {
  Vec3 eye;      //!< camera position in world space
  Vec3 forward;  //!< unit view direction
  Vec3 right;    //!< unit right vector
  Vec3 up;       //!< unit up vector
  float tan_half_fov;  //!< tan(fov_y / 2)
  float aspect;        //!< width / height
};

enum class RenderMode : int { DVR = 0, Isosurface = 1 };

/**
 * Everything the ray-march kernel needs besides the raw buffers. The volume
 * lives in the axis-aligned box [0,box.x] x [0,box.y] x [0,box.z] in world
 * space; voxel (i,j,k) sits at the centre of its cell. `box` is the grid shape
 * normalized so the longest axis is 1 (a cubic grid gives [0,1]^3, unchanged;
 * a 2D grid with nz==1 gives a thin box of z-extent 1/max(nx,ny) — a plane).
 * With periodic replicas (rep_* > 1) the rendered box grows to
 * [0, box.x*rep_x] x [0, box.y*rep_y] x [0, box.z*rep_z] and sampling
 * coordinates wrap back into the unit cell.
 * Data is sampled from a contiguous float buffer in column-major (muGrid)
 * order: index = i + nx*(j + ny*k).
 *
 * `inv_cell` is the world->fractional map: a world point p maps to a fractional
 * cell coordinate uvw = inv_cell * p, and the sampler reads at uvw (in [0,1]^3
 * for one cell). For an axis-aligned box this is diag(1/box) — set it with
 * axis_aligned_inv_cell(box). For a sheared (Bravais) cell whose edge vectors
 * are the columns of a 3x3 matrix C, inv_cell = C^{-1}; the rendered cell is
 * then the parallelepiped C * [0,1]^3. `box` is retained for camera/step
 * bookkeeping and is the axis-aligned extent of the *undeformed* cell.
 */
struct RenderParams {
  int nx, ny, nz;        //!< grid dimensions
  Vec3 box;              //!< world-space extents of the undeformed box [0,box]
  Mat3 inv_cell{{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                 1.0f}};  //!< world->fractional map (see class doc)
  int rep_x{1};          //!< periodic replicas along x (>= 1; 1 = no tiling)
  int rep_y{1};          //!< periodic replicas along y
  int rep_z{1};          //!< periodic replicas along z
  float step;            //!< ray-march step length in world units
  float data_min;        //!< value mapped to LUT entry 0 / iso slider minimum
  float data_max;        //!< value mapped to LUT entry (lut_size-1) / iso maximum
  int lut_size;          //!< number of RGBA entries in the transfer-function LUT
  float density_scale;   //!< global opacity multiplier for DVR
  float iso_value;       //!< iso level (Isosurface mode), in data units
  RenderMode mode;       //!< DVR or Isosurface
  Vec3 bg;               //!< background colour

  // --- optional displacement warp (deformed geometry) -------------------
  // When warp_enabled, the volume is rendered in its deformed configuration
  // x = C*s + D(s), where s is the fractional cell coordinate and D(s) =
  // (warp_scale / max_dim) * u(s) is the world-space displacement (u is the
  // selected vector field, sampled through a DispSampler). The trace kernels
  // invert this per sample (fixed-point) to recover s, so they march the
  // world-space AABB [warp_lo, warp_hi] that encloses the deformed body.
  // Periodic replicas are disabled while warping.
  int warp_enabled{0};
  float warp_scale{1.0f};  //!< user gain on the displacement (world units)
  int warp_iters{6};       //!< fixed-point iterations for the inverse map
  Vec3 warp_lo{0.0f, 0.0f, 0.0f};  //!< world AABB of the deformed body (min)
  Vec3 warp_hi{0.0f, 0.0f, 0.0f};  //!< world AABB of the deformed body (max)
};

// ---------------------------------------------------------------------------
// Sampling.
//
// The volume is reached through a lightweight "sampler" object exposing a
// single value_at(params, uvw) returning the trilinearly filtered value at box
// coordinate uvw in [0,1]^3. ArraySampler below does the filtering in software
// from a contiguous column-major float buffer; it is used by the CPU backend
// and as the reference in muEye_check. GPU backends substitute a sampler backed
// by a hardware 3-D texture that exposes the SAME value_at() interface (see
// GpuRenderer.cc's TextureSampler and the MSL in MetalRenderer.mm), so the
// gradient() and trace_ray() templates below are shared verbatim across all
// backends — only the sampler type differs.
// ---------------------------------------------------------------------------

/** Software trilinear sampler over a column-major float volume. */
struct ArraySampler {
  const float *MUEYE_RESTRICT vol;

  MUEYE_HD float voxel_at(const RenderParams &p, int i, int j, int k) const {
    i = i < 0 ? 0 : (i >= p.nx ? p.nx - 1 : i);
    j = j < 0 ? 0 : (j >= p.ny ? p.ny - 1 : j);
    k = k < 0 ? 0 : (k >= p.nz ? p.nz - 1 : k);
    return vol[i + p.nx * (j + p.ny * k)];
  }

  /** Trilinear sample. @p uvw is in normalized [0,1]^3 box coordinates. */
  MUEYE_HD float value_at(const RenderParams &p, const Vec3 &uvw) const {
    // Map box coordinate to a continuous voxel-centre coordinate.
    float fx = uvw.x * p.nx - 0.5f;
    float fy = uvw.y * p.ny - 0.5f;
    float fz = uvw.z * p.nz - 0.5f;
    int i0 = (int)floorf(fx), j0 = (int)floorf(fy), k0 = (int)floorf(fz);
    float tx = fx - i0, ty = fy - j0, tz = fz - k0;

    float c000 = voxel_at(p, i0, j0, k0);
    float c100 = voxel_at(p, i0 + 1, j0, k0);
    float c010 = voxel_at(p, i0, j0 + 1, k0);
    float c110 = voxel_at(p, i0 + 1, j0 + 1, k0);
    float c001 = voxel_at(p, i0, j0, k0 + 1);
    float c101 = voxel_at(p, i0 + 1, j0, k0 + 1);
    float c011 = voxel_at(p, i0, j0 + 1, k0 + 1);
    float c111 = voxel_at(p, i0 + 1, j0 + 1, k0 + 1);

    float c00 = c000 * (1 - tx) + c100 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;
    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
  }
};

/**
 * Wrap a box coordinate into [0,1) when the axis is periodically replicated
 * (rep > 1); pass it through untouched otherwise, so single-box rendering is
 * bit-identical to the non-replicated path. Samplers clamp at cell faces, so
 * interpolation does not cross replica seams — a half-voxel-wide seam that is
 * identical across backends.
 */
MUEYE_HD inline float wrap_coord(float u, int rep) {
  return rep > 1 ? u - floorf(u) : u;
}

/** Wrap a box coordinate per axis according to the replica counts. */
MUEYE_HD inline Vec3 wrap_uvw(const RenderParams &p, const Vec3 &uvw) {
  return Vec3{wrap_coord(uvw.x, p.rep_x), wrap_coord(uvw.y, p.rep_y),
              wrap_coord(uvw.z, p.rep_z)};
}

/** Central-difference gradient in box coordinates (for surface shading). */
template <class Sampler>
MUEYE_HD inline Vec3 gradient(const Sampler &s, const RenderParams &p,
                              const Vec3 &uvw) {
  float hx = 1.0f / p.nx, hy = 1.0f / p.ny, hz = 1.0f / p.nz;
  float gx = s.value_at(p, Vec3{wrap_coord(uvw.x + hx, p.rep_x), uvw.y, uvw.z}) -
             s.value_at(p, Vec3{wrap_coord(uvw.x - hx, p.rep_x), uvw.y, uvw.z});
  float gy = s.value_at(p, Vec3{uvw.x, wrap_coord(uvw.y + hy, p.rep_y), uvw.z}) -
             s.value_at(p, Vec3{uvw.x, wrap_coord(uvw.y - hy, p.rep_y), uvw.z});
  float gz = s.value_at(p, Vec3{uvw.x, uvw.y, wrap_coord(uvw.z + hz, p.rep_z)}) -
             s.value_at(p, Vec3{uvw.x, uvw.y, wrap_coord(uvw.z - hz, p.rep_z)});
  return Vec3{gx, gy, gz};
}

MUEYE_HD inline float normalize_value(const RenderParams &p, float v) {
  float range = p.data_max - p.data_min;
  if (range <= 0.0f) return 0.0f;
  return clampf((v - p.data_min) / range, 0.0f, 1.0f);
}

/** Look up the transfer-function LUT (RGBA, premultiplied later). */
MUEYE_HD inline Vec4 lut_lookup(const Vec4 *lut, const RenderParams &p,
                                float normalized) {
  float f = normalized * (p.lut_size - 1);
  int i0 = (int)f;
  if (i0 < 0) i0 = 0;
  if (i0 > p.lut_size - 1) i0 = p.lut_size - 1;
  int i1 = i0 < p.lut_size - 1 ? i0 + 1 : i0;
  float t = f - i0;
  Vec4 a = lut[i0], b = lut[i1];
  return Vec4{a.x * (1 - t) + b.x * t, a.y * (1 - t) + b.y * t,
              a.z * (1 - t) + b.z * t, a.w * (1 - t) + b.w * t};
}

// ---------------------------------------------------------------------------
// Ray / box intersection.
// ---------------------------------------------------------------------------

/** Intersect a ray with the axis-aligned box [0,box]. Returns false if missed. */
MUEYE_HD inline bool intersect_box(const Vec3 &o, const Vec3 &d, const Vec3 &box,
                                   float &t_near, float &t_far) {
  float tmin = -1e30f, tmax = 1e30f;
  for (int axis = 0; axis < 3; ++axis) {
    float oa = axis == 0 ? o.x : (axis == 1 ? o.y : o.z);
    float da = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
    float hi = axis == 0 ? box.x : (axis == 1 ? box.y : box.z);
    if (fabsf(da) < 1e-8f) {
      if (oa < 0.0f || oa > hi) return false;
    } else {
      float inv = 1.0f / da;
      float t1 = (0.0f - oa) * inv;
      float t2 = (hi - oa) * inv;
      if (t1 > t2) {
        float tmp = t1;
        t1 = t2;
        t2 = tmp;
      }
      tmin = maxf(tmin, t1);
      tmax = minf(tmax, t2);
      if (tmin > tmax) return false;
    }
  }
  t_near = maxf(tmin, 0.0f);
  t_far = tmax;
  return t_far > t_near;
}

/** Intersect a ray with the world-space AABB [lo,hi] (used by the warp path,
 *  which cannot fold the deformed body into the unit box). */
MUEYE_HD inline bool intersect_aabb(const Vec3 &o, const Vec3 &d, const Vec3 &lo,
                                    const Vec3 &hi, float &t_near, float &t_far) {
  float tmin = -1e30f, tmax = 1e30f;
  for (int axis = 0; axis < 3; ++axis) {
    float oa = axis == 0 ? o.x : (axis == 1 ? o.y : o.z);
    float da = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
    float l = axis == 0 ? lo.x : (axis == 1 ? lo.y : lo.z);
    float h = axis == 0 ? hi.x : (axis == 1 ? hi.y : hi.z);
    if (fabsf(da) < 1e-8f) {
      if (oa < l || oa > h) return false;
    } else {
      float inv = 1.0f / da;
      float t1 = (l - oa) * inv;
      float t2 = (h - oa) * inv;
      if (t1 > t2) {
        float tmp = t1;
        t1 = t2;
        t2 = tmp;
      }
      tmin = maxf(tmin, t1);
      tmax = minf(tmax, t2);
      if (tmin > tmax) return false;
    }
  }
  t_near = maxf(tmin, 0.0f);
  t_far = tmax;
  return t_far > t_near;
}

/** Software trilinear sampler of a 3-vector displacement field stored as 4
 *  floats per voxel (x,y,z,unused), column-major like ArraySampler. The CPU
 *  backend uses this; GPU backends substitute a float4-texture sampler exposing
 *  the same disp_at(). */
struct ArrayDispSampler {
  const float *MUEYE_RESTRICT disp;  //!< 4 floats per voxel, or null when unused

  MUEYE_HD Vec3 voxel(const RenderParams &p, int i, int j, int k) const {
    i = i < 0 ? 0 : (i >= p.nx ? p.nx - 1 : i);
    j = j < 0 ? 0 : (j >= p.ny ? p.ny - 1 : j);
    k = k < 0 ? 0 : (k >= p.nz ? p.nz - 1 : k);
    const float *v = disp + 4 * (i + p.nx * (j + p.ny * k));
    return Vec3{v[0], v[1], v[2]};
  }

  MUEYE_HD Vec3 disp_at(const RenderParams &p, const Vec3 &uvw) const {
    float fx = uvw.x * p.nx - 0.5f;
    float fy = uvw.y * p.ny - 0.5f;
    float fz = uvw.z * p.nz - 0.5f;
    int i0 = (int)floorf(fx), j0 = (int)floorf(fy), k0 = (int)floorf(fz);
    float tx = fx - i0, ty = fy - j0, tz = fz - k0;
    Vec3 c000 = voxel(p, i0, j0, k0), c100 = voxel(p, i0 + 1, j0, k0);
    Vec3 c010 = voxel(p, i0, j0 + 1, k0), c110 = voxel(p, i0 + 1, j0 + 1, k0);
    Vec3 c001 = voxel(p, i0, j0, k0 + 1), c101 = voxel(p, i0 + 1, j0, k0 + 1);
    Vec3 c011 = voxel(p, i0, j0 + 1, k0 + 1),
         c111 = voxel(p, i0 + 1, j0 + 1, k0 + 1);
    Vec3 c00 = c000 * (1 - tx) + c100 * tx, c10 = c010 * (1 - tx) + c110 * tx;
    Vec3 c01 = c001 * (1 - tx) + c101 * tx, c11 = c011 * (1 - tx) + c111 * tx;
    Vec3 c0 = c00 * (1 - ty) + c10 * ty, c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
  }
};

/** The world->fractional displacement gain: raw field-unit displacements are
 *  scaled to world units by 1/max_dim (one voxel of the longest axis spans that
 *  in world space) times the user's warp_scale. */
MUEYE_HD inline float warp_gain(const RenderParams &p) {
  int max_dim = p.nx > p.ny ? p.nx : p.ny;
  if (p.nz > max_dim) max_dim = p.nz;
  return p.warp_scale / static_cast<float>(max_dim > 0 ? max_dim : 1);
}

/**
 * Recover the fractional cell coordinate s of a world point x under the
 * displacement warp x = C*s + D(s). Fixed-point iteration
 * s <- inv_cell * (x - D(s)); converges for moderate displacement gradients.
 * `outside` is set when the recovered s leaves the unit cell (i.e. the world
 * point is not inside the deformed body).
 */
template <class DispSampler>
MUEYE_HD inline Vec3 world_to_reference(const DispSampler &d,
                                        const RenderParams &p, const Vec3 &x,
                                        bool &outside) {
  Vec3 s = mat3_mul(p.inv_cell, x);
  float g = warp_gain(p);
  for (int it = 0; it < p.warp_iters; ++it) {
    Vec3 sc = Vec3{clampf(s.x, 0.0f, 1.0f), clampf(s.y, 0.0f, 1.0f),
                   clampf(s.z, 0.0f, 1.0f)};
    Vec3 world_d = d.disp_at(p, sc) * g;
    s = mat3_mul(p.inv_cell, x - world_d);
  }
  const float e = 1e-4f;
  outside = s.x < -e || s.x > 1.0f + e || s.y < -e || s.y > 1.0f + e ||
            s.z < -e || s.z > 1.0f + e;
  return s;
}

// ---------------------------------------------------------------------------
// The ray-march kernel. (u, v) are normalized image coordinates in [0,1].
// ---------------------------------------------------------------------------

/** Primary (pinhole) ray direction through normalized image coordinate (u,v). */
MUEYE_HD inline Vec3 primary_ray_dir(const Camera &cam, float u, float v) {
  float sx = (2.0f * u - 1.0f) * cam.aspect * cam.tan_half_fov;
  float sy = (1.0f - 2.0f * v) * cam.tan_half_fov;
  return normalize(cam.forward + cam.right * sx + cam.up * sy);
}

/**
 * Isosurface ray-cast: march for a sign change against p.iso_value, then Phong
 * shade the refined crossing.
 *
 * DVR and isosurface are separate entry points (rather than one function with a
 * runtime mode branch) so the GPU backends can launch a single-path kernel per
 * mode: p.mode is grid-uniform, so the branch never diverges, but compiling both
 * paths into one kernel inflates register/instruction footprint and can cap
 * occupancy. The CPU dispatcher trace_ray() below still selects at runtime — a
 * scalar CPU core has no such footprint concern.
 */
template <class Sampler, class DispSampler>
MUEYE_HD inline Vec4 trace_ray_iso(const Sampler &s, const DispSampler &disp,
                                   const RenderParams &p, const Camera &cam,
                                   float u, float v) {
  Vec3 dir = primary_ray_dir(cam, u, v);
  float t_near, t_far;
  Vec4 bg = Vec4{p.bg.x, p.bg.y, p.bg.z, 1.0f};

  if (p.warp_enabled) {
    // Deformed geometry: march the world AABB of the deformed body and recover
    // the fractional coordinate per sample (fixed-point inverse of the warp).
    if (!intersect_aabb(cam.eye, dir, p.warp_lo, p.warp_hi, t_near, t_far))
      return bg;
    float t = t_near;
    bool have_prev = false;
    float prev_v = 0.0f;
    float prev_t = t_near;
    while (t < t_far) {
      Vec3 x = cam.eye + dir * t;
      bool outside;
      Vec3 sref = world_to_reference(disp, p, x, outside);
      if (!outside) {
        float cur_v = s.value_at(p, sref) - p.iso_value;
        if (have_prev && prev_v * cur_v <= 0.0f) {
          float denom = (cur_v - prev_v);
          float frac = fabsf(denom) > 1e-12f ? prev_v / -denom : 0.0f;
          float t_hit = prev_t + (t - prev_t) * frac;
          Vec3 xh = cam.eye + dir * t_hit;
          bool o2;
          Vec3 sh = world_to_reference(disp, p, xh, o2);
          Vec3 n = normalize(mat3t_mul(p.inv_cell, gradient(s, p, sh)));
          Vec3 l = normalize(cam.eye - xh);
          float diffuse = fabsf(dot(n, l));
          Vec3 base = make_vec3(0.82f, 0.45f, 0.20f);
          Vec3 col = base * (0.20f + 0.80f * diffuse);
          return Vec4{col.x, col.y, col.z, 1.0f};
        }
        prev_v = cur_v;
        prev_t = t;
        have_prev = true;
      } else {
        have_prev = false;  // don't bridge a crossing across the empty margin
      }
      t += p.step;
    }
    return bg;
  }

  // Work in fractional cell coordinates: the affine map uvw = inv_cell * world
  // turns the (possibly sheared) cell into the unit box, so the ray-parameter t
  // is preserved and the existing box test applies to the tiled unit box.
  Vec3 o_frac = mat3_mul(p.inv_cell, cam.eye);
  Vec3 d_frac = mat3_mul(p.inv_cell, dir);
  Vec3 tiled_box = make_vec3(static_cast<float>(p.rep_x),
                             static_cast<float>(p.rep_y),
                             static_cast<float>(p.rep_z));
  if (!intersect_box(o_frac, d_frac, tiled_box, t_near, t_far)) return bg;

  float t = t_near;
  Vec3 prev = o_frac + d_frac * t;
  float prev_v = s.value_at(p, wrap_uvw(p, prev)) - p.iso_value;
  t += p.step;
  while (t < t_far) {
    Vec3 uvw = o_frac + d_frac * t;
    float cur_v = s.value_at(p, wrap_uvw(p, uvw)) - p.iso_value;
    if (prev_v * cur_v <= 0.0f) {
      // Linear refinement of the crossing, in the fractional-coordinate ray.
      float denom = (cur_v - prev_v);
      float frac = fabsf(denom) > 1e-12f ? prev_v / -denom : 0.0f;
      Vec3 hit = prev + (uvw - prev) * frac;
      // Gradient is in fractional coords; inv_cell^T maps it to a world-space
      // normal (a no-op for a cubic cell, a shear-correcting rotation otherwise).
      Vec3 n = normalize(mat3t_mul(p.inv_cell, gradient(s, p, wrap_uvw(p, hit))));
      // Two-sided Phong with a head light at the eye. The world hit position is
      // recovered from the same crossing parameter along the world ray.
      float t_hit = (t - p.step) + p.step * frac;
      Vec3 world_hit = cam.eye + dir * t_hit;
      // mid-tone material so the surface reads clearly against a light/white
      // background (a near-white material would disappear).
      Vec3 l = normalize(cam.eye - world_hit);
      float diff = fabsf(dot(n, l));
      Vec3 base = make_vec3(0.82f, 0.45f, 0.20f);
      Vec3 col = base * (0.20f + 0.80f * diff);
      return Vec4{col.x, col.y, col.z, 1.0f};
    }
    prev = uvw;
    prev_v = cur_v;
    t += p.step;
  }
  return bg;
}

/** Direct volume rendering: front-to-back emission-absorption compositing. */
template <class Sampler, class DispSampler>
MUEYE_HD inline Vec4 trace_ray_dvr(const Sampler &s, const DispSampler &disp,
                                   const Vec4 *MUEYE_RESTRICT lut,
                                   const RenderParams &p, const Camera &cam,
                                   float u, float v) {
  Vec3 dir = primary_ray_dir(cam, u, v);
  float t_near, t_far;
  // Opacity correction: p.step is in world units and the longest grid axis
  // spans one world unit, so step * max_dim is the step length in voxels —
  // the unit the LUT's per-voxel opacity is defined in. Using the longest
  // axis (not nx) keeps the density independent of the data's axis order.
  int max_dim = p.nx > p.ny ? p.nx : p.ny;
  if (p.nz > max_dim) max_dim = p.nz;

  if (p.warp_enabled) {
    // Deformed geometry: march the world AABB, invert the warp per sample.
    if (!intersect_aabb(cam.eye, dir, p.warp_lo, p.warp_hi, t_near, t_far))
      return Vec4{p.bg.x, p.bg.y, p.bg.z, 1.0f};
    Vec3 accum = make_vec3(0.0f, 0.0f, 0.0f);
    float trans = 1.0f;
    for (float t = t_near; t < t_far; t += p.step) {
      Vec3 x = cam.eye + dir * t;
      bool outside;
      Vec3 sref = world_to_reference(disp, p, x, outside);
      if (outside) continue;  // outside the deformed body
      float val = s.value_at(p, sref);
      float nv = normalize_value(p, val);
      Vec4 c = lut_lookup(lut, p, nv);
      float alpha = clampf(c.w * p.density_scale * p.step * max_dim, 0.0f, 1.0f);
      accum = accum + make_vec3(c.x, c.y, c.z) * (alpha * trans);
      trans *= (1.0f - alpha);
      if (trans < 0.003f) break;
    }
    Vec3 out = accum + p.bg * trans;
    return Vec4{out.x, out.y, out.z, 1.0f};
  }

  // Work in fractional cell coordinates: uvw = inv_cell * world maps the
  // (possibly sheared) cell onto the unit box, preserving the ray-parameter t.
  Vec3 o_frac = mat3_mul(p.inv_cell, cam.eye);
  Vec3 d_frac = mat3_mul(p.inv_cell, dir);
  Vec3 tiled_box = make_vec3(static_cast<float>(p.rep_x),
                             static_cast<float>(p.rep_y),
                             static_cast<float>(p.rep_z));
  if (!intersect_box(o_frac, d_frac, tiled_box, t_near, t_far))
    return Vec4{p.bg.x, p.bg.y, p.bg.z, 1.0f};

  Vec3 accum = make_vec3(0.0f, 0.0f, 0.0f);
  float trans = 1.0f;  // remaining transparency
  for (float t = t_near; t < t_far; t += p.step) {
    Vec3 uvw = o_frac + d_frac * t;
    float val = s.value_at(p, wrap_uvw(p, uvw));
    float nv = normalize_value(p, val);
    Vec4 c = lut_lookup(lut, p, nv);
    // Opacity correction for the step size, then global density scale.
    float alpha = clampf(c.w * p.density_scale * p.step * max_dim, 0.0f, 1.0f);
    Vec3 crgb = make_vec3(c.x, c.y, c.z);
    accum = accum + crgb * (alpha * trans);
    trans *= (1.0f - alpha);
    if (trans < 0.003f) break;  // early ray termination
  }
  // Composite over the background.
  Vec3 out = accum + p.bg * trans;
  return Vec4{out.x, out.y, out.z, 1.0f};
}

/** Runtime-dispatched entry point (CPU backend + single-kernel callers). */
template <class Sampler, class DispSampler>
MUEYE_HD inline Vec4 trace_ray(const Sampler &s, const DispSampler &disp,
                               const Vec4 *MUEYE_RESTRICT lut,
                               const RenderParams &p, const Camera &cam, float u,
                               float v) {
  return p.mode == RenderMode::Isosurface
             ? trace_ray_iso(s, disp, p, cam, u, v)
             : trace_ray_dvr(s, disp, lut, p, cam, u, v);
}

}  // namespace mueye

#endif  // MUEYE_RENDER_CORE_HH_
