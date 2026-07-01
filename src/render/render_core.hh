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
 * Everything the ray-march kernel needs besides the raw buffers. The volume is
 * assumed to live in the axis-aligned box [0,1]^3 in world space; voxel (i,j,k)
 * sits at the centre of its cell. Data is sampled from a contiguous float buffer
 * in column-major (muGrid) order: index = i + nx*(j + ny*k).
 */
struct RenderParams {
  int nx, ny, nz;        //!< grid dimensions
  float step;            //!< ray-march step length in world units (box is unit-size)
  float data_min;        //!< value mapped to LUT entry 0 / iso slider minimum
  float data_max;        //!< value mapped to LUT entry (lut_size-1) / iso maximum
  int lut_size;          //!< number of RGBA entries in the transfer-function LUT
  float density_scale;   //!< global opacity multiplier for DVR
  float iso_value;       //!< iso level (Isosurface mode), in data units
  RenderMode mode;       //!< DVR or Isosurface
  Vec3 bg;               //!< background colour
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

/** Central-difference gradient in box coordinates (for surface shading). */
template <class Sampler>
MUEYE_HD inline Vec3 gradient(const Sampler &s, const RenderParams &p,
                              const Vec3 &uvw) {
  float hx = 1.0f / p.nx, hy = 1.0f / p.ny, hz = 1.0f / p.nz;
  float gx = s.value_at(p, Vec3{uvw.x + hx, uvw.y, uvw.z}) -
             s.value_at(p, Vec3{uvw.x - hx, uvw.y, uvw.z});
  float gy = s.value_at(p, Vec3{uvw.x, uvw.y + hy, uvw.z}) -
             s.value_at(p, Vec3{uvw.x, uvw.y - hy, uvw.z});
  float gz = s.value_at(p, Vec3{uvw.x, uvw.y, uvw.z + hz}) -
             s.value_at(p, Vec3{uvw.x, uvw.y, uvw.z - hz});
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

/** Intersect a ray with the unit box [0,1]^3. Returns false if missed. */
MUEYE_HD inline bool intersect_unit_box(const Vec3 &o, const Vec3 &d,
                                        float &t_near, float &t_far) {
  float tmin = -1e30f, tmax = 1e30f;
  // x
  for (int axis = 0; axis < 3; ++axis) {
    float oa = axis == 0 ? o.x : (axis == 1 ? o.y : o.z);
    float da = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
    if (fabsf(da) < 1e-8f) {
      if (oa < 0.0f || oa > 1.0f) return false;
    } else {
      float inv = 1.0f / da;
      float t1 = (0.0f - oa) * inv;
      float t2 = (1.0f - oa) * inv;
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

// ---------------------------------------------------------------------------
// The ray-march kernel. (u, v) are normalized image coordinates in [0,1].
// ---------------------------------------------------------------------------

template <class Sampler>
MUEYE_HD inline Vec4 trace_ray(const Sampler &s, const Vec4 *MUEYE_RESTRICT lut,
                               const RenderParams &p, const Camera &cam,
                               float u, float v) {
  // Build the primary ray through pixel (u, v).
  float sx = (2.0f * u - 1.0f) * cam.aspect * cam.tan_half_fov;
  float sy = (1.0f - 2.0f * v) * cam.tan_half_fov;
  Vec3 dir = normalize(cam.forward + cam.right * sx + cam.up * sy);

  float t_near, t_far;
  Vec4 bg = Vec4{p.bg.x, p.bg.y, p.bg.z, 1.0f};
  if (!intersect_unit_box(cam.eye, dir, t_near, t_far)) return bg;

  if (p.mode == RenderMode::Isosurface) {
    // March looking for a sign change relative to the iso value.
    float t = t_near;
    Vec3 prev = cam.eye + dir * t;
    float prev_v = s.value_at(p, prev) - p.iso_value;
    t += p.step;
    while (t < t_far) {
      Vec3 pos = cam.eye + dir * t;
      float cur_v = s.value_at(p, pos) - p.iso_value;
      if (prev_v * cur_v <= 0.0f) {
        // Linear refinement of the crossing.
        float denom = (cur_v - prev_v);
        float frac = fabsf(denom) > 1e-12f ? prev_v / -denom : 0.0f;
        Vec3 hit = prev + (pos - prev) * frac;
        Vec3 n = normalize(gradient(s, p, hit));
        // Two-sided Phong with a head light along the view direction. A warm,
        // mid-tone material so the surface reads clearly against a light/white
        // background (a near-white material would disappear).
        Vec3 l = normalize(cam.eye - hit);
        float diff = fabsf(dot(n, l));
        Vec3 base = make_vec3(0.82f, 0.45f, 0.20f);
        Vec3 col = base * (0.20f + 0.80f * diff);
        return Vec4{col.x, col.y, col.z, 1.0f};
      }
      prev = pos;
      prev_v = cur_v;
      t += p.step;
    }
    return bg;
  }

  // Direct volume rendering: front-to-back emission-absorption compositing.
  Vec3 accum = make_vec3(0.0f, 0.0f, 0.0f);
  float trans = 1.0f;  // remaining transparency
  for (float t = t_near; t < t_far; t += p.step) {
    Vec3 pos = cam.eye + dir * t;
    float val = s.value_at(p, pos);
    float nv = normalize_value(p, val);
    Vec4 c = lut_lookup(lut, p, nv);
    // Opacity correction for the step size, then global density scale.
    float alpha = clampf(c.w * p.density_scale * p.step * p.nx, 0.0f, 1.0f);
    Vec3 crgb = make_vec3(c.x, c.y, c.z);
    accum = accum + crgb * (alpha * trans);
    trans *= (1.0f - alpha);
    if (trans < 0.003f) break;  // early ray termination
  }
  // Composite over the background.
  Vec3 out = accum + p.bg * trans;
  return Vec4{out.x, out.y, out.z, 1.0f};
}

}  // namespace mueye

#endif  // MUEYE_RENDER_CORE_HH_
