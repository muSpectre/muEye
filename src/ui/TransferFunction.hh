/**
 * @file   TransferFunction.hh
 *
 * @brief  Transfer-function LUT (colormap + opacity) for direct volume
 *         rendering.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_TRANSFER_FUNCTION_HH_
#define MUEYE_TRANSFER_FUNCTION_HH_

#include <vector>

#include "render/render_core.hh"

namespace mueye {

enum class Colormap : int { Viridis = 0, Grayscale = 1, CoolWarm = 2 };

const char *to_string(Colormap c);

/** Shape of the opacity ramp over the normalized value range. */
enum class OpacityRamp : int { Ascending = 0, Descending = 1, Symmetric = 2 };

const char *to_string(OpacityRamp r);

/** Builds a 256-entry RGBA LUT from a named colormap and an opacity ramp. */
class TransferFunction {
 public:
  static constexpr int kSize = 256;

  TransferFunction() { rebuild(); }

  void set_colormap(Colormap c) {
    if (c != colormap_) {
      colormap_ = c;
      rebuild();
    }
  }
  Colormap colormap() const { return colormap_; }

  /** Global opacity multiplier applied to the (value-proportional) ramp. */
  void set_opacity_scale(float s) {
    if (s != opacity_scale_) {
      opacity_scale_ = s;
      rebuild();
    }
  }
  float opacity_scale() const { return opacity_scale_; }

  /** Exponent of the opacity ramp (1 = linear; >1 emphasises high values). */
  void set_opacity_gamma(float g) {
    if (g != opacity_gamma_) {
      opacity_gamma_ = g;
      rebuild();
    }
  }
  float opacity_gamma() const { return opacity_gamma_; }

  /** Fraction of the normalized value range below which voxels are fully
   *  transparent (alpha exactly 0); the ramp restarts from zero at the cutoff.
   *  Besides hiding the "void" of a density field, this is what lets the ray
   *  marcher skip bricks whose values all lie below the cutoff. */
  void set_opacity_cutoff(float c) {
    c = c < 0.0f ? 0.0f : (c > 0.99f ? 0.99f : c);
    if (c != opacity_cutoff_) {
      opacity_cutoff_ = c;
      rebuild();
    }
  }
  float opacity_cutoff() const { return opacity_cutoff_; }

  /** Which end of the value range is transparent. Ascending (the default)
   *  hides low values; Descending hides high values; Symmetric is transparent
   *  at the centre of the range and opaque towards both ends, the natural
   *  choice for a diverging colormap around zero (with Ascending the whole
   *  negative half of a Cool-Warm map would be invisible). */
  void set_ramp(OpacityRamp r) {
    if (r != ramp_) {
      ramp_ = r;
      rebuild();
    }
  }
  OpacityRamp ramp() const { return ramp_; }

  const Vec4 *data() const { return lut_.data(); }
  int size() const { return kSize; }

  /** Largest normalized value v such that every LUT lookup of a value < v is
   *  exactly transparent (both interpolated entries have alpha 0), derived from
   *  the actual LUT contents. Returns 0 when no such band exists. Used to
   *  decide which bricks the ray marcher may skip. */
  float transparent_below() const;

  /** RenderParams::skip_below for a volume with range [data_min, data_max]:
   *  the data value below which lookups are exactly transparent, less a small
   *  margin for the trilinear interpolation's rounding; -1e30 when the LUT
   *  has no transparent band (skipping disabled). */
  float skip_below(float data_min, float data_max) const {
    float thr = transparent_below();
    float range = data_max - data_min;
    if (thr <= 0.0f || range <= 0.0f) return -1e30f;
    return data_min + thr * range - 1e-6f * range;
  }

 private:
  void rebuild();

  Colormap colormap_{Colormap::Viridis};
  float opacity_scale_{1.0f};
  float opacity_gamma_{1.5f};
  float opacity_cutoff_{0.02f};
  OpacityRamp ramp_{OpacityRamp::Ascending};
  std::vector<Vec4> lut_ = std::vector<Vec4>(kSize);
};

}  // namespace mueye

#endif  // MUEYE_TRANSFER_FUNCTION_HH_
