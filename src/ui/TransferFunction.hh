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

  const Vec4 *data() const { return lut_.data(); }
  int size() const { return kSize; }

 private:
  void rebuild();

  Colormap colormap_{Colormap::Viridis};
  float opacity_scale_{1.0f};
  float opacity_gamma_{1.5f};
  std::vector<Vec4> lut_ = std::vector<Vec4>(kSize);
};

}  // namespace mueye

#endif  // MUEYE_TRANSFER_FUNCTION_HH_
