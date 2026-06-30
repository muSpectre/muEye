/**
 * @file   TransferFunction.cc
 *
 * @brief  Colormap tables and LUT assembly.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "ui/TransferFunction.hh"

#include <cmath>

namespace mueye {

const char *to_string(Colormap c) {
  switch (c) {
    case Colormap::Viridis:
      return "Viridis";
    case Colormap::Grayscale:
      return "Grayscale";
    case Colormap::CoolWarm:
      return "Cool-Warm";
  }
  return "?";
}

namespace {

// A few control points of the matplotlib "viridis" colormap.
const float kViridis[][3] = {
    {0.267f, 0.005f, 0.329f}, {0.283f, 0.141f, 0.458f},
    {0.254f, 0.265f, 0.530f}, {0.207f, 0.372f, 0.553f},
    {0.164f, 0.471f, 0.558f}, {0.128f, 0.567f, 0.551f},
    {0.135f, 0.659f, 0.518f}, {0.267f, 0.749f, 0.441f},
    {0.478f, 0.821f, 0.318f}, {0.741f, 0.873f, 0.150f},
    {0.993f, 0.906f, 0.144f}};

// "cool-warm" diverging map (blue -> light grey -> red).
const float kCoolWarm[][3] = {
    {0.230f, 0.299f, 0.754f}, {0.406f, 0.538f, 0.934f},
    {0.602f, 0.731f, 0.999f}, {0.788f, 0.846f, 0.939f},
    {0.917f, 0.831f, 0.759f}, {0.958f, 0.680f, 0.531f},
    {0.870f, 0.466f, 0.341f}, {0.706f, 0.016f, 0.150f}};

void eval_ramp(const float table[][3], int n, float t, float &r, float &g,
               float &b) {
  if (t < 0.f) t = 0.f;
  if (t > 1.f) t = 1.f;
  float f = t * (n - 1);
  int i0 = static_cast<int>(f);
  int i1 = i0 < n - 1 ? i0 + 1 : i0;
  float u = f - i0;
  r = table[i0][0] * (1 - u) + table[i1][0] * u;
  g = table[i0][1] * (1 - u) + table[i1][1] * u;
  b = table[i0][2] * (1 - u) + table[i1][2] * u;
}

}  // namespace

void TransferFunction::rebuild() {
  for (int i = 0; i < kSize; ++i) {
    float t = static_cast<float>(i) / (kSize - 1);
    float r, g, b;
    switch (colormap_) {
      case Colormap::Grayscale:
        r = g = b = t;
        break;
      case Colormap::CoolWarm:
        eval_ramp(kCoolWarm, 8, t, r, g, b);
        break;
      case Colormap::Viridis:
      default:
        eval_ramp(kViridis, 11, t, r, g, b);
        break;
    }
    float alpha = std::pow(t, opacity_gamma_) * opacity_scale_;
    if (alpha < 0.f) alpha = 0.f;
    if (alpha > 1.f) alpha = 1.f;
    lut_[i] = Vec4{r, g, b, alpha};
  }
}

}  // namespace mueye
