/**
 * @file   OrbitCamera.cc
 *
 * @brief  Implementation of the orbit camera.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "ui/OrbitCamera.hh"

#include <algorithm>
#include <cmath>

namespace mueye {

void OrbitCamera::reset() {
  target_ = Vec3{0.5f, 0.5f, 0.5f};
  distance_ = 2.5f;
  yaw_ = 0.785f;
  pitch_ = 0.5f;
  fov_y_deg = 45.0f;
}

void OrbitCamera::orbit(float dyaw, float dpitch) {
  yaw_ += dyaw;
  pitch_ += dpitch;
  const float lim = 1.55f;  // ~89 degrees, avoid gimbal flip
  pitch_ = std::clamp(pitch_, -lim, lim);
}

void OrbitCamera::zoom(float delta) {
  distance_ *= std::exp(-delta * 0.15f);
  distance_ = std::clamp(distance_, 0.3f, 20.0f);
}

void OrbitCamera::pan(float dx, float dy) {
  // Translate the target in the current view plane.
  float cy = std::cos(yaw_), sy = std::sin(yaw_);
  float cp = std::cos(pitch_), sp = std::sin(pitch_);
  Vec3 forward{cp * cy, sp, cp * sy};
  Vec3 world_up{0.0f, 1.0f, 0.0f};
  Vec3 right = normalize(cross(forward, world_up));
  Vec3 up = normalize(cross(right, forward));
  float scale = distance_ * 0.5f;
  target_ = target_ + right * (-dx * scale) + up * (dy * scale);
}

Camera OrbitCamera::to_camera(float aspect) const {
  float cy = std::cos(yaw_), sy = std::sin(yaw_);
  float cp = std::cos(pitch_), sp = std::sin(pitch_);
  // Eye orbits target at the given spherical angles.
  Vec3 dir{cp * cy, sp, cp * sy};
  Vec3 eye = target_ + dir * distance_;

  Vec3 forward = normalize(target_ - eye);
  Vec3 world_up{0.0f, 1.0f, 0.0f};
  Vec3 right = normalize(cross(forward, world_up));
  Vec3 up = normalize(cross(right, forward));

  Camera cam;
  cam.eye = eye;
  cam.forward = forward;
  cam.right = right;
  cam.up = up;
  cam.tan_half_fov = std::tan(0.5f * fov_y_deg * 3.14159265f / 180.0f);
  cam.aspect = aspect;
  return cam;
}

}  // namespace mueye
