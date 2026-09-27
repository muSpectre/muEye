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

void OrbitCamera::frame_box(const Vec3 &box) {
  reset();
  target_ = box * 0.5f;  // centre of the [0,box] volume box
}

void OrbitCamera::retarget_box(const Vec3 &box) {
  target_ = box * 0.5f;
  // Same distance-to-extent ratio as reset() uses for the unit box.
  float m = box.x;
  if (box.y > m) m = box.y;
  if (box.z > m) m = box.z;
  distance_ = 2.5f * (m > 0.0f ? m : 1.0f);
}

void OrbitCamera::frame_aabb(const Vec3 &center, float extent) {
  reset();
  target_ = center;
  distance_ = 2.5f * (extent > 0.0f ? extent : 1.0f);
}

void OrbitCamera::retarget_aabb(const Vec3 &center, float extent) {
  target_ = center;
  distance_ = 2.5f * (extent > 0.0f ? extent : 1.0f);
}

void OrbitCamera::orbit(float dyaw, float dpitch) {
  set_view_angles(yaw_ + dyaw, pitch_ + dpitch);
}

void OrbitCamera::set_view_angles(float yaw, float pitch) {
  yaw_ = yaw;
  const float lim = 1.55f;  // ~89 degrees, avoid gimbal flip
  pitch_ = std::clamp(pitch, -lim, lim);
}

void OrbitCamera::zoom(float delta) {
  distance_ *= std::exp(-delta * 0.15f);
  // The upper bound must stay comfortably beyond retarget_box()'s distance
  // for the largest allowed replica tiling.
  distance_ = std::clamp(distance_, 0.3f, 50.0f);
}

void OrbitCamera::pan(float dx, float dy, float aspect) {
  // Translate the target in the current view plane, using the same basis the
  // renderer uses. "Grab" semantics: the scene follows the cursor 1:1. At the
  // target's depth the viewport spans 2*distance*tan(fov/2) world units
  // vertically (times the aspect horizontally), so a drag across the whole
  // viewport moves the target by exactly that. Image-right is +cam.right, so
  // a rightward drag (dx > 0) must move the camera *left*; image-down is
  // -cam.up, so a downward drag (dy > 0, screen coordinates) moves the
  // camera up.
  Camera cam = to_camera(aspect);
  float span_y = 2.0f * distance_ * cam.tan_half_fov;
  float span_x = span_y * aspect;
  target_ = target_ - cam.right * (dx * span_x) + cam.up * (dy * span_y);
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
