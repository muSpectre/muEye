/**
 * @file   OrbitCamera.hh
 *
 * @brief  Orbit camera that maps mouse input to a render_core::Camera.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_ORBIT_CAMERA_HH_
#define MUEYE_ORBIT_CAMERA_HH_

#include "render/render_core.hh"

namespace mueye {

/** A camera orbiting a target point. The volume occupies the unit box, so the
 *  default target is its centre (0.5, 0.5, 0.5). */
class OrbitCamera {
 public:
  OrbitCamera() { reset(); }

  void reset();

  /** Orbit by mouse drag deltas (in radians-equivalent screen units). */
  void orbit(float dyaw, float dpitch);
  /** Dolly in/out (e.g. mouse wheel); positive zooms in. */
  void zoom(float delta);
  /** Pan the target in the camera plane. */
  void pan(float dx, float dy);

  /** Build the render_core camera for an image of the given aspect ratio. */
  Camera to_camera(float aspect) const;

  float fov_y_deg{45.0f};

 private:
  Vec3 target_{0.5f, 0.5f, 0.5f};
  float distance_{2.5f};
  float yaw_{0.785f};    //!< azimuth (radians)
  float pitch_{0.5f};    //!< elevation (radians)
};

}  // namespace mueye

#endif  // MUEYE_ORBIT_CAMERA_HH_
