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

  /** Reset the view and centre the target on a volume box of the given extents
   *  (see RenderParams::box) — so 2D/planar and non-cubic grids frame nicely. */
  void frame_box(const Vec3 &box);

  /** Re-centre the target and viewing distance on a box of the given extents
   *  *without* resetting the orbit angles — used when periodic tiling grows or
   *  shrinks the rendered box under an existing view. */
  void retarget_box(const Vec3 &box);

  /** Frame an arbitrary axis-aligned bounding box given its centre and largest
   *  side length — used for sheared (Bravais) cells, whose bounds are not
   *  simply [0,box]. Resets the orbit angles. */
  void frame_aabb(const Vec3 &center, float extent);
  /** Like frame_aabb but keeps the current orbit angles. */
  void retarget_aabb(const Vec3 &center, float extent);

  /** Set the orbit angles directly (radians; pitch is clamped like orbit()). */
  void set_view_angles(float yaw, float pitch);

  /** Look straight down the z axis at the x-y plane (image right = +x, image
   *  up = +y): the natural view of a 2D field, which the default oblique
   *  angles would show as a tilted slab. */
  void set_face_on() { set_view_angles(1.5707963f, 0.0f); }

  /** Orbit by mouse drag deltas (in radians-equivalent screen units). */
  void orbit(float dyaw, float dpitch);
  /** Dolly in/out (e.g. mouse wheel); positive zooms in. */
  void zoom(float delta);
  /** Pan the target in the camera plane by a drag of (@p dx, @p dy) viewport
   *  widths/heights (screen coordinates: +dy is downwards); the scene follows
   *  the cursor 1:1 for a viewport of the given aspect ratio. */
  void pan(float dx, float dy, float aspect);

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
