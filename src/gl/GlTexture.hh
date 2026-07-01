/**
 * @file   GlTexture.hh
 *
 * @brief  Minimal RGBA8 OpenGL texture wrapper for displaying the rendered
 *         framebuffer via ImGui::Image.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_GL_TEXTURE_HH_
#define MUEYE_GL_TEXTURE_HH_

#include <cstdint>

#include "render/Renderer.hh"

namespace mueye {

class GlTexture {
 public:
  GlTexture() = default;
  ~GlTexture();

  GlTexture(const GlTexture &) = delete;
  GlTexture &operator=(const GlTexture &) = delete;

  /** Upload the framebuffer, (re)allocating the texture if the size changed. */
  void upload(const Framebuffer &fb);

  /** Ensure a GL_TEXTURE_2D of size w x h exists (allocating RGBA8 storage,
   *  without uploading pixels) and return its GL name. Used as the destination
   *  for a backend's zero-copy Renderer::render_to_gl path. */
  unsigned int ensure(int w, int h);

  /** Texture id suitable for ImGui::Image (cast to ImTextureID). */
  std::uintptr_t id() const { return static_cast<std::uintptr_t>(tex_); }

 private:
  unsigned int tex_{0};
  int width_{0};
  int height_{0};
};

}  // namespace mueye

#endif  // MUEYE_GL_TEXTURE_HH_
