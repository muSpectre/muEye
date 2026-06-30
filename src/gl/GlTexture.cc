/**
 * @file   GlTexture.cc
 *
 * @brief  RGBA8 texture upload. Uses only OpenGL 1.x texture entry points
 *         (glGenTextures/glTexImage2D/...), which are available from the
 *         platform GL header without a function loader.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "gl/GlTexture.hh"

#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

namespace mueye {

GlTexture::~GlTexture() {
  if (tex_ != 0) {
    GLuint t = tex_;
    glDeleteTextures(1, &t);
  }
}

void GlTexture::upload(const Framebuffer &fb) {
  if (fb.width <= 0 || fb.height <= 0) return;

  if (tex_ == 0) {
    GLuint t = 0;
    glGenTextures(1, &t);
    tex_ = t;
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    glBindTexture(GL_TEXTURE_2D, tex_);
  }

  if (fb.width != width_ || fb.height != height_) {
    width_ = fb.width;
    height_ = fb.height;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, fb.rgba.data());
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA,
                    GL_UNSIGNED_BYTE, fb.rgba.data());
  }
}

}  // namespace mueye
