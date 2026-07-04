/**
 * @file   PngWriter.cc
 *
 * @brief  PNG output via the vendored stb_image_write (third_party/).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "io/PngWriter.hh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace mueye {

bool write_png(const std::string &path, const Framebuffer &fb) {
  if (fb.width <= 0 || fb.height <= 0 ||
      fb.rgba.size() < static_cast<std::size_t>(fb.width) * fb.height * 4)
    return false;
  return stbi_write_png(path.c_str(), fb.width, fb.height, 4, fb.rgba.data(),
                        fb.width * 4) != 0;
}

}  // namespace mueye
