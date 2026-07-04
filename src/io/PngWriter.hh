/**
 * @file   PngWriter.hh
 *
 * @brief  Save a rendered Framebuffer as a PNG file.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_IO_PNG_WRITER_HH_
#define MUEYE_IO_PNG_WRITER_HH_

#include <string>

#include "render/Renderer.hh"

namespace mueye {

/** Write @p fb (RGBA8, top-left origin) to @p path as a PNG.
 *  @returns true on success. */
bool write_png(const std::string &path, const Framebuffer &fb);

}  // namespace mueye

#endif  // MUEYE_IO_PNG_WRITER_HH_
