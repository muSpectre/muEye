/**
 * @file   MetalRenderer.mm
 *
 * @brief  Metal compute-shader implementation of the ray tracer.
 *
 * The Metal Shading Language kernel below mirrors render_core.hh's trace_ray()
 * (DVR + isosurface). It is compiled at runtime via newLibraryWithSource: so no
 * offline `metal` toolchain / .metallib build step is required.
 *
 * The parameter struct `GpuParams` is laid out using only 4-byte scalars (float
 * and int, no vector types) so its memory layout is identical between this
 * Objective-C++ translation unit and the MSL source string.
 *
 * Part of muEye, a viewer for muGrid data.
 */

#include "render/MetalRenderer.hh"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <vector>

namespace mueye {

namespace {

// Must match `struct P` in the MSL source below, field-for-field.
struct GpuParams {
  float eye_x, eye_y, eye_z;
  float fwd_x, fwd_y, fwd_z;
  float right_x, right_y, right_z;
  float up_x, up_y, up_z;
  float tan_half_fov;
  float aspect;
  int nx, ny, nz;
  float step;
  float data_min, data_max;
  int lut_size;
  float density_scale;
  float iso_value;
  int mode;  // 0 = DVR, 1 = isosurface
  float bg_x, bg_y, bg_z;
  int width, height;
};

// The compute kernel. Kept close to render_core.hh so CPU and Metal match.
const char *kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct P {
    float eye_x, eye_y, eye_z;
    float fwd_x, fwd_y, fwd_z;
    float right_x, right_y, right_z;
    float up_x, up_y, up_z;
    float tan_half_fov;
    float aspect;
    int nx, ny, nz;
    float step;
    float data_min, data_max;
    int lut_size;
    float density_scale;
    float iso_value;
    int mode;
    float bg_x, bg_y, bg_z;
    int width, height;
};

// The volume lives in a 3-D texture; the texture unit does the trilinear
// filtering. Normalized coords + linear filter + clamp-to-edge reproduce
// render_core.hh's ArraySampler (voxel centre at (i+0.5)/dim, clamped).
constexpr sampler volSampler(coord::normalized, address::clamp_to_edge,
                             filter::linear);

static float sample(texture3d<float> vol, float3 uvw) {
    return vol.sample(volSampler, uvw).r;
}

static float3 gradient(texture3d<float> vol, constant P& p, float3 uvw) {
    float hx = 1.0/p.nx, hy = 1.0/p.ny, hz = 1.0/p.nz;
    float gx = sample(vol,float3(uvw.x+hx,uvw.y,uvw.z)) - sample(vol,float3(uvw.x-hx,uvw.y,uvw.z));
    float gy = sample(vol,float3(uvw.x,uvw.y+hy,uvw.z)) - sample(vol,float3(uvw.x,uvw.y-hy,uvw.z));
    float gz = sample(vol,float3(uvw.x,uvw.y,uvw.z+hz)) - sample(vol,float3(uvw.x,uvw.y,uvw.z-hz));
    return float3(gx, gy, gz);
}

static float normalize_value(constant P& p, float v) {
    float range = p.data_max - p.data_min;
    if (range <= 0.0) return 0.0;
    return clamp((v - p.data_min) / range, 0.0, 1.0);
}

static bool intersect_unit_box(float3 o, float3 d, thread float& t_near, thread float& t_far) {
    float tmin = -1e30, tmax = 1e30;
    for (int axis = 0; axis < 3; ++axis) {
        float oa = o[axis], da = d[axis];
        if (fabs(da) < 1e-8) {
            if (oa < 0.0 || oa > 1.0) return false;
        } else {
            float inv = 1.0 / da;
            float t1 = (0.0 - oa) * inv;
            float t2 = (1.0 - oa) * inv;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tmin = max(tmin, t1);
            tmax = min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    t_near = max(tmin, 0.0);
    t_far = tmax;
    return t_far > t_near;
}

// Primary (pinhole) ray direction through normalized image coordinate (u,v).
static float3 primary_dir(constant P& p, float u, float v) {
    float3 fwd = float3(p.fwd_x, p.fwd_y, p.fwd_z);
    float3 rt  = float3(p.right_x, p.right_y, p.right_z);
    float3 up  = float3(p.up_x, p.up_y, p.up_z);
    float sx = (2.0*u - 1.0) * p.aspect * p.tan_half_fov;
    float sy = (1.0 - 2.0*v) * p.tan_half_fov;
    return normalize(fwd + rt*sx + up*sy);
}

static void store_pixel(device uchar4* out, constant P& p, uint2 gid, float3 col) {
    col = clamp(col, 0.0, 1.0);
    out[gid.y * uint(p.width) + gid.x] =
        uchar4(uchar(col.x*255.0+0.5), uchar(col.y*255.0+0.5),
               uchar(col.z*255.0+0.5), 255);
}

// DVR and isosurface are separate kernels (rather than one with a mode branch)
// so each compiles a single path: p.mode is grid-uniform so the branch never
// diverges, but a combined kernel inflates register/instruction footprint and
// can cap occupancy. The host binds the matching pipeline (see render()).
kernel void raymarch_dvr(texture3d<float>     vol [[texture(0)]],
                         device const float4* lut [[buffer(1)]],
                         constant P&          p   [[buffer(2)]],
                         device uchar4*       out [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= p.width || int(gid.y) >= p.height) return;
    float3 eye = float3(p.eye_x, p.eye_y, p.eye_z);
    float3 bg  = float3(p.bg_x, p.bg_y, p.bg_z);
    float u = (float(gid.x) + 0.5) / p.width;
    float v = (float(gid.y) + 0.5) / p.height;
    float3 dir = primary_dir(p, u, v);

    float3 col = bg;
    float t_near, t_far;
    if (intersect_unit_box(eye, dir, t_near, t_far)) {
        float3 accum = float3(0.0);
        float trans = 1.0;
        for (float t = t_near; t < t_far; t += p.step) {
            float3 pos = eye + dir*t;
            float val = sample(vol, pos);
            float nv = normalize_value(p, val);
            float f = nv * (p.lut_size - 1);
            int i0 = clamp(int(f), 0, p.lut_size - 1);
            int i1 = min(i0 + 1, p.lut_size - 1);
            float4 c = mix(lut[i0], lut[i1], f - i0);
            float alpha = clamp(c.w * p.density_scale * p.step * p.nx, 0.0, 1.0);
            accum += c.rgb * (alpha * trans);
            trans *= (1.0 - alpha);
            if (trans < 0.003) break;
        }
        col = accum + bg * trans;
    }
    store_pixel(out, p, gid, col);
}

kernel void raymarch_iso(texture3d<float>     vol [[texture(0)]],
                         device const float4* lut [[buffer(1)]],
                         constant P&          p   [[buffer(2)]],
                         device uchar4*       out [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= p.width || int(gid.y) >= p.height) return;
    float3 eye = float3(p.eye_x, p.eye_y, p.eye_z);
    float3 bg  = float3(p.bg_x, p.bg_y, p.bg_z);
    float u = (float(gid.x) + 0.5) / p.width;
    float v = (float(gid.y) + 0.5) / p.height;
    float3 dir = primary_dir(p, u, v);

    float3 col = bg;
    float t_near, t_far;
    if (intersect_unit_box(eye, dir, t_near, t_far)) {
        float t = t_near;
        float3 prev = eye + dir*t;
        float prev_v = sample(vol, prev) - p.iso_value;
        t += p.step;
        while (t < t_far) {
            float3 pos = eye + dir*t;
            float cur_v = sample(vol, pos) - p.iso_value;
            if (prev_v * cur_v <= 0.0) {
                float denom = cur_v - prev_v;
                float frac = fabs(denom) > 1e-12 ? prev_v / -denom : 0.0;
                float3 hitp = prev + (pos - prev)*frac;
                float3 nrm = normalize(gradient(vol, p, hitp));
                float3 l = normalize(eye - hitp);
                float diff = fabs(dot(nrm, l));
                float3 base = float3(0.82, 0.45, 0.20);
                col = base * (0.20 + 0.80*diff);
                break;
            }
            prev = pos; prev_v = cur_v; t += p.step;
        }
    }
    store_pixel(out, p, gid, col);
}
)METAL";

}  // namespace

struct MetalRenderer::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline_dvr = nil;
  id<MTLComputePipelineState> pipeline_iso = nil;

  id<MTLTexture> volume = nil;
  id<MTLBuffer> lut = nil;
  id<MTLBuffer> output = nil;
  int out_w = 0, out_h = 0;
  int nx = 0, ny = 0, nz = 0;

  bool build() {
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) return false;
    queue = [device newCommandQueue];
    if (queue == nil) return false;

    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:kShaderSource];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    if (lib == nil) {
      NSLog(@"muEye Metal: shader compile failed: %@", err);
      return false;
    }
    id<MTLFunction> fn_dvr = [lib newFunctionWithName:@"raymarch_dvr"];
    id<MTLFunction> fn_iso = [lib newFunctionWithName:@"raymarch_iso"];
    if (fn_dvr == nil || fn_iso == nil) return false;
    pipeline_dvr = [device newComputePipelineStateWithFunction:fn_dvr error:&err];
    if (pipeline_dvr == nil) {
      NSLog(@"muEye Metal: DVR pipeline creation failed: %@", err);
      return false;
    }
    pipeline_iso = [device newComputePipelineStateWithFunction:fn_iso error:&err];
    if (pipeline_iso == nil) {
      NSLog(@"muEye Metal: isosurface pipeline creation failed: %@", err);
      return false;
    }
    return true;
  }
};

bool MetalRenderer::available() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil;
  }
}

MetalRenderer::MetalRenderer() : impl_(new Impl()) {
  @autoreleasepool {
    if (!impl_->build()) {
      // Leave pipeline nil; ok() reports failure and the factory skips us.
    }
  }
}

MetalRenderer::~MetalRenderer() { delete impl_; }

bool MetalRenderer::ok() const {
  return impl_ && impl_->pipeline_dvr != nil && impl_->pipeline_iso != nil;
}

const char *MetalRenderer::name() const { return "Metal (GPU)"; }

void MetalRenderer::set_volume(const float *data, int nx, int ny, int nz) {
  if (!ok() || data == nullptr) return;
  @autoreleasepool {
    // A 3-D R32Float texture so the shader gets hardware trilinear filtering.
    // (Apple-family GPUs support linear filtering of 32-bit float textures.)
    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = MTLPixelFormatR32Float;
    desc.width = static_cast<NSUInteger>(nx);
    desc.height = static_cast<NSUInteger>(ny);
    desc.depth = static_cast<NSUInteger>(nz);
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    impl_->volume = [impl_->device newTextureWithDescriptor:desc];

    // Volume is column-major (idx = i + nx*(j + ny*k)), i.e. x fastest, then y
    // rows, then z slices — matching the texture's (width=x, height=y, depth=z)
    // row/image strides.
    MTLRegion region = MTLRegionMake3D(0, 0, 0, static_cast<NSUInteger>(nx),
                                       static_cast<NSUInteger>(ny),
                                       static_cast<NSUInteger>(nz));
    [impl_->volume replaceRegion:region
                     mipmapLevel:0
                           slice:0
                       withBytes:data
                     bytesPerRow:static_cast<NSUInteger>(nx) * sizeof(float)
                   bytesPerImage:static_cast<NSUInteger>(nx) * ny * sizeof(float)];
    impl_->nx = nx;
    impl_->ny = ny;
    impl_->nz = nz;
  }
}

void MetalRenderer::set_transfer_function(const Vec4 *lut, int n) {
  if (!ok() || lut == nullptr) return;
  @autoreleasepool {
    // Vec4 {x,y,z,w} matches MSL float4 layout (16 bytes).
    impl_->lut = [impl_->device newBufferWithBytes:lut
                                            length:static_cast<std::size_t>(n) * sizeof(Vec4)
                                           options:MTLResourceStorageModeShared];
  }
}

void MetalRenderer::render(const RenderParams &params, const Camera &camera,
                           Framebuffer &fb) {
  if (!ok() || impl_->volume == nil || impl_->lut == nil) return;
  const int w = fb.width, h = fb.height;
  if (w <= 0 || h <= 0) return;

  @autoreleasepool {
    // (Re)allocate the output buffer if the size changed.
    if (impl_->output == nil || impl_->out_w != w || impl_->out_h != h) {
      impl_->output = [impl_->device
          newBufferWithLength:static_cast<std::size_t>(w) * h * 4
                      options:MTLResourceStorageModeShared];
      impl_->out_w = w;
      impl_->out_h = h;
    }

    GpuParams p{};
    p.eye_x = camera.eye.x; p.eye_y = camera.eye.y; p.eye_z = camera.eye.z;
    p.fwd_x = camera.forward.x; p.fwd_y = camera.forward.y; p.fwd_z = camera.forward.z;
    p.right_x = camera.right.x; p.right_y = camera.right.y; p.right_z = camera.right.z;
    p.up_x = camera.up.x; p.up_y = camera.up.y; p.up_z = camera.up.z;
    p.tan_half_fov = camera.tan_half_fov;
    p.aspect = camera.aspect;
    p.nx = params.nx; p.ny = params.ny; p.nz = params.nz;
    p.step = params.step;
    p.data_min = params.data_min; p.data_max = params.data_max;
    p.lut_size = params.lut_size;
    p.density_scale = params.density_scale;
    p.iso_value = params.iso_value;
    p.mode = (params.mode == RenderMode::Isosurface) ? 1 : 0;
    p.bg_x = params.bg.x; p.bg_y = params.bg.y; p.bg_z = params.bg.z;
    p.width = w; p.height = h;

    id<MTLComputePipelineState> pipeline =
        (params.mode == RenderMode::Isosurface) ? impl_->pipeline_iso
                                                : impl_->pipeline_dvr;

    id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setTexture:impl_->volume atIndex:0];
    [enc setBuffer:impl_->lut offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(GpuParams) atIndex:2];
    [enc setBuffer:impl_->output offset:0 atIndex:3];

    MTLSize tg = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((w + 15) / 16, (h + 15) / 16, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Unified memory: copy straight out of the shared buffer.
    std::memcpy(fb.rgba.data(), [impl_->output contents],
                static_cast<std::size_t>(w) * h * 4);
  }
}

}  // namespace mueye
