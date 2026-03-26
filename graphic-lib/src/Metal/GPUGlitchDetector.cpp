#include "GPUGlitchDetector.hpp"
#import <Foundation/Foundation.h>
#include <cmath>
#include <iostream>
#include <limits>

GPUGlitchDetector::GPUGlitchDetector(MetalDevice& device)
    : m_device(device) {}

// ---------------------------------------------------------------------------
// Full diagnostic pass
// ---------------------------------------------------------------------------
std::vector<GlitchReport> GPUGlitchDetector::detect() {
    m_reports.clear();
    checkCommandBufferHealth();
    // Additional texture / buffer checks are dispatched by the caller with
    // specific resources; detect() runs the stateless checks only.
    return m_reports;
}

// ---------------------------------------------------------------------------
// Texture integrity – read back and scan for NaN / Inf pixels
// ---------------------------------------------------------------------------
bool GPUGlitchDetector::checkTextureIntegrity(id<MTLTexture> texture) {
    if (!texture) return false;

    NSUInteger width  = texture.width;
    NSUInteger height = texture.height;
    NSUInteger bytesPerRow = width * 4 * sizeof(float);   // assume RGBA32Float
    std::vector<float> pixels(width * height * 4);

    [texture getBytes:pixels.data()
          bytesPerRow:bytesPerRow
           fromRegion:MTLRegionMake2D(0, 0, width, height)
          mipmapLevel:0];

    size_t badPixels = 0;
    for (size_t i = 0; i < pixels.size(); ++i) {
        if (std::isnan(pixels[i]) || std::isinf(pixels[i])) {
            ++badPixels;
        }
    }

    if (badPixels > 0) {
        float severity = static_cast<float>(badPixels) /
                         static_cast<float>(pixels.size());
        addReport(GlitchReport::Type::TextureCorruption,
                  "NaN/Inf pixels detected (" + std::to_string(badPixels) + " components)",
                  severity);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Depth buffer – look for sudden depth discontinuities (z-fighting)
// ---------------------------------------------------------------------------
bool GPUGlitchDetector::checkDepthBuffer(id<MTLTexture> depthTexture) {
    if (!depthTexture) return false;

    NSUInteger width  = depthTexture.width;
    NSUInteger height = depthTexture.height;
    NSUInteger bytesPerRow = width * sizeof(float);
    std::vector<float> depth(width * height);

    [depthTexture getBytes:depth.data()
               bytesPerRow:bytesPerRow
                fromRegion:MTLRegionMake2D(0, 0, width, height)
               mipmapLevel:0];

    size_t artifacts = 0;
    const float kThreshold = 0.95f; // extreme depth jump between neighbours
    for (NSUInteger y = 0; y < height; ++y) {
        for (NSUInteger x = 1; x < width; ++x) {
            float d0 = depth[y * width + (x - 1)];
            float d1 = depth[y * width + x];
            if (std::fabs(d1 - d0) > kThreshold) ++artifacts;
        }
    }

    if (artifacts > width) {    // more than a full row of sharp edges
        float severity = std::min(1.0f,
            static_cast<float>(artifacts) / static_cast<float>(width * height));
        addReport(GlitchReport::Type::DepthBufferArtifact,
                  "Depth discontinuity artefacts (" + std::to_string(artifacts) + ")",
                  severity);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Vertex buffer – detect "vertex explosion" (outlier positions)
// ---------------------------------------------------------------------------
bool GPUGlitchDetector::checkVertexBuffer(id<MTLBuffer> vertexBuffer,
                                          size_t vertexCount,
                                          size_t stride) {
    if (!vertexBuffer || vertexCount == 0 || stride == 0) return false;

    const uint8_t* base = static_cast<const uint8_t*>([vertexBuffer contents]);
    size_t outliers = 0;
    const float kBound = 1e6f;

    for (size_t i = 0; i < vertexCount; ++i) {
        const float* pos = reinterpret_cast<const float*>(base + i * stride);
        for (int c = 0; c < 3; ++c) {
            if (std::fabs(pos[c]) > kBound ||
                std::isnan(pos[c]) ||
                std::isinf(pos[c])) {
                ++outliers;
                break;
            }
        }
    }

    if (outliers > 0) {
        float severity = static_cast<float>(outliers) /
                         static_cast<float>(vertexCount);
        addReport(GlitchReport::Type::VertexExplosion,
                  "Outlier vertices: " + std::to_string(outliers) + "/" +
                  std::to_string(vertexCount),
                  severity);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Tile gaps – scan render-target for fully-black rectangular regions that
// indicate an Apple GPU tile was not flushed.
// ---------------------------------------------------------------------------
bool GPUGlitchDetector::checkForTileGaps(id<MTLTexture> renderTarget) {
    if (!renderTarget) return false;

    NSUInteger width  = renderTarget.width;
    NSUInteger height = renderTarget.height;
    NSUInteger bytesPerRow = width * 4; // assume BGRA8Unorm
    std::vector<uint8_t> pixels(width * height * 4);

    [renderTarget getBytes:pixels.data()
               bytesPerRow:bytesPerRow
                fromRegion:MTLRegionMake2D(0, 0, width, height)
               mipmapLevel:0];

    // Scan in 32x32 tile increments (Apple GPU tile size)
    const NSUInteger tileSize = 32;
    size_t emptyTiles = 0;
    size_t totalTiles = 0;

    for (NSUInteger ty = 0; ty + tileSize <= height; ty += tileSize) {
        for (NSUInteger tx = 0; tx + tileSize <= width; tx += tileSize) {
            ++totalTiles;
            bool allBlack = true;
            for (NSUInteger row = ty; row < ty + tileSize && allBlack; ++row) {
                for (NSUInteger col = tx; col < tx + tileSize && allBlack; ++col) {
                    size_t idx = (row * width + col) * 4;
                    if (pixels[idx] != 0 || pixels[idx+1] != 0 ||
                        pixels[idx+2] != 0 || pixels[idx+3] != 0) {
                        allBlack = false;
                    }
                }
            }
            if (allBlack) ++emptyTiles;
        }
    }

    // A few black tiles may be legitimate; flag if >5 % are black
    if (totalTiles > 0 && emptyTiles * 20 > totalTiles) {
        float severity = static_cast<float>(emptyTiles) /
                         static_cast<float>(totalTiles);
        addReport(GlitchReport::Type::TileRenderingGap,
                  "Empty tiles: " + std::to_string(emptyTiles) + "/" +
                  std::to_string(totalTiles),
                  severity);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Command buffer health – submit a no-op and check for GPU timeout
// ---------------------------------------------------------------------------
bool GPUGlitchDetector::checkCommandBufferHealth() {
    id<MTLCommandQueue> queue = m_device.getQueue();
    if (!queue) return false;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    [blit endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (cmdBuf.status == MTLCommandBufferStatusError) {
        addReport(GlitchReport::Type::CommandBufferTimeout,
                  "Command buffer error: " +
                  std::string([[cmdBuf.error localizedDescription] UTF8String]),
                  1.0f);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
void GPUGlitchDetector::addReport(GlitchReport::Type type,
                                  const std::string& desc,
                                  float severity) {
    m_reports.push_back({type, desc, m_frameIndex, severity});
    std::cout << "[GlitchDetector] " << desc
              << "  (severity=" << severity << ", frame=" << m_frameIndex << ")\n";
}
