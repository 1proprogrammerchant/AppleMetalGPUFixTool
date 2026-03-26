#include "FrameBufferValidator.hpp"
#import <Foundation/Foundation.h>
#include <cstring>
#include <cmath>
#include <iostream>

FrameBufferValidator::FrameBufferValidator(MetalDevice& device)
    : m_device(device) {}

// ---------------------------------------------------------------------------
void FrameBufferValidator::captureReference(id<MTLTexture> renderTarget) {
    if (!renderTarget) return;

    m_refWidth  = static_cast<uint32_t>(renderTarget.width);
    m_refHeight = static_cast<uint32_t>(renderTarget.height);
    size_t bytesPerRow = m_refWidth * 4; // BGRA8Unorm
    m_reference.resize(m_refWidth * m_refHeight * 4);

    [renderTarget getBytes:m_reference.data()
               bytesPerRow:bytesPerRow
                fromRegion:MTLRegionMake2D(0, 0, m_refWidth, m_refHeight)
               mipmapLevel:0];

    std::cout << "[FrameBufferValidator] Reference captured: "
              << m_refWidth << "x" << m_refHeight << "\n";
}

// ---------------------------------------------------------------------------
float FrameBufferValidator::compare(id<MTLTexture> current, uint8_t threshold) {
    if (!current || m_reference.empty()) return 1.0f;

    uint32_t w = static_cast<uint32_t>(current.width);
    uint32_t h = static_cast<uint32_t>(current.height);
    if (w != m_refWidth || h != m_refHeight) return 1.0f;

    size_t bytesPerRow = w * 4;
    std::vector<uint8_t> pixels(w * h * 4);
    [current getBytes:pixels.data()
          bytesPerRow:bytesPerRow
           fromRegion:MTLRegionMake2D(0, 0, w, h)
          mipmapLevel:0];

    size_t totalPixels  = static_cast<size_t>(w) * h;
    size_t diffPixels   = 0;

    for (size_t i = 0; i < totalPixels; ++i) {
        size_t base = i * 4;
        bool differ = false;
        for (int c = 0; c < 4; ++c) {
            int d = static_cast<int>(pixels[base + c]) -
                    static_cast<int>(m_reference[base + c]);
            if (std::abs(d) > threshold) { differ = true; break; }
        }
        if (differ) ++diffPixels;
    }

    return static_cast<float>(diffPixels) / static_cast<float>(totalPixels);
}

// ---------------------------------------------------------------------------
bool FrameBufferValidator::validate(id<MTLTexture> current, float maxDrift) {
    float drift = compare(current);
    if (drift > maxDrift) {
        std::cerr << "[FrameBufferValidator] Frame drift " << drift * 100.0f
                  << "% exceeds limit " << maxDrift * 100.0f << "%.\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
void FrameBufferValidator::blitReference(id<MTLTexture> destination) {
    if (m_reference.empty() || !destination) return;

    uint32_t w = static_cast<uint32_t>(destination.width);
    uint32_t h = static_cast<uint32_t>(destination.height);
    if (w != m_refWidth || h != m_refHeight) return;

    size_t bytesPerRow = w * 4;
    [destination replaceRegion:MTLRegionMake2D(0, 0, w, h)
                   mipmapLevel:0
                     withBytes:m_reference.data()
                   bytesPerRow:bytesPerRow];

    std::cout << "[FrameBufferValidator] Blitted reference back to render target.\n";
}
