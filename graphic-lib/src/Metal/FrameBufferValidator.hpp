#pragma once
#include "Device.hpp"
#include <vector>
#include <cstdint>
#include <functional>

// Validates frame-buffer contents between passes to catch transient
// rendering corruption on Apple TBDR GPUs.
class FrameBufferValidator {
public:
    explicit FrameBufferValidator(MetalDevice& device);
    ~FrameBufferValidator() = default;

    // Capture a reference snapshot of the current render target.
    // Call once at the start of a known-good frame.
    void captureReference(id<MTLTexture> renderTarget);

    // Compare |current| against the reference and return the fraction of
    // pixels that differ beyond |threshold| (per-component, 0-255 scale).
    float compare(id<MTLTexture> current, uint8_t threshold = 8);

    // Convenience: returns true if the frame looks "healthy" (< maxDrift %).
    bool validate(id<MTLTexture> current, float maxDrift = 0.01f);

    // Fix: if a frame fails validation, blit the last-known-good reference
    // back into |destination| to suppress the flicker.
    void blitReference(id<MTLTexture> destination);

    bool hasReference() const { return !m_reference.empty(); }

private:
    MetalDevice&          m_device;
    std::vector<uint8_t>  m_reference;
    uint32_t              m_refWidth  = 0;
    uint32_t              m_refHeight = 0;
};
