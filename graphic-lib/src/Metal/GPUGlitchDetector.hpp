#pragma once
#include "Device.hpp"
#include <vector>
#include <string>
#include <cstdint>

// Describes a single detected GPU glitch/anomaly
struct GlitchReport {
    enum class Type : uint8_t {
        TextureCorruption,      // Garbled or NaN pixels in render targets
        VertexExplosion,        // Vertices at extreme coordinates
        FlickerDetected,        // Alternating frame content mismatch
        DepthBufferArtifact,    // Depth discontinuities / z-fighting
        TileRenderingGap,       // Missing tile regions on Apple GPU tile-based renderer
        CommandBufferTimeout,   // GPU hang / watchdog timeout
        ShaderMiscompile        // Known bad shader instruction pattern
    };

    Type        type;
    std::string description;
    uint64_t    affectedFrameIndex;
    float       severity;           // 0.0 – 1.0
};

// Scans Metal state for common Apple-GPU visual glitches (2026 family)
class GPUGlitchDetector {
public:
    explicit GPUGlitchDetector(MetalDevice& device);
    ~GPUGlitchDetector() = default;

    // Run all diagnostic checks and return any findings
    std::vector<GlitchReport> detect();

    // Individual detectors
    bool checkTextureIntegrity(id<MTLTexture> texture);
    bool checkDepthBuffer(id<MTLTexture> depthTexture);
    bool checkVertexBuffer(id<MTLBuffer> vertexBuffer, size_t vertexCount, size_t stride);
    bool checkForTileGaps(id<MTLTexture> renderTarget);
    bool checkCommandBufferHealth();

    uint64_t currentFrameIndex() const { return m_frameIndex; }
    void     advanceFrame() { ++m_frameIndex; }

    const std::vector<GlitchReport>& lastReports() const { return m_reports; }

private:
    MetalDevice&              m_device;
    uint64_t                  m_frameIndex = 0;
    std::vector<GlitchReport> m_reports;

    void addReport(GlitchReport::Type type, const std::string& desc, float severity);
};
