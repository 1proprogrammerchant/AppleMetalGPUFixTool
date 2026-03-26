#pragma once
#include "Device.hpp"
#include "GPUGlitchDetector.hpp"
#include "ShaderPatchManager.hpp"
#include "FrameBufferValidator.hpp"
#include "CommandBufferSanitizer.hpp"
#include <memory>
#include <string>

// Top-level orchestrator that ties together detection, shader patching,
// frame validation, and command-buffer sanitisation into a single
// "fix pipeline" that can be dropped into any Metal application.
class GPUFixPipeline {
public:
    GPUFixPipeline();
    ~GPUFixPipeline() = default;

    // Initialise – returns false if no Metal device is available.
    bool init();

    // Run the full diagnostic suite (no fixes applied yet).
    std::vector<GlitchReport> diagnose();

    // Apply all available fixes and return the number of patches applied.
    size_t applyFixes();

    // Per-frame hook: call once per frame to validate the render target,
    // recover from hangs, and log telemetry.
    bool validateFrame(id<MTLTexture> renderTarget);

    // Capture a known-good reference frame for flicker suppression.
    void captureGoodFrame(id<MTLTexture> renderTarget);

    // Submit GPU work through the sanitizer (auto-retry on failure).
    bool submitWork(const CommandBufferSanitizer::EncodeBlock& work);

    // Access sub-components
    MetalDevice&            device()    { return *m_device; }
    GPUGlitchDetector&      detector()  { return *m_detector; }
    ShaderPatchManager&     patcher()   { return *m_patcher; }
    FrameBufferValidator&   validator() { return *m_validator; }
    CommandBufferSanitizer& sanitizer() { return *m_sanitizer; }

    void printReport() const;

private:
    std::unique_ptr<MetalDevice>            m_device;
    std::unique_ptr<GPUGlitchDetector>      m_detector;
    std::unique_ptr<ShaderPatchManager>     m_patcher;
    std::unique_ptr<FrameBufferValidator>   m_validator;
    std::unique_ptr<CommandBufferSanitizer> m_sanitizer;

    uint64_t m_framesValidated = 0;
    uint64_t m_framesRecovered = 0;
};
