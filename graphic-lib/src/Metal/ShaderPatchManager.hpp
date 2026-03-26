#pragma once
#include "Device.hpp"
#include "GPUGlitchDetector.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// Represents a patch that can be applied to a Metal shader function
struct ShaderPatch {
    std::string functionName;
    std::string description;
    // MSL source snippet injected right after the function signature
    std::string mslPreambleInjection;
    // MSL source snippet appended before the closing brace
    std::string mslEpilogueInjection;
    bool        applied = false;
};

// Manages runtime shader patches to work around known Apple GPU miscompiles
// and tile-based-deferred-rendering (TBDR) edge cases on 2026 hardware.
class ShaderPatchManager {
public:
    explicit ShaderPatchManager(MetalDevice& device);
    ~ShaderPatchManager() = default;

    // Register a well-known patch
    void registerPatch(const ShaderPatch& patch);

    // Apply all registered patches that target |library| and return a
    // freshly compiled library with the fixes.  Returns nil on failure.
    id<MTLLibrary> applyPatches(const std::string& originalMSL,
                                NSError** outError);

    // Convenience: compile & replace a specific function in the existing
    // pipeline with the patched variant.
    id<MTLComputePipelineState> patchedComputePipeline(
        const std::string& functionName,
        const std::string& originalMSL,
        NSError** outError);

    id<MTLRenderPipelineState> patchedRenderPipeline(
        const std::string& vertexFunc,
        const std::string& fragmentFunc,
        const std::string& originalMSL,
        MTLRenderPipelineDescriptor* descriptor,
        NSError** outError);

    size_t patchCount() const { return m_patches.size(); }
    const std::vector<ShaderPatch>& patches() const { return m_patches; }

    // Load the built-in Apple-GPU-2026 patch set
    void loadDefaultPatches();

private:
    MetalDevice&             m_device;
    std::vector<ShaderPatch> m_patches;

    std::string injectPatches(const std::string& msl) const;
};
