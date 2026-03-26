#include "GPUFixPipeline.hpp"
#include <iostream>

GPUFixPipeline::GPUFixPipeline() = default;

// ---------------------------------------------------------------------------
bool GPUFixPipeline::init() {
    m_device = std::make_unique<MetalDevice>();
    if (!m_device->getDevice()) {
        std::cerr << "[GPUFixPipeline] No Metal device – cannot initialise.\n";
        return false;
    }

    m_detector  = std::make_unique<GPUGlitchDetector>(*m_device);
    m_patcher   = std::make_unique<ShaderPatchManager>(*m_device);
    m_validator = std::make_unique<FrameBufferValidator>(*m_device);
    m_sanitizer = std::make_unique<CommandBufferSanitizer>(*m_device);

    m_device->logGPUInfo();
    m_patcher->loadDefaultPatches();

    std::cout << "[GPUFixPipeline] Initialised with "
              << m_patcher->patchCount() << " default patches.\n";
    return true;
}

// ---------------------------------------------------------------------------
std::vector<GlitchReport> GPUFixPipeline::diagnose() {
    return m_detector->detect();
}

// ---------------------------------------------------------------------------
size_t GPUFixPipeline::applyFixes() {
    // The actual shader recompilation happens lazily when the application
    // calls patchedComputePipeline / patchedRenderPipeline.  Here we just
    // confirm the patch set is loaded.
    size_t count = m_patcher->patchCount();
    std::cout << "[GPUFixPipeline] " << count
              << " shader patches ready to apply.\n";
    return count;
}

// ---------------------------------------------------------------------------
void GPUFixPipeline::captureGoodFrame(id<MTLTexture> renderTarget) {
    m_validator->captureReference(renderTarget);
}

// ---------------------------------------------------------------------------
bool GPUFixPipeline::validateFrame(id<MTLTexture> renderTarget) {
    ++m_framesValidated;
    m_detector->advanceFrame();

    if (!m_validator->hasReference()) {
        // No reference yet – capture the first frame as baseline.
        m_validator->captureReference(renderTarget);
        return true;
    }

    bool ok = m_validator->validate(renderTarget, 0.02f);
    if (!ok) {
        ++m_framesRecovered;
        std::cerr << "[GPUFixPipeline] Frame "
                  << m_detector->currentFrameIndex()
                  << " failed validation – blitting last-known-good frame.\n";
        m_validator->blitReference(renderTarget);
    }
    return ok;
}

// ---------------------------------------------------------------------------
bool GPUFixPipeline::submitWork(
        const CommandBufferSanitizer::EncodeBlock& work) {
    return m_sanitizer->submit(work);
}

// ---------------------------------------------------------------------------
void GPUFixPipeline::printReport() const {
    std::cout << "\n===== GPU Fix Pipeline Report =====\n";
    std::cout << "Frames validated : " << m_framesValidated << "\n";
    std::cout << "Frames recovered : " << m_framesRecovered << "\n";
    std::cout << "Cmd submissions  : " << m_sanitizer->totalSubmissions() << "\n";
    std::cout << "Cmd retries      : " << m_sanitizer->totalRetries() << "\n";
    std::cout << "Cmd failures     : " << m_sanitizer->totalFailures() << "\n";
    std::cout << "Shader patches   : " << m_patcher->patchCount() << "\n";

    const auto& reports = m_detector->lastReports();
    if (!reports.empty()) {
        std::cout << "Last glitch reports:\n";
        for (const auto& r : reports) {
            std::cout << "  - " << r.description
                      << "  (severity=" << r.severity
                      << ", frame=" << r.affectedFrameIndex << ")\n";
        }
    }
    std::cout << "===================================\n\n";
}
