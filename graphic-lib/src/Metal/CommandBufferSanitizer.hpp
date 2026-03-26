#pragma once
#include "Device.hpp"
#include <functional>
#include <string>
#include <vector>

// Wraps Metal command buffer submission with automatic error detection,
// retry logic, and GPU-hang recovery for Apple 2026 GPUs.
class CommandBufferSanitizer {
public:
    explicit CommandBufferSanitizer(MetalDevice& device);
    ~CommandBufferSanitizer() = default;

    // Submit a block that encodes GPU work.  The sanitizer handles commit,
    // wait, and – on failure – automatic retry with reduced load.
    using EncodeBlock = std::function<void(id<MTLCommandBuffer>)>;

    bool submit(const EncodeBlock& encode, int maxRetries = 3);

    // Enable / disable automatic GPU-hang recovery via device reset
    void setAutoRecovery(bool enabled) { m_autoRecover = enabled; }
    bool autoRecovery() const { return m_autoRecover; }

    // Statistics
    uint64_t totalSubmissions() const { return m_submissions; }
    uint64_t totalRetries()     const { return m_retries; }
    uint64_t totalFailures()    const { return m_failures; }

private:
    MetalDevice& m_device;
    bool         m_autoRecover = true;
    uint64_t     m_submissions = 0;
    uint64_t     m_retries     = 0;
    uint64_t     m_failures    = 0;

    bool waitAndCheck(id<MTLCommandBuffer> cmdBuf);
};
