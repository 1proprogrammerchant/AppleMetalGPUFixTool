#include "CommandBufferSanitizer.hpp"
#import <Foundation/Foundation.h>
#include <iostream>
#include <thread>
#include <chrono>

CommandBufferSanitizer::CommandBufferSanitizer(MetalDevice& device)
    : m_device(device) {}

// ---------------------------------------------------------------------------
bool CommandBufferSanitizer::submit(const EncodeBlock& encode, int maxRetries) {
    ++m_submissions;

    for (int attempt = 0; attempt <= maxRetries; ++attempt) {
        id<MTLCommandQueue> queue = m_device.getQueue();
        if (!queue) {
            std::cerr << "[CmdBufSanitizer] No command queue.\n";
            ++m_failures;
            return false;
        }

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        if (!cmdBuf) {
            std::cerr << "[CmdBufSanitizer] Failed to create command buffer.\n";
            ++m_failures;
            return false;
        }

        // Let the caller encode work
        encode(cmdBuf);

        [cmdBuf commit];

        if (waitAndCheck(cmdBuf)) {
            return true;
        }

        // Failure path
        if (attempt < maxRetries) {
            ++m_retries;
            std::cerr << "[CmdBufSanitizer] Attempt " << (attempt + 1)
                      << " failed, retrying (" << (maxRetries - attempt - 1)
                      << " left)...\n";

            // Brief pause to let the GPU recover
            std::this_thread::sleep_for(std::chrono::milliseconds(50 * (attempt + 1)));
        }
    }

    ++m_failures;
    std::cerr << "[CmdBufSanitizer] All " << (maxRetries + 1)
              << " attempts exhausted.\n";
    return false;
}

// ---------------------------------------------------------------------------
bool CommandBufferSanitizer::waitAndCheck(id<MTLCommandBuffer> cmdBuf) {
    [cmdBuf waitUntilCompleted];

    switch (cmdBuf.status) {
        case MTLCommandBufferStatusCompleted:
            return true;

        case MTLCommandBufferStatusError: {
            NSError* err = cmdBuf.error;
            std::cerr << "[CmdBufSanitizer] GPU error: "
                      << [[err localizedDescription] UTF8String] << "\n";

            // Check for specific Apple GPU timeout codes
            if (err.code == MTLCommandBufferErrorTimeout) {
                std::cerr << "[CmdBufSanitizer] GPU watchdog timeout detected "
                             "(common on Apple M4/M5 with heavy compute).\n";
            }
            if (err.code == MTLCommandBufferErrorPageFault) {
                std::cerr << "[CmdBufSanitizer] GPU page fault – possible "
                             "buffer overrun or use-after-free.\n";
            }
            return false;
        }

        default:
            std::cerr << "[CmdBufSanitizer] Unexpected status: "
                      << static_cast<int>(cmdBuf.status) << "\n";
            return false;
    }
}
