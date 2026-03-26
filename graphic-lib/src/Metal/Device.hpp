#pragma once
#include <Metal/Metal.h>
#include <string>

class MetalDevice {
public:
    MetalDevice();
    ~MetalDevice();

    id<MTLDevice> getDevice() const { return m_device; }
    id<MTLCommandQueue> getQueue() const { return m_queue; }

    void logGPUInfo() const;
    bool runComputeKernel(const std::string& kernelName, size_t threadCount);

private:
    id<MTLDevice> m_device = nullptr;
    id<MTLCommandQueue> m_queue = nullptr;
};