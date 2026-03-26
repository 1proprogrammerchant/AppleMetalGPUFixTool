#include "Device.hpp"
#import <Foundation/Foundation.h>
#include <iostream>

MetalDevice::MetalDevice() {
    m_device = MTLCreateSystemDefaultDevice();
    if (!m_device) {
        std::cerr << "[MetalDevice] ERROR: No Metal-capable GPU found.\n";
        return;
    }
    m_queue = [m_device newCommandQueue];
    if (!m_queue) {
        std::cerr << "[MetalDevice] ERROR: Failed to create command queue.\n";
    }
}

MetalDevice::~MetalDevice() {
    m_queue = nil;
    m_device = nil;
}

void MetalDevice::logGPUInfo() const {
    if (!m_device) {
        std::cerr << "[MetalDevice] No device available.\n";
        return;
    }
    NSString* name = [m_device name];
    std::cout << "[MetalDevice] GPU: " << [name UTF8String] << "\n";
    std::cout << "[MetalDevice] Max threads per threadgroup: "
              << [m_device maxThreadsPerThreadgroup].width << "\n";
    std::cout << "[MetalDevice] Recommended max working set size: "
              << [m_device recommendedMaxWorkingSetSize] / (1024 * 1024) << " MB\n";
}

bool MetalDevice::runComputeKernel(const std::string& kernelName, size_t threadCount) {
    if (!m_device || !m_queue) return false;

    NSError* error = nil;
    id<MTLLibrary> library = [m_device newDefaultLibrary];
    if (!library) {
        std::cerr << "[MetalDevice] Failed to load default Metal library.\n";
        return false;
    }

    NSString* funcName = [NSString stringWithUTF8String:kernelName.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (!function) {
        std::cerr << "[MetalDevice] Kernel '" << kernelName << "' not found.\n";
        return false;
    }

    id<MTLComputePipelineState> pipeline =
        [m_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        std::cerr << "[MetalDevice] Pipeline creation failed: "
                  << [[error localizedDescription] UTF8String] << "\n";
        return false;
    }

    id<MTLCommandBuffer> cmdBuf = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    MTLSize gridSize = MTLSizeMake(threadCount, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, threadCount);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (cmdBuf.error) {
        std::cerr << "[MetalDevice] Kernel execution error: "
                  << [[cmdBuf.error localizedDescription] UTF8String] << "\n";
        return false;
    }
    return true;
}