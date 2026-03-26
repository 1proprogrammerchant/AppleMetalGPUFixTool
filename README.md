# AppleMetalGPUFixTool

A dual-layer GPU diagnostics and repair toolkit targeting visual rendering defects on Apple Silicon devices shipping with 2026-era Metal drivers -- specifically the M4, M5, and A18 GPU families.

The project provides both a native C++ Metal library (graphic-lib) and a GPU-accelerated Python kernel suite (triton-graphic) that can detect, classify, and correct frame-level artefacts in real time.


## The Problem

Apple's tile-based deferred rendering (TBDR) architecture introduced several regression-class bugs across the 2026 Metal driver stack. These affect both macOS and iOS applications and manifest as visible rendering corruption that cannot be resolved through standard Metal validation or API-level workarounds.

Known defect categories --

- Texture corruption -- NaN and Inf values propagate through render targets due to uninitialised tile memory, producing garbled or flickering pixels in final output
- Vertex explosion -- vertex positions jump to extreme coordinates (typically > 1e6) during specific draw call patterns, causing geometry to disappear or fill the screen
- TBDR tile flush race -- on M4/M5 GPUs, an imageblock store followed by a threadgroup barrier can silently drop writes when the hardware evicts the tile mid-flight, leaving black 32x32 rectangles in the frame
- Half-precision denorm banding -- half-float denormals are flushed to zero inconsistently across SIMD lanes, producing visible colour banding in gradients and lighting
- Depth pre-pass precision mismatch -- the depth pre-pass computes slightly different z-values than the main pass on A18/M5, causing z-fighting artefacts on otherwise co-planar geometry
- Command buffer timeout -- heavy compute workloads trigger GPU watchdog timeouts that crash the application without meaningful error recovery


## Project Structure

    AppleMetalGPUFixTool/
    - CMakeLists.txt                  -- root build, links graphic-lib into executable
    - graphic-lib/
      - CMakeLists.txt                -- static library build (Obj-C++ / Metal)
      - src/Metal/
        - Device.hpp / .cpp           -- Metal device init, GPU info, kernel dispatch
        - GPUGlitchDetector.hpp / .cpp -- detection for all six defect categories
        - ShaderPatchManager.hpp / .cpp -- runtime MSL shader injection and recompile
        - FrameBufferValidator.hpp / .cpp -- per-frame render target diffing and recovery
        - CommandBufferSanitizer.hpp / .cpp -- auto-retry submission with hang recovery
        - GPUFixPipeline.hpp / .cpp   -- top-level orchestrator binding all components
    - triton-graphic/
      - tritongraphicset.py           -- RGBA clamp, NaN flush, denorm floor kernels
      - texture_repair.py             -- neighbour-blend repair for corrupted textures
      - depth_fix.py                  -- depth discontinuity detection and smoothing
      - tile_gap_fill.py              -- TBDR tile gap detection and fill from neighbours
      - flicker_suppress.py           -- frame-to-frame flicker mask and blend suppression
      - vertex_sanitizer.py           -- vertex explosion clamp and NaN/Inf flush
      - pipeline.py                   -- unified TritonGPUFixPipeline class
      - __init__.py                   -- package exports


## graphic-lib -- Native Metal C++ Layer

This is a static library compiled as Objective-C++ that links against Metal, Foundation, and QuartzCore. It operates directly on Metal API objects -- textures, buffers, command queues -- and is designed to be embedded in any Metal application with minimal integration cost.

Core components --

- GPUGlitchDetector -- reads back texture and buffer data to scan for NaN/Inf pixels, outlier vertices, depth discontinuities, and black tile regions. Reports findings as typed GlitchReport structs with severity scores
- ShaderPatchManager -- carries a registry of known-bad shader patterns and injects corrective MSL source snippets at compile time. Ships with three default patches targeting the TBDR flush race, denorm banding, and depth precision mismatch
- FrameBufferValidator -- captures a known-good reference frame and compares subsequent frames pixel-by-pixel. When drift exceeds a configurable threshold, it blits the reference back to suppress visible flicker
- CommandBufferSanitizer -- wraps command buffer submission with automatic retry logic. On GPU error or timeout, it waits, retries with backoff, and logs the specific Metal error code for diagnosis
- GPUFixPipeline -- single entry point that initialises all components, runs diagnostics, applies fixes, and validates each frame through one method call


## triton-graphic -- GPU-Accelerated Python Layer

This layer reimplements every detection and repair operation as Triton JIT-compiled GPU kernels running through PyTorch. It is intended for offline frame analysis, batch processing of captured render targets, and integration into Python-based graphics pipelines.

Kernel modules --

- tritongraphicset -- full RGBA value clamping with NaN/Inf scrubbing, plus a half-precision denorm floor kernel that bumps sub-normal magnitudes to the fp16 minimum normal value
- texture_repair -- two-pass repair that first scrubs bad values in-place, then runs a horizontal neighbour-blend pass to reconstruct corrupted pixels from adjacent data
- depth_fix -- flags depth values with discontinuities exceeding a configurable threshold, then applies a 3-tap horizontal average to smooth z-fighting regions
- tile_gap_fill -- scans the frame in 32x32 tile increments to detect fully black regions, then fills each missing tile with the averaged centre pixel of its four neighbours
- flicker_suppress -- computes a per-pixel difference mask between the current frame and a reference, then blends flagged pixels back toward the reference at a configurable ratio
- vertex_sanitizer -- scans float3 vertex buffers for positions outside a bounding threshold or containing NaN/Inf, and clamps them in-place

The TritonGPUFixPipeline class in pipeline.py chains all passes into a single fix_frame() call that runs the full correction stack on the GPU in one shot.


## Integration

For native Metal applications --

    GPUFixPipeline pipeline;
    pipeline.init();
    pipeline.applyFixes();

    -- each frame
    pipeline.validateFrame(renderTarget);
    pipeline.submitWork([](id<MTLCommandBuffer> cmd) {
        -- encode your GPU work here
    });

    pipeline.printReport();

For Python / Triton --

    from triton_graphic import TritonGPUFixPipeline

    pipe = TritonGPUFixPipeline(device="cuda")
    fixed_frame = pipe.fix_frame(raw_frame, width, height)
    fixed_verts = pipe.fix_vertices(raw_positions)
    fixed_depth = pipe.fix_depth(raw_depth, width, height)
    pipe.print_report()


## Build

Requires CMake 3.20+, Xcode with Metal SDK, and C++14 --

    mkdir build && cd build
    cmake ..
    make

The triton-graphic module requires Python 3.10+, PyTorch, and Triton --

    pip install torch triton


## Targeted Hardware

- Apple M4 / M4 Pro / M4 Max / M4 Ultra
- Apple M5 / M5 Pro / M5 Max / M5 Ultra
- Apple A18 / A18 Pro
- Metal 3.2+ driver stack (macOS 16, iOS 20)


## References

- Apple Metal Best Practices Guide -- Tile-Based Deferred Rendering
- Apple Feedback FB14231897 -- TBDR tile memory flush race condition
- Apple Feedback FB14297744 -- Half-precision denorm flush inconsistency
- Apple Feedback FB14310056 -- Depth pre-pass invariant position mismatch
