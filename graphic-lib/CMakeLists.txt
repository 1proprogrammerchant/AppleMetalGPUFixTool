cmake_minimum_required(VERSION 3.20)

# --- graphic-lib: Apple Metal GPU Fix Library (2026) ---
# Built as a static library that can be linked into any Metal application.

set(GRAPHIC_LIB_SOURCES
    src/Metal/Device.cpp
    src/Metal/GPUGlitchDetector.cpp
    src/Metal/ShaderPatchManager.cpp
    src/Metal/FrameBufferValidator.cpp
    src/Metal/CommandBufferSanitizer.cpp
    src/Metal/GPUFixPipeline.cpp
)

set(GRAPHIC_LIB_HEADERS
    src/Metal/Device.hpp
    src/Metal/GPUGlitchDetector.hpp
    src/Metal/ShaderPatchManager.hpp
    src/Metal/FrameBufferValidator.hpp
    src/Metal/CommandBufferSanitizer.hpp
    src/Metal/GPUFixPipeline.hpp
)

add_library(graphic-lib STATIC ${GRAPHIC_LIB_SOURCES} ${GRAPHIC_LIB_HEADERS})

# Objective-C++ is required for Metal API calls
set_source_files_properties(${GRAPHIC_LIB_SOURCES}
    PROPERTIES COMPILE_FLAGS "-x objective-c++"
)

target_include_directories(graphic-lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Metal
)

# Link Apple frameworks
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(QUARTZCORE_FRAMEWORK QuartzCore)

target_link_libraries(graphic-lib PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
)

target_compile_features(graphic-lib PUBLIC cxx_std_14)
