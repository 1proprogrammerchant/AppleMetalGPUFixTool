cmake_minimum_required(VERSION 3.20)
project(AppleMetalGPUFixTool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Sub-libraries ---
add_subdirectory(graphic-lib)

# --- Find Apple frameworks ---
find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(QUARTZCORE_LIBRARY QuartzCore)

add_executable(AppleMetalGPUFixTool
    src/main.cpp
    src/MetalDevice.cpp
)

target_include_directories(AppleMetalGPUFixTool PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(AppleMetalGPUFixTool PRIVATE
    graphic-lib
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${QUARTZCORE_LIBRARY}
)