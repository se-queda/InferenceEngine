cmake_minimum_required(VERSION 3.14)
project(InferenceEngine VERSION 1.0.0 LANGUAGES CXX)

## Global Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependency Management
include(FetchContent)

# --- FFMPEG (Clean PkgConfig Approach) ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavformat libavcodec libswresample libavutil
)

# --- PyBind11 ---
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# --- KissFFT ---
set(KISSFFT_TEST OFF CACHE BOOL "Disable KissFFT tests" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "Disable KissFFT tools" FORCE)
set(KISSFFT_DATATYPE "float" CACHE STRING "KissFFT data type" FORCE)

FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(kissfft)



# --- ONNX Runtime---
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS 
    "$ENV{CONDA_PREFIX}/include/onnxruntime/core/session"
    "$ENV{CONDA_PREFIX}/include"
)

find_library(ORT_LIB onnxruntime
    HINTS "$ENV{CONDA_PREFIX}/lib"
)

if(ORT_INCLUDE_DIR AND ORT_LIB)
    message(STATUS "Found ONNX Runtime Headers: ${ORT_INCLUDE_DIR}")
    message(STATUS "Found ONNX Runtime Lib: ${ORT_LIB}")
else()
    message(FATAL_ERROR "Could not find onnxruntime. Check paths in CMakeLists.txt")
endif()

# Sources
set(CORE_SOURCES
    src/Preprocessor.cpp
    src/AudioLoader.cpp
    src/InferenceEngine.cpp
)

# --- Target 1: Python Module ---
pybind11_add_module(audioguard_core
    bindings/python_bindings.cpp
    ${CORE_SOURCES}
)

target_include_directories(audioguard_core PRIVATE include ${ORT_INCLUDE_DIR})
target_link_libraries(audioguard_core PRIVATE kissfft PkgConfig::LIBAV ${ORT_LIB})

# --- Target 2: Executable ---
add_executable(AudioGuardApp
    App/main.cpp
    ${CORE_SOURCES}
)

target_include_directories(AudioGuardApp PRIVATE include ${ORT_INCLUDE_DIR})
target_link_libraries(AudioGuardApp PRIVATE kissfft PkgConfig::LIBAV ${ORT_LIB})