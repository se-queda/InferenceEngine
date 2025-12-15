cmake_minimum_required(VERSION 3.14)
project(InferenceEngine VERSION 1.0.0 LANGUAGES CXX)

## Global Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependency Management
include(FetchContent)

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

# Sources
# Note: Ensure these files actually exist in 'src/'
set(CORE_SOURCES
    src/Preprocessor.cpp
    src/AudioLoader.cpp
    src/InferenceEngine.cpp
)

# --- Target 1: Python Module ---
# CRITICAL: You must create the bindings folder!
pybind11_add_module(audioguard_core
    bindings/python_bindings.cpp
    ${CORE_SOURCES}
)

target_include_directories(audioguard_core PRIVATE include)
target_link_libraries(audioguard_core PRIVATE kissfft)

# --- Target 2: Executable ---
# UPDATED: Changed 'app' to 'App' to match your folder
add_executable(AudioGuardApp
    App/main.cpp
    ${CORE_SOURCES}
)

target_include_directories(AudioGuardApp PRIVATE include)
target_link_libraries(AudioGuardApp PRIVATE kissfft)