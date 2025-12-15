# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-src")
  file(MAKE_DIRECTORY "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-src")
endif()
file(MAKE_DIRECTORY
  "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build"
  "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-subbuild/kissfft-populate-prefix"
  "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-subbuild/kissfft-populate-prefix/tmp"
  "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-subbuild/kissfft-populate-prefix/src/kissfft-populate-stamp"
  "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-subbuild/kissfft-populate-prefix/src"
  "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-subbuild/kissfft-populate-prefix/src/kissfft-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-subbuild/kissfft-populate-prefix/src/kissfft-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-subbuild/kissfft-populate-prefix/src/kissfft-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
