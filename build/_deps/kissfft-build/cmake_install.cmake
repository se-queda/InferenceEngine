# Install script for directory: /home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/utsab/miniconda3/envs/tf-gpu/bin/x86_64-conda-linux-gnu-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkissfft-float.so.131.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkissfft-float.so.131"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/libkissfft-float.so.131.1.0"
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/libkissfft-float.so.131"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkissfft-float.so.131.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libkissfft-float.so.131"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/home/utsab/miniconda3/envs/tf-gpu/bin/x86_64-conda-linux-gnu-strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/libkissfft-float.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kissfft" TYPE FILE FILES
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-src/kiss_fft.h"
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-src/kissfft.hh"
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-src/kiss_fftnd.h"
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-src/kiss_fftndr.h"
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-src/kiss_fftr.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/kissfft/kissfft-float-shared-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/kissfft/kissfft-float-shared-targets.cmake"
         "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/CMakeFiles/Export/94098735959e389eabbc4f7bd7723935/kissfft-float-shared-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/kissfft/kissfft-float-shared-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/kissfft/kissfft-float-shared-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/kissfft" TYPE FILE FILES "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/CMakeFiles/Export/94098735959e389eabbc4f7bd7723935/kissfft-float-shared-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/kissfft" TYPE FILE FILES "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/CMakeFiles/Export/94098735959e389eabbc4f7bd7723935/kissfft-float-shared-targets-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/kissfft" TYPE FILE FILES
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/kissfft-config.cmake"
    "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/kissfft-config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/kissfft-float.pc")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/utsab/PycharmProjects/InferenceEngine/build/_deps/kissfft-build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
