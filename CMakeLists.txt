cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Most of the configurations are taken from PyTorch
# https://github.com/pytorch/pytorch/blob/0c9fb4aff0d60eaadb04e4d5d099fb1e1d5701a9/CMakeLists.txt

# Use compiler ID "AppleClang" instead of "Clang" for XCode.
# Not setting this sometimes makes XCode C compiler gets detected as "Clang",
# even when the C++ one is detected as "AppleClang".
cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0025 NEW)

# Suppress warning flags in default MSVC configuration.  It's not
# mandatory that we do this (and we don't if cmake is old), but it's
# nice when it's possible, and it's possible on our Windows configs.
if(NOT CMAKE_VERSION VERSION_LESS 3.15.0)
  cmake_policy(SET CMP0092 NEW)
endif()

project(torchtext)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++14. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Apple specific
if(APPLE)
  # Get clang version on macOS
  execute_process( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
  string(REGEX REPLACE "Apple LLVM version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
  message( STATUS "CLANG_VERSION_STRING:         " ${CLANG_VERSION_STRING} )

  # RPATH stuff
  set(CMAKE_MACOSX_RPATH ON)

  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

# Windows specific
# if (MSVC)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")
# endif()


find_package(Torch REQUIRED)

# TORCH_CXX_FLAGS contains the same -D_GLIBCXX_USE_CXX11_ABI value as PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${TORCH_CXX_FLAGS}")

add_subdirectory(third_party)
add_subdirectory(torchtext/csrc)
