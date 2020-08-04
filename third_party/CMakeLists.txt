# Old enough to support Ubuntu Xenial.
cmake_minimum_required(VERSION 3.5.1)

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(thirdparty CXX)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_subdirectory(re2)
