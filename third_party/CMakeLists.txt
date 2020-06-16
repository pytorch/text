if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(thirdparty CXX)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

add_subdirectory(re2)