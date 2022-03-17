# # Old enough to support Ubuntu Xenial.
# # cmake_minimum_required(VERSION 3.5.1)

# # set(TORCHTEXT_THIRD_PARTIES "")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")


# if(POLICY CMP0091)
#   cmake_policy(SET CMP0091 NEW)
# endif()

# project(thirdparty CXX)
# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# add_subdirectory(re2)
# add_subdirectory(double-conversion)



set(TORCHTEXT_THIRD_PARTIES "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")


if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_subdirectory(re2)
add_subdirectory(double-conversion)

set_property(GLOBAL PROPERTY TORCHTEXT_THIRD_PARTIES re2 double-conversion)