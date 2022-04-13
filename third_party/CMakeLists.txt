set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")


if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

add_subdirectory(re2)
add_subdirectory(double-conversion)
add_subdirectory(sentencepiece)
