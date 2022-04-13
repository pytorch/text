set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")


if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

add_subdirectory(re2 EXCLUDE_FROM_ALL)
add_subdirectory(double-conversion EXCLUDE_FROM_ALL)
add_subdirectory(sentencepiece EXCLUDE_FROM_ALL)
