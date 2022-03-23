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


# # Sentencepiece cannot be built outside of the tree because the `third_party/absl` directory
# # is shadowed by the `third_party` directory in build binary directory and headers are
# # unaccessible.
# #
# # Apply custom patch to resolve that
# set(SPM_REPO ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece)
# execute_process(
#   WORKING_DIRECTORY ${SPM_REPO}
#   COMMAND "git" "checkout" "."
#   )
# execute_process(
#   WORKING_DIRECTORY ${SPM_REPO}
#   COMMAND git apply ../spm.patch
#   )
add_subdirectory(sentencepiece)