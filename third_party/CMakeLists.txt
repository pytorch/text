if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_subdirectory(re2)
add_subdirectory(double-conversion)

# Sentencepiece cannot be built outside of the tree because the `third_party/absl` directory
# is shadowed by the `third_party` directory in build binary directory and headers are
# unaccessible.
#
# Apply custom patch to resolve that
set(SPM_REPO ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece)
execute_process(
  WORKING_DIRECTORY ${SPM_REPO}
  COMMAND "git" "checkout" "."
  )
execute_process(
  WORKING_DIRECTORY ${SPM_REPO}
  COMMAND git apply ../spm.patch
  )
add_subdirectory(sentencepiece)

set(TORCHTEXT_THIRDPARTY_LIBRARIES
  re2
  double-conversion
  sentencepiece
  sentencepiece_train
  )
set(TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/re2
  ${CMAKE_CURRENT_SOURCE_DIR}/double-conversion
  ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece/src
  )
set_property(GLOBAL PROPERTY TORCHTEXT_THIRDPARTY_LIBRARIES "${TORCHTEXT_THIRDPARTY_LIBRARIES}")
set_property(GLOBAL PROPERTY TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES "${TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES}")
