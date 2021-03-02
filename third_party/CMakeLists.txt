if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_subdirectory(re2)
add_subdirectory(double-conversion)

# Apply custom patch
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
  sentencepiece-static
  sentencepiece_train-static
  )
set(TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/re2
  ${CMAKE_CURRENT_SOURCE_DIR}/double-conversion
  ${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece/src
  )
set_property(GLOBAL PROPERTY TORCHTEXT_THIRDPARTY_LIBRARIES "${TORCHTEXT_THIRDPARTY_LIBRARIES}")
set_property(GLOBAL PROPERTY TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES "${TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES}")
