get_property(TORCHTEXT_THIRDPARTY_LIBRARIES GLOBAL PROPERTY TORCHTEXT_THIRDPARTY_LIBRARIES)
get_property(TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES GLOBAL PROPERTY TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES)

set(
  TORCHTEXT_SOURCES
  common.cpp
  regex.cpp
  regex_tokenizer.cpp
  register_bindings.cpp
  sentencepiece.cpp
  vectors.cpp
  vocab.cpp
  )

add_library(
  _torchtext
  SHARED
  ${TORCHTEXT_SOURCES}
  )

set_target_properties(_torchtext PROPERTIES PREFIX "")

if (APPLE)
  # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
  # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
  set_target_properties(_torchtext PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

target_compile_definitions(
  _torchtext PRIVATE TORCH_API_INCLUDE_EXTENSION_H
  )

target_include_directories(
  _torchtext
  PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${Python_INCLUDE_DIR}
  ${TORCHTEXT_THIRDPARTY_INCLUDE_DIRECTORIES}
  )

# See https://github.com/pytorch/pytorch/issues/38122
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(
  _torchtext
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  ${TORCHTEXT_THIRDPARTY_LIBRARIES}
  )

install(TARGETS _torchtext LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
