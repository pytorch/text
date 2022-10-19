################################################################################
# libtorchtext
################################################################################
set(
  LIBTORCHTEXT_SOURCES
  clip_tokenizer.cpp
  common.cpp
  gpt2_bpe_tokenizer.cpp
  regex.cpp
  regex_tokenizer.cpp
  register_torchbindings.cpp
  sentencepiece.cpp
  vectors.cpp
  vocab.cpp
  bert_tokenizer.cpp
  )

set(
  LIBTORCHTEXT_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/sentencepiece/src
  $<TARGET_PROPERTY:re2,INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:double-conversion,INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:utf8proc,INCLUDE_DIRECTORIES>
  ${TORCH_INSTALL_PREFIX}/include
  ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include
  )

set(
  LIBTORCHTEXT_LINK_LIBRARIES
  ${TORCH_C10_LIBRARY}
  ${TORCH_LIBRARY}
  ${TORCH_CPU_LIBRARY}
  re2
  double-conversion
  sentencepiece
  sentencepiece_train
  utf8proc
  )

set(
  LIBTORCHTEXT_COMPILE_DEFINITIONS
  TORCHTEXT_BUILD_MAIN_LIB)

function (define_library name source include_dirs link_libraries compile_defs)
  add_library(${name} SHARED ${source})
  target_include_directories(${name} PRIVATE ${include_dirs})
  target_link_libraries(${name} ${link_libraries})
  target_compile_definitions(${name} PRIVATE ${compile_defs})
  set_target_properties(${name} PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
  endif(MSVC)
  install(
    TARGETS ${name}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib  # For Windows
    )
endfunction()

define_library(
  libtorchtext
  "${LIBTORCHTEXT_SOURCES}"
  "${LIBTORCHTEXT_INCLUDE_DIRS}"
  "${LIBTORCHTEXT_LINK_LIBRARIES}"
  "${LIBTORCHTEXT_COMPILE_DEFINITIONS}"
  )

if (APPLE)
  set(TORCHTEXT_LIBRARY libtorchtext CACHE INTERNAL "")
else()
  set(TORCHTEXT_LIBRARY -Wl,--no-as-needed libtorchtext -Wl,--as-needed CACHE INTERNAL "")
endif()


################################################################################
# _torchtext.so
################################################################################
if (BUILD_TORCHTEXT_PYTHON_EXTENSION)
  # See https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  if (WIN32)
    find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)
    set(ADDITIONAL_ITEMS Python3::Python)
  endif()
  function(define_extension name sources include_dirs link_libraries definitions)
    add_library(${name} SHARED ${sources})
    target_compile_definitions(${name} PRIVATE "${definitions}")
    target_include_directories(
      ${name} PRIVATE ${Python_INCLUDE_DIR} ${include_dirs})
    target_link_libraries(
      ${name}
      ${link_libraries}
      ${TORCH_PYTHON_LIBRARY}
      ${ADDITIONAL_ITEMS}
      )
    set_target_properties(${name} PROPERTIES PREFIX "")
    if (MSVC)
      set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
    endif(MSVC)
    if (APPLE)
      # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
      # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
      set_target_properties(${name} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    install(
      TARGETS ${name}
      LIBRARY DESTINATION .
      RUNTIME DESTINATION .  # For Windows
      )
  endfunction()

  set(
    EXTENSION_SOURCES
    register_pybindings.cpp
    vocab_factory.cpp
    )

  set(
    EXTENSION_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party/sentencepiece/src
    $<TARGET_PROPERTY:re2,INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:double-conversion,INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:utf8proc,INCLUDE_DIRECTORIES>
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include
    )

  set(
    EXTENSION_LINK_LIBRARIES
    libtorchtext
  )

  define_extension(
    _torchtext
    "${EXTENSION_SOURCES}"
    "${EXTENSION_INCLUDE_DIRS}"
    "${EXTENSION_LINK_LIBRARIES}"
    "${LIBTORCHTEXT_COMPILE_DEFINITIONS}"
    )
endif()
