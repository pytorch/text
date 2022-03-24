################################################################################
# libtorchtext
################################################################################
set(
  LIBTORCHTEXT_SOURCES
  clip_tokenizer.cpp
  clip_tokenizer.h
  common.cpp
  common.h
  gpt2_bpe_tokenizer.cpp
  gpt2_bpe_tokenizer.h
  regex.cpp
  regex.h
  regex_tokenizer.cpp
  regex_tokenizer.h
  register_torchbindings.cpp
  sentencepiece.cpp
  sentencepiece.h
  vectors.cpp
  vectors.h
  vocab.cpp
  vocab.h
  vocab_factory.h
  )

set(
  LIBTORCHTEXT_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/sentencepiece/src
  $<TARGET_PROPERTY:re2,INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:double-conversion,INCLUDE_DIRECTORIES>
  )

set(
  LIBTORCHTEXT_LINK_LIBRARIES
  torch
  re2
  double-conversion
  sentencepiece
  sentencepiece_train
  )

set(
  LIBTORCHTEXT_COMPILE_DEFINITIONS)

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
  function(define_extension name sources include_dirs libraries definitions)
    add_library(${name} SHARED ${sources})
    target_compile_definitions(${name} PRIVATE "${definitions}")
    target_include_directories(
      ${name} PRIVATE ${PROJECT_SOURCE_DIR} ${Python_INCLUDE_DIR} ${include_dirs})
    target_link_libraries(
      ${name}
      ${libraries}
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
    )

  set(
    EXTENSION_INCLUDE_DIRS
    $<TARGET_PROPERTY:re2,INCLUDE_DIRECTORIES>
    )

  set(
    EXTENSION_LIBRARIES
    libtorchtext
  )

  define_extension(
    _torchtext
    "${EXTENSION_INCLUDE_DIRS}"
    "${EXTENSION_SOURCES}"
    "${EXTENSION_LIBRARIES}"
    "${LIBTORCHTEXT_COMPILE_DEFINITIONS}"
    )
endif()
