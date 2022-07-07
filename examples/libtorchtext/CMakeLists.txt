cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorchtext_cpp_example)

SET(BUILD_TORCHTEXT_PYTHON_EXTENSION OFF CACHE BOOL "Build Python binding")

find_package(Torch REQUIRED)
message("libtorchtext CMakeLists: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(../.. libtorchtext)
add_subdirectory(tokenizer)
