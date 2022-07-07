cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(libtorchtext_cpp_example)

SET(BUILD_TORCHTEXT_PYTHON_EXTENSION OFF CACHE BOOL "Build Python binding")

find_package(Torch REQUIRED)
message("libtorchtext CMakeLists: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_library(torchtext SHARED IMPORTED)
set_property(TARGET torchtext PROPERTY IMPORTED_LOCATION /home/ubuntu/text/build/torchtext/csrc/libtorchtext.so)

add_subdirectory(../.. libtorchaudio)
add_executable(libtorchtext_cpp_example main.cpp)
target_link_libraries(libtorchtext_cpp_example  "${TORCH_LIBRARIES}" "-Wl,--no-as-needed -Wl,--whole-archive" torchtext "-Wl,--as-needed -Wl,--no-whole-archive")



cmake_minimum_required(VERSION 3.5)

project(libtorchaudio-cpp-example)

SET(BUILD_SOX ON CACHE BOOL "Build libsox into libtorchaudio")

SET(BUILD_KALDI OFF CACHE BOOL "Build Kaldi into libtorchaudio")
SET(BUILD_RNNT ON CACHE BOOL "Build RNN transducer into libtorchaudio")
SET(BUILD_TORCHAUDIO_PYTHON_EXTENSION OFF CACHE BOOL "Build Python binding")

find_package(Torch REQUIRED)
message("libtorchaudio CMakeLists: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(../.. libtorchaudio)
add_subdirectory(augmentation)
add_subdirectory(speech_recognition)
