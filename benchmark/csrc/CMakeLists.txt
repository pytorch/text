cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)


add_library(TorchText::TorchText SHARED IMPORTED) # or STATIC instead of SHARED
set_target_properties(TorchText::TorchText PROPERTIES IMPORTED_LOCATION "/private/home/nayef211/torchtext/build/lib.linux-x86_64-3.7/torchtext/_torchtext.so")

add_executable(benchmark-jit_pipeline benchmark_jit_pipeline.cpp)
target_link_libraries(benchmark-jit_pipeline "${TORCH_LIBRARIES}" TorchText::TorchText)
set_property(TARGET benchmark-jit_pipeline PROPERTY CXX_STANDARD 14)