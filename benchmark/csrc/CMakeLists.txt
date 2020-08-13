cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(benchmark-jit_pipeline benchmark_jit_pipeline.cpp)
target_link_libraries(benchmark-jit_pipeline "${TORCH_LIBRARIES}")
set_property(TARGET benchmark-jit_pipeline PROPERTY CXX_STANDARD 14)