cmake_minimum_required(VERSION 2.8)
project(torchtext)
set(CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)

file(GLOB HEADERS torchtext/csrc/text.h)
file(GLOB CORE_HEADERS torchtext/csrc/core/*.h)
file(GLOB CORE_SOURCES torchtext/csrc/core/*.h torchtext/csrc/core/*.cpp)

add_library (${PROJECT_NAME} SHARED ${CORE_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES ${CORE_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/core)

add_executable(benchmarks torchtext/csrc/benchmarks.cpp)
target_link_libraries(benchmarks PUBLIC benchmark)
target_link_libraries(benchmarks PUBLIC ${PROJECT_NAME})
configure_file(assets/benchmakr-article.txt assets/benchmakr-article.txt COPYONLY)
