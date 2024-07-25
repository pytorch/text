add_executable(tokenize main.cpp)
target_link_libraries(tokenize "${TORCH_LIBRARIES}" "${TORCHTEXT_LIBRARY}")
set_property(TARGET tokenize PROPERTY CXX_STANDARD 14)
