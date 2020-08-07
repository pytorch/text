#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/script.h>

namespace torchtext {

void register_vectors_pybind(pybind11::module m);
} // namespace torchtext
