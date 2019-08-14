#include <torch/extension.h>
#include <string>

namespace torch {
namespace text {
int x(int temp) { return temp + 1; }
}  // namespace text
}  // namespace torch

// PYBIND11_MODULE(_C, m) { m.def("x", &torch::text::x, "x"); }
