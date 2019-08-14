#include <torch/extension.h>
#include <string>

namespace torch {
namespace text {
// int x(int temp) { return 100; }
void x() { 0; }
}  // namespace text
}  // namespace torch

PYBIND11_MODULE(_wasup, m) { m.def("x", &torch::text::x, "x"); }
