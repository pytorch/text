#include <torch/extension.h>
#include <string>
std::string x(int temp) { return "foobarbaz"; }
PYBIND11_MODULE(_C, m) { m.def("x", &x, "x"); }
