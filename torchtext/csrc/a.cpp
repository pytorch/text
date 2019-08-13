#include <torch/extension.h>
#include <string>
int x(int temp) { return temp + 1; }
PYBIND11_MODULE(_C, m) { m.def("x", &x, "x"); }
