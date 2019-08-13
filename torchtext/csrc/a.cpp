#include <torch/extension.h>

int x() { return 11; }
PYBIND11_MODULE(_C, m) { m.def("x", &x, "x"); }
