#include <torch/extension.h>
#include <string>
std::string x(std::string temp) { return temp + " world"; }
PYBIND11_MODULE(_C, m) { m.def("x", &x, "x"); }
