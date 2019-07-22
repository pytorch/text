#include "text.h"

#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
  py::class_<torchtext::core::Dictionary>(m, "Dictionary")
      .def(py::init<>())
      .def("add_word", &torchtext::core::Dictionary::add_word)
      .def("get_index", &torchtext::core::Dictionary::get_index)
      .def("__repr__", [](const torchtext::core::Dictionary& v) {
        (void)v;
        return "<_C.Core.Vocabulary>";
      });
}
