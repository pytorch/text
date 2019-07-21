#include "text.h"

#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
  py::class_<torchtext::core::Vocabulary>(m, "Vocabulary")
      .def(py::init<>())
      .def("add_word", &torchtext::core::Vocabulary::add_word)
      .def("get_index", &torchtext::core::Vocabulary::get_index)
      .def("__repr__", [](const torchtext::core::Vocabulary& v) {
        (void)v;
        return "<_C.Core.Vocabulary>";
      });
}
