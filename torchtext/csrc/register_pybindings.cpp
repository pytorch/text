#include <torch/extension.h>
#include <torchtext/csrc/foo.h>

PYBIND11_MODULE(_torchtext, m) {
  m.def("foo", &torchtext::foo);
}
