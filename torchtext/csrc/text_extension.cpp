#include "text.h"

#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
  m.def("basic_english_normalize",
        &torch::text::core::impl::basic_english_normalize,
        "basic_english_normalize");
}
