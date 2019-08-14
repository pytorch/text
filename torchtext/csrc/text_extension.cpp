#include "text.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("basic_english_normalize",
        &torch::text::core::impl::basic_english_normalize,
        "basic_english_normalize");
}
