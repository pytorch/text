#include "text.h"

#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
    m.def("split_tokenizer", &torch::text::core::impl::split_tokenizer, "split_tokenizer");
}
