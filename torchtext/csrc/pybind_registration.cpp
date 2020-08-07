#include <regex_tokenizer.h>
#include <torch/script.h>

using namespace torchtext;
namespace py = pybind11;

register_regex_tokenizer_pybind();