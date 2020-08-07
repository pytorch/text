#include <pybind11/pybind11.h>
#include <regex_tokenizer.h>
#include <torch/script.h>
#include <vectors.h>
#include <vocab.h>

using namespace torchtext;
// namespace py = pybind11;

PYBIND11_MODULE(_torchtext, m) {
  register_regex_tokenizer_pybind(m);
  register_vectors_pybind(m);
  register_vocab_pybind(m);
}