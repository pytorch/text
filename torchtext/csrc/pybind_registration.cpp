#include <pybind11/pybind11.h>
#include <regex_tokenizer.h>
#include <torch/script.h>
#include <vectors.h>
#include <vocab.h>

using namespace torchtext;

// c10::intrusive_ptr<Vectors> to_jit_instance(Vectors vector){
//     return c10::make_intrusive<Vectors>(Vectors(stoindex, data_tensor,
//                                                 unk_tensor))};

// c10::intrusive_ptr<RegexTokenizer>
// to_jit_instance(RegexTokenizer regex_tokenizer) {
//   return c10::make_intrusive<RegexTokenizer>(regex_tokenizer);
// };

// void to_jit_instance(std::string b);
// // Registers our custom op with torch.
// TORCH_LIBRARY(torchtext, m) { m.def("to_jit_instance", &to_jit_instance);
// }

PYBIND11_MODULE(_torchtext, m) {

  //    m.def("to_jit_instance", (void (Pet::*)(int)) &Pet::to_jit_instance,
  //    "Set the pet's age")
  //    m.def("to_jit_instance", (void (Pet::*)(const std::string &))
  //    &Pet::to_jit_instance, "Set the pet's name");

  //   m.def("to_jit_instance", &to_jit_instance,
  //         "Convert an eager mode instance into a JITable instance.");

  register_regex_tokenizer_pybind(m);
  //   register_vectors_pybind(m);
  //   register_vocab_pybind(m);
}