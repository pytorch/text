#include <regex_tokenizer.h>
#include <sstream>

namespace torchtext {

RegexTokenizer::RegexTokenizer(const std::vector<std::string> &patterns,
                               const std::vector<std::string> &replacements,
                               const bool to_lower = false)
    : patterns_(std::move(patterns)), replacements_(std::move(replacements)),
      to_lower_(to_lower) {
  TORCH_CHECK(patterns.size() == replacements.size(),
              "Expected `patterns` and `replacements` to have same size!");

  for (const auto &pattern : patterns_) {
    compiled_patterns_.push_back(new RE2(pattern));
  }
}

RegexTokenizer::RegexTokenizer(const RegexTokenizer &regex_tokenizer) {
  patterns_ = regex_tokenizer.patterns_;
  replacements_ = regex_tokenizer.replacements_;
  compiled_patterns_ = regex_tokenizer.compiled_patterns_;
}

std::vector<std::string> RegexTokenizer::forward(std::string str) const {
  // str tolower
  if (to_lower_) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  }

  for (size_t i = 0; i < compiled_patterns_.size(); i++) {
    RE2::GlobalReplace(&str, *compiled_patterns_[i], replacements_[i]);
  }

  std::vector<std::string> tokens;
  split_(str, tokens);
  return tokens;
}

void RegexTokenizer::split_(std::string &str, std::vector<std::string> &tokens,
                            const char &delimiter) const {
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }
}

// Registers our custom class with torch.
static auto regex_tokenizer =
    torch::class_<RegexTokenizer>("torchtext", "RegexTokenizer")
        .def(torch::init<std::vector<std::string>, std::vector<std::string>,
                         bool>())
        .def("forward", &RegexTokenizer::forward)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<RegexTokenizer> &self) -> std::tuple<
                std::vector<std::string>, std::vector<std::string>, bool> {
              return std::make_tuple(self->patterns_, self->replacements_,
                                     self->to_lower_);
            },
            // __getstate__
            [](std::tuple<std::vector<std::string>, std::vector<std::string>,
                          bool>
                   states) -> c10::intrusive_ptr<RegexTokenizer> {
              auto patterns = std::get<0>(states);
              auto replacements = std::get<1>(states);
              auto to_lower = std::get<2>(states);

              return c10::make_intrusive<RegexTokenizer>(
                  std::move(patterns), std::move(replacements), to_lower);
            });

namespace py = pybind11;
// Registers our custom class with pybind11.
void register_regex_tokenizer_pybind(pybind11::module m) {
  py::class_<RegexTokenizer>(m, "RegexTokenizer")
      .def(py::init<std::vector<std::string>, std::vector<std::string>, bool>())
      .def("forward", &RegexTokenizer::forward);
  // m.def("to_jit", &to_jit);
}

c10::IValue to_jit(c10::IValue val) {
  std::string temp = "test1";
  std::cout << "Returning an IValue" << std::endl;
  return c10::IValue(temp);
  // return torch::make_custom_class<std::string>(temp);
}

// c10::IValue to_jit(c10::IValue val) {
//   // c10::intrusive_ptr<RegexTokenizer> to_jit(c10::IValue val) {
//   // PyObject *p = val.toPyObject();
//   // py::object o = py::reinterpret_borrow<py::object>(p);
//   // RegexTokenizer result = o.cast<RegexTokenizer>();
//   // std::cout <<;

//   // return torch::make_custom_class<RegexTokenizer>(result);
//   // return c10::make_intrusive<RegexTokenizer>(result);
//   std::vector<std::string> temp;
//   return torch::make_custom_class<RegexTokenizer>(temp, temp, false);

//   // return c10::make_intrusive<RegexTokenizer>(temp, temp, false);
// }

// Registers our custom op with torch.
TORCH_LIBRARY(torchtext, m) { m.def("to_jit", &to_jit); }

} // namespace torchtext
