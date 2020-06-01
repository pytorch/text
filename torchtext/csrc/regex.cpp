#include <regex>
#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

struct Regex : torch::CustomClassHolder {
private:
  std::regex re_;

public:
  // re_str_ holds the serialized regex string passed at the initialization.
  // We need this because we need to be able to serialize the model so that we
  // can save the scripted object. pickle will get the
  // serialized model from this re_str_ member, thus it needs to be public.
  std::string re_str_;

  Regex(const std::string &re_str) { UpdateRe(re_str_); }

  void UpdateRe(const std::string &re_str) {
    re_str_ = re_str;
    re_ = std::regex(re_str_);
  }

  std::string ReplaceFirstMatch(const std::string &str,
                                const std::string &repl) const {
    return std::regex_replace(str, re_, repl);
  }
};

// Registers our custom class with torch.
static auto regex =
    torch::class_<Regex>("torchtext", "Regex")
        .def(torch::init<std::string>())
        .def("UpdateRe", &Regex::UpdateRe)
        .def("ReplaceFirstMatch", &Regex::ReplaceFirstMatch)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<Regex> &self) -> std::string {
              return self->re_str_;
            },
            // __setstate__
            [](std::string state) -> c10::intrusive_ptr<Regex> {
              return c10::make_intrusive<Regex>(std::move(state));
            });

} // namespace
} // namespace torchtext
