#include <re2/re2.h>
#include <string>
#include <torch/script.h>

namespace torchtext {
namespace {

struct Regex : torch::CustomClassHolder {
public:
  // re_str_ holds the serialized regex string passed at the initialization.
  // We need this because we need to be able to serialize the model so that we
  // can save the scripted object. pickle will get the
  // serialized model from this re_str_ member, thus it needs to be public.
  std::string re_str_;

  Regex(const std::string &re_str) : re_str_(re_str) {}

  std::string Sub(const std::string &str, const std::string &repl) const {
    std::string mutable_str = str;
    RE2::GlobalReplace(&mutable_str, re_str_, repl);
    return mutable_str;
  }
};

// Registers our custom class with torch.
static auto regex =
    torch::class_<Regex>("torchtext", "Regex")
        .def(torch::init<std::string>())
        .def("Sub", &Regex::Sub)
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
