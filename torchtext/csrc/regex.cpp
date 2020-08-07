#include <regex.h>
#include <re2/re2.h>

namespace torchtext {

Regex::Regex(const std::string &re_str) : re_str_(re_str) {
  compiled_pattern_ = new RE2(re_str_);
}

std::string Regex::Sub(std::string str, const std::string &repl) const {
  RE2::GlobalReplace(&str, *compiled_pattern_, repl);
  return str;
}

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

} // namespace torchtext
