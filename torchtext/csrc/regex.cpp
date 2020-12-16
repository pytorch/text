#include <regex.h>

namespace torchtext {

Regex::Regex(const std::string &re_str) : re_str_(re_str) {
  compiled_pattern_ = new RE2(re_str_);
}

std::string Regex::Sub(std::string str, const std::string &repl) const {
  RE2::GlobalReplace(&str, *compiled_pattern_, repl);
  return str;
}

std::string _serialize_regex(const c10::intrusive_ptr<Regex> &self) {
  return self->re_str_;
}

c10::intrusive_ptr<Regex> _deserialize_regex(std::string state) {
  return c10::make_intrusive<Regex>(std::move(state));
}

} // namespace torchtext
