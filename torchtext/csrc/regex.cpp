#include <regex.h>

namespace torchtext {

Regex::Regex(const_string re_str) : re_str_(re_str) {
  compiled_pattern_ = new RE2(re_str_);
}

std::string Regex::Sub(std::string str, std::string repl) const {
  RE2::GlobalReplace(&str, *compiled_pattern_, repl);
  return str;
}

} // namespace torchtext
