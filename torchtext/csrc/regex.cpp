#include <regex.h>

namespace torchtext {

Regex::Regex(const std::string &re_str) : re_str_(re_str) {
  std::cout << "[re_str] " << re_str << std::endl;
  compiled_pattern_ = new pcrecpp::RE(re_str_, pcrecpp::UTF8());
}

std::string Regex::Sub(std::string str, const std::string &repl) const {
  (*compiled_pattern_).GlobalReplace(repl, &str);
  return str;
}

std::vector<std::string> Regex::find_all(std::string input) {
  pcrecpp::StringPiece line(input);
  std::string token;
  std::vector<std::string> tokens;

  std::cout << "[line] " << line << std::endl;
  while ((*compiled_pattern_).FindAndConsume(&line, &token)) {
    std::cout << "[token] " << token << std::endl;
    tokens.push_back(token);
  }

  return tokens;
}

std::string _serialize_regex(const c10::intrusive_ptr<Regex> &self) {
  return self->re_str_;
}

c10::intrusive_ptr<Regex> _deserialize_regex(std::string &&state) {
  return c10::make_intrusive<Regex>(std::move(state));
}

} // namespace torchtext
