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

std::vector<std::string> RegexTokenizer::forward(std::string_view str_) const {
  // str tolower
  std::string str(str_.size(), ' ');
  // if (to_lower_) {
    for (int64_t i = 0; i < str_.size(); i++) {
      str[i] = std::tolower(str_[i]);
    }
    // std::transform(str.begin(), str.end(), str.begin(),
    //                [](unsigned char c) { return std::tolower(c); });
  // }

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

} // namespace torchtext
