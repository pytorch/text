#include "basic_english_normalize.h"
#include <sstream>

namespace torchtext {

BasicEnglishNormalize::BasicEnglishNormalize() {
  for (const auto &pattern : patterns_) {
    compiled_patterns_.push_back(new RE2(pattern));
  }
}

void BasicEnglishNormalize::split_(std::string &str,
                                   std::vector<std::string> &tokens,
                                   const char &delimiter = ' ') const {
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }
}

std::vector<std::string> BasicEnglishNormalize::forward(std::string str) const {
  // str tolower
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  for (size_t i = 0; i < compiled_patterns_.size(); i++) {
    RE2::GlobalReplace(&str, *compiled_patterns_[i], replacements_[i]);
  }

  std::vector<std::string> tokens;
  split_(str, tokens);
  return tokens;
}

// Registers our custom class with torch.
static auto basic_english_normalize =
    torch::class_<BasicEnglishNormalize>("torchtext", "BasicEnglishNormalize")
        .def(torch::init<>())
        .def("forward", &BasicEnglishNormalize::forward);

} // namespace torchtext
