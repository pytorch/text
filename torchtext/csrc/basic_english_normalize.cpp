#include "basic_english_normalize.h"
#include <sstream>

namespace torchtext {

BasicEnglishNormalize::BasicEnglishNormalize() {
  for (const auto &pattern : patterns_) {
    compiled_patterns_.push_back(new RE2(pattern));
  }
}

std::vector<std::string>
BasicEnglishNormalize::split_(std::string &str,
                              const char &delimiter = ' ') const {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }

  return tokens;
}

std::vector<std::string> BasicEnglishNormalize::forward(std::string str) const {
  // str tolower
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  for (size_t i = 0; i < compiled_patterns_.size(); i++) {
    RE2::GlobalReplace(&str, *compiled_patterns_[i], replacements_[i]);
  }
  return split_(str);
}

// Registers our custom class with torch.
static auto basic_english_normalize =
    torch::class_<BasicEnglishNormalize>("torchtext", "BasicEnglishNormalize")
        .def(torch::init<>())
        .def("forward", &BasicEnglishNormalize::forward);

} // namespace torchtext
