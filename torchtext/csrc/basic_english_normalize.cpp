#include "basic_english_normalize.h"
#include <sstream>

namespace torchtext {

BasicEnglishNormalize::BasicEnglishNormalize() {
  for (const auto &pattern : patterns_) {
    regex_objects_.push_back(Regex(pattern));
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

  for (size_t i = 0; i < regex_objects_.size(); i++) {
    str = regex_objects_[i].Sub(str, replacements_[i]);
  }

  return split_(str);
}

// Registers our custom class with torch.
static auto basic_english_normalize =
    torch::class_<BasicEnglishNormalize>("torchtext", "BasicEnglishNormalize")
        .def(torch::init<>())
        .def("forward", &BasicEnglishNormalize::forward);

} // namespace torchtext
