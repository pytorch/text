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

std::vector<std::string>
BasicEnglishNormalize::forward(const std::string &str) const {
  std::string str_copy = str;

  // str tolower
  std::transform(str_copy.begin(), str_copy.end(), str_copy.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  for (size_t i = 0; i < regex_objects_.size(); i++) {
    str_copy = regex_objects_[i].Sub(str_copy, replacements_[i]);
  }

  return split_(str_copy);
}

// Registers our custom class with torch.
static auto basic_english_normalize =
    torch::class_<BasicEnglishNormalize>("torchtext", "BasicEnglishNormalize")
        .def(torch::init<>())
        .def("forward", &BasicEnglishNormalize::forward);

} // namespace torchtext
