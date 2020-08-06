#include <re2/re2.h>
#include <torch/script.h>

namespace torchtext {

struct BasicEnglishNormalize : torch::CustomClassHolder {
private:
  std::vector<std::string> patterns_{"'",   "\"",  "\\.", "<br \\/>",
                                     ",",   "\\(", "\\)", "\\!",
                                     "\\?", "\\;", "\\:", "\\s+"};
  std::vector<std::string> replacements_{
      " '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "};
  std::vector<RE2 *> compiled_patterns_;

  std::vector<std::string> split_(std::string &str,
                                  const char &delimiter) const;

public:
  explicit BasicEnglishNormalize();

  std::vector<std::string> forward(std::string str) const;
};
} // namespace torchtext
