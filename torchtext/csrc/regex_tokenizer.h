#pragma once
#include <common.h>
#include <re2/re2.h>
#include <torch/script.h>

namespace torchtext {

struct RegexTokenizer : torch::CustomClassHolder {
private:
  std::vector<RE2 *> compiled_patterns_;
  void split_(std::string &str, StringList &tokens,
              const char &delimiter = ' ') const;

public:
  std::vector<std::string> patterns_;
  std::vector<std::string> replacements_;
  bool to_lower_;

  explicit RegexTokenizer(ConstStringList patterns,
                          ConstStringList replacements, const bool to_lower);
  std::vector<std::string> forward(std::string str) const;
};

} // namespace torchtext
