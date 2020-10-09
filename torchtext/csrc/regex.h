#pragma once
#include <re2/re2.h>
#include <string>
#include <torch/script.h>
#include <common.h>

namespace torchtext {
struct Regex : torch::CustomClassHolder {
private:
  RE2 *compiled_pattern_;

public:
  std::string re_str_;

  Regex(const_string re_str);
  std::string Sub(std::string str, std::string repl) const;
};
} // namespace torchtext
