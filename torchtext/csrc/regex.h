#include <string>
#include <torch/script.h>

namespace torchtext {
struct Regex : torch::CustomClassHolder {
public:
  std::string re_str_;

  Regex(const std::string &re_str);
  std::string Sub(const std::string &str, const std::string &repl) const;
};
}