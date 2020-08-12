#include <re2/re2.h>
#include <torch/script.h>

namespace torchtext {

struct RegexTokenizer : torch::CustomClassHolder {
private:
  std::vector<RE2 *> compiled_patterns_;
  void split_(std::string &str, std::vector<std::string> &tokens,
              const char &delimiter = ' ') const;

public:
  bool to_lower_;
  std::vector<std::string> patterns_;
  std::vector<std::string> replacements_;

  explicit RegexTokenizer(const std::vector<std::string> &patterns,
                          const std::vector<std::string> &replacements,
                          const bool to_lower);
  RegexTokenizer(const RegexTokenizer &regex_tokenizer);
  std::vector<std::string> forward(std::string str) const;
};

} // namespace torchtext
