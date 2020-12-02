#include <re2/re2.h>
#include <torch/script.h>

namespace torchtext {

typedef std::tuple<std::vector<std::string>, std::vector<std::string>, bool>
    RegexTokenizerStates;

struct RegexTokenizer : torch::CustomClassHolder {
private:
  std::vector<RE2 *> compiled_patterns_;
  void split_(std::string &str, std::vector<std::string> &tokens,
              const char &delimiter = ' ') const;

public:
  std::vector<std::string> patterns_;
  std::vector<std::string> replacements_;
  bool to_lower_;

  explicit RegexTokenizer(const std::vector<std::string> &patterns,
                          const std::vector<std::string> &replacements,
                          const bool to_lower);
  std::vector<std::string> forward(std::string str) const;
};

c10::intrusive_ptr<RegexTokenizer>
_get_regex_tokenizer_from_states(RegexTokenizerStates states);
RegexTokenizerStates _set_regex_tokenizer_states(const RegexTokenizer &self);

} // namespace torchtext
