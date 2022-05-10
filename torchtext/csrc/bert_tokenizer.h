#include <torchtext/csrc/vocab.h>
#include <string>
#include <vector>

namespace torchtext {

typedef std::basic_string<uint32_t> UString;

struct BERTEncoder:torch::CustomClassHolder {
  BERTEncoder(const std::string& vocab_file);
  std::vector<std::string> Tokenize(std::string text);
  std::vector<int64_t> Encode(std::string text);

 protected:
  Vocab vocab_;
  UString _clean(UString text);
  void max_seg_(std::string s, std::vector<std::string>& results);
  UString _basic_tokenize(UString text);
  void split_(
      std::string& str,
      std::vector<std::string>& tokens,
      const char& delimiter = ' ');
  static std::string kUnkToken;
};
} // namespace torchtext
