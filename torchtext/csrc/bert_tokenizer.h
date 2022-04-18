#include <string>
#include <vector>

namespace torchtext {
struct BERTEncoder {
  BERTEncoder(const std::string& vocab_file);
  std::vector<std::string> tokenize(const std::string& text);
};
} // namespace torchtext
