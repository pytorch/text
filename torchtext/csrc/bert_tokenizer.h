#include <torchtext/csrc/vocab.h>
#include <string>
#include <vector>

namespace torchtext {

typedef std::basic_string<uint32_t> UString;

typedef std::tuple<bool, c10::optional<bool>, std::vector<std::string>>
    BERTEncoderStates;

struct BERTEncoder : torch::CustomClassHolder {
  BERTEncoder(
      const std::string& vocab_file,
      bool do_lower_case,
      c10::optional<bool> strip_accents);
  BERTEncoder(
      Vocab vocab,
      bool do_lower_case,
      c10::optional<bool> strip_accents);
  std::vector<std::string> Tokenize(std::string text);
  std::vector<int64_t> Encode(std::string text);
  Vocab vocab_;
  bool do_lower_case_;
  c10::optional<bool> strip_accents_ = {};

 protected:
  UString _clean(UString text, bool strip_accents);
  void _max_seg(std::string s, std::vector<std::string>& results);
  UString _basic_tokenize(UString text);
  void split_(
      std::string& str,
      std::vector<std::string>& tokens,
      const char& delimiter = ' ');
  static std::string kUnkToken;
};

BERTEncoderStates _serialize_bert_encoder(
    const c10::intrusive_ptr<BERTEncoder>& self);
c10::intrusive_ptr<BERTEncoder> _deserialize_bert_encoder(
    BERTEncoderStates states);
} // namespace torchtext
