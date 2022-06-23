#ifndef CLIP_TOKENIZER_H_
#define CLIP_TOKENIZER_H_

#include <torchtext/csrc/export.h>
#include <torchtext/csrc/gpt2_bpe_tokenizer.h>

namespace torchtext {

typedef std::tuple<
    std::unordered_map<std::string, int64_t>,
    std::unordered_map<std::string, int64_t>,
    std::string,
    std::unordered_map<int64_t, std::string>,
    bool>
    CLIPEncoderStatesPybind;

typedef std::tuple<
    c10::Dict<std::string, int64_t>,
    c10::Dict<std::string, int64_t>,
    std::string,
    c10::Dict<int64_t, std::string>,
    bool>
    CLIPEncoderStatesTorchbind;

struct CLIPEncoder : GPT2BPEEncoder {
 public:
  using GPT2BPEEncoder::GPT2BPEEncoder;

  TORCHTEXT_API std::vector<int64_t> Encode(const std::string& text);
  TORCHTEXT_API std::vector<std::string> Tokenize(const std::string& text);

 protected:
  TORCHTEXT_API std::vector<std::string> BPE_(
      const std::vector<std::string>& token_list) override;

  TORCHTEXT_API std::vector<std::string> PreTokenize_(
      std::string input) override;
};

TORCHTEXT_API CLIPEncoderStatesPybind
_serialize_clip_encoder_pybind(const c10::intrusive_ptr<CLIPEncoder>& self);
CLIPEncoderStatesTorchbind _serialize_clip_encoder_torchbind(
    const c10::intrusive_ptr<CLIPEncoder>& self);
TORCHTEXT_API c10::intrusive_ptr<CLIPEncoder> _deserialize_clip_encoder_pybind(
    CLIPEncoderStatesPybind states);
c10::intrusive_ptr<CLIPEncoder> _deserialize_clip_encoder_torchbind(
    CLIPEncoderStatesTorchbind states);

} // namespace torchtext

#endif // CLIP_TOKENIZER_H_
