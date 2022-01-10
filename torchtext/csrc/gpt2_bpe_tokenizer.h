#include <torch/script.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace torchtext {

typedef std::tuple<std::unordered_map<std::string, int64_t>,
                   std::unordered_map<std::string, int64_t>, std::string,
                   std::unordered_map<int64_t, std::string>, bool>
    GPT2BPEEncoderStatesPybind;

typedef std::tuple<c10::Dict<std::string, int64_t>,
                   c10::Dict<std::string, int64_t>, std::string,
                   c10::Dict<int64_t, std::string>, bool>
    GPT2BPEEncoderStatesTorchbind;

// Applies regex based pre-tokenization step for GPT-2 BPE tokenizer
// and returns a list of tokens.
std::vector<std::string> gpt2_bpe_pre_tokenizer(std::string input);
struct GPT2BPEEncoder : torch::CustomClassHolder {
 private:
  const int64_t inf_;
  c10::Dict<std::string, std::vector<std::string>> cache_;
  // Encode byte into an unicode character.
  std::vector<std::string> ByteEncode_(std::string token);
  // Return a list of bpe tokens.
  std::vector<std::string> BPE_(const std::vector<std::string> &token_list);
  // Return the token pair(e.g bpe merge) with lowest rank.
  std::string FindBestPair_(std::vector<std::string> pairs);
  int64_t GetBPEMergeRank_(std::string pair);

 public:
  const c10::Dict<std::string, int64_t> bpe_encoder_;
  const c10::Dict<std::string, int64_t> bpe_merge_ranks_;
  const c10::Dict<int64_t, std::string> byte_encoder_;
  const std::string seperator_;
  const bool caching_enabled_;
  explicit GPT2BPEEncoder(
      const c10::Dict<std::string, int64_t> &bpe_encoder,
      const c10::Dict<std::string, int64_t> &bpe_merge_ranks,
      const std::string &seperator,
      const c10::Dict<int64_t, std::string> &byte_encoder,
      bool caching_enabled = false);

  explicit GPT2BPEEncoder(
      const std::unordered_map<std::string, int64_t> &bpe_encoder,
      const std::unordered_map<std::string, int64_t> &bpe_merge_ranks,
      const std::string &seperator,
      const std::unordered_map<int64_t, std::string> &byte_encoder,
      bool caching_enabled = false);

  // Encode text into a list of bpe token ids.
  //
  // Split text into a list of token unit, and generate a list of bpe tokens
  // for each token unit. Lastly encode bpe tokens into bpe token ids.
  //
  // For example: "awesome,awe"
  //  --> tokenize(regex) --> tokens: ["awesome", ",", "awe"]
  //  --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
  //  --> bpe encode --> bpe token ids: [707, 5927], [11], [707, 68]
  //  --> result --> [707, 5927, 11, 707, 68]
  //
  std::vector<int64_t> Encode(const std::string &text);

  std::unordered_map<std::string, int64_t> GetBPEEncoder() const;
  std::unordered_map<std::string, int64_t> GetBPEMergeRanks() const;
  std::unordered_map<int64_t, std::string> GetByteEncoder() const;
};

GPT2BPEEncoderStatesPybind _serialize_gpt2_bpe_encoder_pybind(
    const c10::intrusive_ptr<GPT2BPEEncoder> &self);
GPT2BPEEncoderStatesTorchbind _serialize_gpt2_bpe_encoder_torchbind(
    const c10::intrusive_ptr<GPT2BPEEncoder> &self);
c10::intrusive_ptr<GPT2BPEEncoder> _deserialize_gpt2_bpe_encoder_pybind(
    GPT2BPEEncoderStatesPybind states);
c10::intrusive_ptr<GPT2BPEEncoder> _deserialize_gpt2_bpe_encoder_torchbind(
    GPT2BPEEncoderStatesTorchbind states);
}  // namespace torchtext
