#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <torch/script.h>
#include <torchtext/csrc/export.h>

namespace torchtext {

struct SentencePiece : torch::CustomClassHolder {
 private:
  sentencepiece::SentencePieceProcessor processor_;

 public:
  // content_ holds the serialized model data passed at the initialization.
  // We need this because the underlying SentencePieceProcessor class does not
  // provide serialization mechanism, yet we still need to be able to serialize
  // the model so that we can save the scripted object. pickle will get the
  // serialized model from this content_ member, thus it needs to be public.
  std::string content_;

  TORCHTEXT_API explicit SentencePiece(const std::string& content);
  TORCHTEXT_API std::vector<std::string> Encode(const std::string& input) const;
  TORCHTEXT_API std::vector<int64_t> EncodeAsIds(
      const std::string& input) const;
  TORCHTEXT_API std::string DecodeIds(const std::vector<int64_t>& ids) const;
  TORCHTEXT_API std::vector<std::string> EncodeAsPieces(
      const std::string& input) const;
  TORCHTEXT_API std::string DecodePieces(
      const std::vector<std::string>& pieces) const;
  TORCHTEXT_API int64_t GetPieceSize() const;
  TORCHTEXT_API int64_t unk_id() const;
  TORCHTEXT_API int64_t PieceToId(const std::string& piece) const;
  TORCHTEXT_API std::string IdToPiece(const int64_t id) const;
};

void generate_sp_model(
    const std::string& filename,
    const int64_t& vocab_size,
    const std::string& model_type,
    const std::string& model_prefix);
c10::intrusive_ptr<SentencePiece> load_sp_model(const std::string& path);
c10::intrusive_ptr<SentencePiece> load_sp_model_string(std::string content);

} // namespace torchtext
