#include <sentencepiece.h>

namespace torchtext {

SentencePiece::SentencePiece(const std::string &content) : content_(content) {
  const auto status = processor_.LoadFromSerializedProto(content_);
  if (!status.ok()) {
    throw std::runtime_error("Failed to load SentencePiece model. Error: " +
                             status.ToString());
  }
}

std::vector<std::string> SentencePiece::Encode(const std::string &input) const {
  std::vector<std::string> pieces;
  processor_.Encode(input, &pieces);
  return pieces;
}

std::vector<int64_t>
SentencePiece::EncodeAsIds(const std::string &input) const {
  const auto val = processor_.EncodeAsIds(input);
  return std::vector<int64_t>(val.begin(), val.end());
}

std::vector<std::string>
SentencePiece::EncodeAsPieces(const std::string &input) const {
  return processor_.EncodeAsPieces(input);
}

int64_t SentencePiece::GetPieceSize() const {
  return processor_.GetPieceSize();
}

int64_t SentencePiece::unk_id() const { return processor_.unk_id(); }

int64_t SentencePiece::PieceToId(const std::string &piece) const {
  return processor_.PieceToId(piece);
}

std::string SentencePiece::IdToPiece(const int64_t id) const {
  return processor_.IdToPiece(id);
}

void generate_sp_model(const std::string &filename, const int64_t &vocab_size,
                       const std::string &model_type,
                       const std::string &model_prefix) {
  const auto status = ::sentencepiece::SentencePieceTrainer::Train(
      "--input=" + filename + " --model_prefix=" + model_prefix +
      " --vocab_size=" + std::to_string(vocab_size) +
      " --model_type=" + model_type);
  if (!status.ok()) {
    throw std::runtime_error("Failed to train SentencePiece model. Error: " +
                             status.ToString());
  }
}

c10::intrusive_ptr<SentencePiece> load_sp_model(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::in);
  if (!file) {
    throw std::runtime_error("Failed to open file :" + path);
  }
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  return c10::make_intrusive<SentencePiece>(std::move(content));
}

} // namespace torchtext
