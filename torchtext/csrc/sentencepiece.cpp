#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <torch/script.h>

#ifdef _WIN32
#include <Python.h>
PyMODINIT_FUNC PyInit__torchtext(void) {
  // No need to do anything.
  // extension.py will run on load
  return NULL;
}
#endif

namespace torchtext {
namespace {

struct SentencePiece : torch::CustomClassHolder {
private:
  sentencepiece::SentencePieceProcessor processor_;

public:
  // content_ holds the serialized model data passed at the initialization.
  // We need this because the underlying SentencePieceProcessor class does not
  // provide serialization mechanism, yet we still need to be able to serialize
  // the model so that we can save the scripted object. pickle will get the
  // serialized model from this content_ member, thus it needs to be public.
  const std::string content_;

  explicit SentencePiece(const std::string &content) : content_(content) {
    const auto status = processor_.LoadFromSerializedProto(content_);
    if (!status.ok()) {
      throw std::runtime_error("Failed to load SentencePiece model. Error: " +
                               status.ToString());
    }
  }

  std::vector<std::string> Encode(const std::string &input) const {
    std::vector<std::string> pieces;
    processor_.Encode(input, &pieces);
    return pieces;
  }

  std::vector<int64_t> EncodeAsIds(const std::string &input) const {
    const auto val = processor_.EncodeAsIds(input);
    return std::vector<int64_t>(val.begin(), val.end());
  }

  std::vector<std::string> EncodeAsPieces(const std::string &input) const {
    return processor_.EncodeAsPieces(input);
  }

  std::vector<std::vector<std::string>>
  NBestEncodeAsPieces(const std::string &input,
                      const int64_t nbest_size) const {
    return processor_.NBestEncodeAsPieces(input, nbest_size);
  }

  std::vector<std::vector<int64_t>>
  NBestEncodeAsIds(const std::string &input, const int64_t nbest_size) const {
    const auto vals = processor_.NBestEncodeAsIds(input, nbest_size);
    std::vector<std::vector<std::int64_t>> ret;
    for (const auto &vec : vals) {
      ret.emplace_back(std::vector<std::int64_t>(vec.begin(), vec.end()));
    }
    return ret;
  }

  std::vector<std::string> SampleEncodeAsPieces(const std::string &input,
                                                const int64_t nbest_size,
                                                const double alpha) const {
    return processor_.SampleEncodeAsPieces(input, nbest_size, alpha);
  }

  std::vector<int64_t> SampleEncodeAsIds(const std::string &input,
                                         const int64_t nbest_size,
                                         const double alpha) const {
    const auto val = processor_.SampleEncodeAsIds(input, nbest_size, alpha);
    return std::vector<int64_t>(val.begin(), val.end());
  }

  std::string DecodePieces(const std::vector<std::string> &input) const {
    return processor_.DecodePieces(input);
  }

  std::string DecodeIds(const std::vector<int64_t> &input) const {
    const std::vector<int> val(input.begin(), input.end());
    return processor_.DecodeIds(val);
  }

  int64_t GetPieceSize() const { return processor_.GetPieceSize(); }

  int64_t PieceToId(const std::string &piece) const {
    return processor_.PieceToId(piece);
  }

  std::string IdToPiece(const int64_t id) const {
    return processor_.IdToPiece(id);
  }

  double GetScore(const int64_t id) const { return processor_.GetScore(id); }

  bool IsUnknown(const int64_t id) const { return processor_.IsUnknown(id); }

  bool IsUnused(const int64_t id) const { return processor_.IsUnused(id); }

  int64_t unk_id() const { return processor_.unk_id(); }

  int64_t bos_id() const { return processor_.bos_id(); }

  int64_t eos_id() const { return processor_.eos_id(); }

  int64_t pad_id() const { return processor_.pad_id(); }

  void SetEncodeExtraOptions(const std::string &extra_option) {
    const auto status = processor_.SetEncodeExtraOptions(extra_option);
    if (!status.ok()) {
      throw std::runtime_error("Failed to set encode extra options. Error: " +
                               status.ToString());
    }
  }

  void SetDecodeExtraOptions(const std::string &extra_option) {
    const auto status = processor_.SetDecodeExtraOptions(extra_option);
    if (!status.ok()) {
      throw std::runtime_error("Failed to set decode extra options. Error: " +
                               status.ToString());
    }
  }

  void SetVocabulary(const std::vector<std::string> &valid_vocab) {
    const auto status = processor_.SetVocabulary(valid_vocab);
    if (!status.ok()) {
      throw std::runtime_error("Failed to set vocabulary. Error: " +
                               status.ToString());
    }
  }

  void ResetVocabulary() {
    const auto status = processor_.ResetVocabulary();
    if (!status.ok()) {
      throw std::runtime_error("Failed to reset vocabulary. Error: " +
                               status.ToString());
    }
  }

  void LoadVocabulary(const std::string &filename, int64_t threshold) {
    const auto status = processor_.LoadVocabulary(filename, threshold);
    if (!status.ok()) {
      throw std::runtime_error("Failed to load vocabulary. Error: " +
                               status.ToString());
    }
  }
};

// Registers our custom class with torch.
static auto sentencepiece =
    torch::class_<SentencePiece>("torchtext", "SentencePiece")
        .def("Encode", &SentencePiece::Encode)
        .def("EncodeAsIds", &SentencePiece::EncodeAsIds)
        .def("EncodeAsPieces", &SentencePiece::EncodeAsPieces)
        .def("NBestEncodeAsPieces", &SentencePiece::NBestEncodeAsPieces)
        .def("NBestEncodeAsIds", &SentencePiece::NBestEncodeAsIds)
        .def("SampleEncodeAsPieces", &SentencePiece::SampleEncodeAsPieces)
        .def("SampleEncodeAsIds", &SentencePiece::SampleEncodeAsIds)
        .def("DecodePieces", &SentencePiece::DecodePieces)
        .def("DecodeIds", &SentencePiece::DecodeIds)
        .def("GetPieceSize", &SentencePiece::GetPieceSize)
        .def("PieceToId", &SentencePiece::PieceToId)
        .def("IdToPiece", &SentencePiece::IdToPiece)
        .def("GetScore", &SentencePiece::GetScore)
        .def("IsUnknown", &SentencePiece::IsUnknown)
        .def("IsUnused", &SentencePiece::IsUnused)
        .def("unk_id", &SentencePiece::unk_id)
        .def("bos_id", &SentencePiece::bos_id)
        .def("eos_id", &SentencePiece::eos_id)
        .def("pad_id", &SentencePiece::pad_id)
        .def("SetEncodeExtraOptions", &SentencePiece::SetEncodeExtraOptions)
        .def("SetDecodeExtraOptions", &SentencePiece::SetDecodeExtraOptions)
        .def("SetVocabulary", &SentencePiece::SetVocabulary)
        .def("ResetVocabulary", &SentencePiece::ResetVocabulary)
        .def("LoadVocabulary", &SentencePiece::LoadVocabulary)
        .def("__len__", &SentencePiece::GetPieceSize)
        .def("__getitem__", &SentencePiece::PieceToId)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<SentencePiece> &self) -> std::string {
              return self->content_;
            },
            // __getstate__
            [](std::string state) -> c10::intrusive_ptr<SentencePiece> {
              return c10::make_intrusive<SentencePiece>(std::move(state));
            });

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

static auto registry =
    torch::RegisterOperators()
        .op("torchtext::generate_sp_model", &generate_sp_model)
        .op(torch::RegisterOperators::options()
                .schema("torchtext::load_sp_model(str path) -> "
                        "__torch__.torch.classes.torchtext.SentencePiece model")
                .catchAllKernel<decltype(load_sp_model), &load_sp_model>());

} // namespace
} // namespace torchtext
