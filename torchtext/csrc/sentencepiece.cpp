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

  int64_t GetPieceSize() const { return processor_.GetPieceSize(); }
};

// Registers our custom class with torch.
static auto sentencepiece =
    torch::class_<SentencePiece>("torchtext", "SentencePiece")
        .def("Encode", &SentencePiece::Encode)
        .def("EncodeAsIds", &SentencePiece::EncodeAsIds)
        .def("EncodeAsPieces", &SentencePiece::EncodeAsPieces)
        .def("GetPieceSize", &SentencePiece::GetPieceSize)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<SentencePiece> &self) -> std::string {
              return self->content_;
            },
            // __getstate__
            [](std::string state) -> c10::intrusive_ptr<SentencePiece> {
              return c10::make_intrusive<SentencePiece>(std::move(state));
            });

void generate_sp_model(const std::string filename, const int64_t vocab_size,
                       const std::string model_type,
                       const std::string model_prefix) {
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
  const std::string content((std::istreambuf_iterator<char>(file)),
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
