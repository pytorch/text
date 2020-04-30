#include <torch/extension.h>
#include <unordered_map>
#include <torch/custom_class.h>
#include <sentencepiece_processor.h>


namespace torchtext {
namespace {
struct SentencePiece : torch::CustomClassHolder {
 public:
  // public because we use it during registration below.
  std::string content_;
  explicit SentencePiece(std::string content) : content_(content) {
    auto status = processor_.LoadFromSerializedProto(content);
    if (!status.ok()) {
        std::stringstream ss;
        ss << "Failed to load SentencePiece model. Error: ";
        ss << status.ToString();
        throw std::runtime_error(ss.str());
    }
  }

  std::vector<std::string> process(const std::string& input) const {
    std::vector<std::string> pieces;
    processor_.Encode(input, &pieces);
    return pieces;
  }

 private:
  sentencepiece::SentencePieceProcessor processor_;
};
}

// Registers our custom class with torch.
static auto sentencepiece =
    torch::class_<SentencePiece>("torchtext", "SentencePiece")
    .def(torch::init<std::string>())
    .def("process", &SentencePiece::process)
    .def_pickle(
        // __setstate__
        [](const c10::intrusive_ptr<SentencePiece>& self) -> std::string {
            return self->content_;
        },
        // __getstate__
        [](std::string state) -> c10::intrusive_ptr<SentencePiece> {
            return c10::make_intrusive<SentencePiece>(std::move(state));
        }
    );
}
