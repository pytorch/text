#include <pybind11/pybind11.h>
#include <vocab.h> // @manual

namespace py = pybind11;

namespace torchtext {

  class SentenceIterator : public sentencepiece::SentenceIterator, torch::CustomClassHolder {
  public:

  SentenceIterator(py::iterator &v) : it(v) {
    // TODO: check if the elements are std::string
  }

  ~SentenceIterator() { }

  bool done() const override {
    return it == py::iterator::sentinel();
  }

  void Next() override {
    ++it;
  }

  const std::string &value() const override {
    return py::str(*it);
  }

  sentencepiece::util::Status status() const override {
    return status_;
  }

  private:
   py::iterator it;
   sentencepiece::util::Status status_;
};

void _generate_sp_model_from_iterator(py::iterator &lines, const int64_t &vocab_size,
                       const std::string &model_type,
                       const std::string &model_prefix) {
  const auto status = ::sentencepiece::SentencePieceTrainer::Train(
      " --model_prefix=" + model_prefix +
      " --vocab_size=" + std::to_string(vocab_size) +
      " --model_type=" + model_type, new SentenceIterator(lines));
  if (!status.ok()) {
    throw std::runtime_error("Failed to train SentencePiece model. Error: " +
                             status.ToString());
  }
}
}