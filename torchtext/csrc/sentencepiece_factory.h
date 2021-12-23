#include <pybind11/pybind11.h>
#include <sentencepiece_trainer.h>

namespace py = pybind11;

namespace torchtext {

  class SentenceIterator : public sentencepiece::SentenceIterator {
  public:

  SentenceIterator(py::iterator &v) : it(v.begin()) {
    copy();
  }

  ~SentenceIterator() { }

  bool done() const override {
    return it == py::iterator::sentinel();
  }

  void Next() override {
    ++it;
    copy();
  }

  const std::string &value() const override {
    return item_;
  }

  sentencepiece::util::Status status() const override {
    return status_;
  }

  private:
   int i;
   py::iterator it;
   sentencepiece::util::Status status_;
   std::string item_;

   void copy() {
     if (it == nullptr) {
       return;
     }
     std::string s = it->cast<std::string>();
     const char *data = s.data();
     size_t size = s.size();
     while (size > 0) {
       if (data[size - 1] == '\r' || data[size - 1] == '\n')
         --size;
       else
         break;
     }
     item_.assign(data, size);
   }

};

void _generate_sp_model_from_iterator(py::iterator &lines, const int64_t &vocab_size,
                       const std::string &model_type,
                       const std::string &model_prefix) {
  SentenceIterator *it = new SentenceIterator(lines);
  const auto status = ::sentencepiece::SentencePieceTrainer::Train(
      " --model_prefix=" + model_prefix +
      " --vocab_size=" + std::to_string(vocab_size) +
      " --model_type=" + model_type, it, nullptr);
  if (!status.ok()) {
    throw std::runtime_error("Failed to train SentencePiece model. Error: " +
                             status.ToString());
  }
}
}