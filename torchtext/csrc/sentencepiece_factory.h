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
   py::iterator it;
   sentencepiece::util::Status status_;
   std::string item_;

   void copy() {
     if (*it == nullptr) {
       return;
     }
     if (!py::isinstance<py::str>(*it)) {
        throw std::runtime_error("Iterator contains non-strings.");
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
  SentenceIterator it = SentenceIterator(lines);
  sentencepiece::util::bytes model_proto;
  const auto status = ::sentencepiece::SentencePieceTrainer::Train(
      " --vocab_size=" + std::to_string(vocab_size) +
      " --model_type=" + model_type +
      " --hard_vocab_limit=false", &it, &model_proto);
  if (!status.ok()) {
    throw std::runtime_error("Failed to train SentencePiece model. Error: " +
                             status.ToString());
  }
  std::ofstream file;
  file.open(model_prefix + ".model");
  file << model_proto;
  file.close();

}
}