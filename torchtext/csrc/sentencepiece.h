#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <torch/script.h>
#include <Python.h>

namespace torchtext {

// Inspired by https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/sentencepiece.i
static PyObject* kUnicodeInput = reinterpret_cast<PyObject* >(0x1);
static PyObject* kByteInput = reinterpret_cast<PyObject* >(0x2);
class PyInputString {
 public:
  explicit PyInputString(PyObject* obj) {
    if (PyUnicode_Check(obj)) {
       // Python3, Unicode
      str_ = const_cast<char *>(PyUnicode_AsUTF8AndSize(obj, &size_));
      input_type_ = kUnicodeInput;
    } else if (PyBytes_Check(obj)) {
       // Python3, Bytes
      PyBytes_AsStringAndSize(obj, &str_, &size_);
      input_type_ = kByteInput;
    } else {
      str_ = nullptr;
    }
  }
  const char* data() const { return str_; }
  Py_ssize_t size() const { return size_; }
  bool IsAvalable() const { return str_ != nullptr; }
  PyObject *input_type() const { return input_type_; }

  static bool IsUnicode(PyObject *resultobj) {
    return (resultobj == nullptr || resultobj == kUnicodeInput);
  }

 private:
  PyObject* input_type_ = nullptr;
  char* str_ = nullptr;
  Py_ssize_t size_ = 0;
};
  // Inspired by https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/sentencepiece.i
  class PySentenceIterator : public sentencepiece::SentenceIterator {
  public:
  PySentenceIterator(PyObject *iter) : iter_(iter) {
    item_ = PyIter_Next(iter_);
    CopyValue();
  }

  ~PySentenceIterator() {
   // Py_XDECREF(iter_);
  }

  bool done() const override {
    return item_ == nullptr;
  }

  void Next() override {
    item_ = PyIter_Next(iter_);
    CopyValue();
  }

  const std::string &value() const override {
    return value_;
  }

  sentencepiece::util::Status status() const override {
    return status_;
  }

  private:
   void CopyValue() {
     if (item_ == nullptr) return;
     const PyInputString ustring(item_);
     if (ustring.IsAvalable()) {
       const char *data = ustring.data();
       size_t size = ustring.size();
       while (size > 0) {
         if (data[size - 1] == '\r' || data[size - 1] == '\n')
           --size;
         else
           break;
       }
       value_.assign(data, size);
     } else {
       status_ = sentencepiece::util::Status(sentencepiece::util::StatusCode::kInternal,
                                             "Not a string.");
     }
     Py_XDECREF(item_);
   }
   PyObject *iter_ = nullptr;
   PyObject *item_ = nullptr;
   std::string value_;
   sentencepiece::util::Status status_;
};

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

  explicit SentencePiece(const std::string &content);
  std::vector<std::string> Encode(const std::string &input) const;
  std::vector<int64_t> EncodeAsIds(const std::string &input) const;
  std::string DecodeIds(const std::vector<int64_t> &ids) const;
  std::vector<std::string> EncodeAsPieces(const std::string &input) const;
  std::string DecodePieces(const std::vector<std::string> &pieces) const;
  int64_t GetPieceSize() const;
  int64_t unk_id() const;
  int64_t PieceToId(const std::string &piece) const;
  std::string IdToPiece(const int64_t id) const;
};

void generate_sp_model(const std::string &filename, const int64_t &vocab_size,
                       const std::string &model_type,
                       const std::string &model_prefix);

void generate_sp_model_from_iterator(PyObject *iter,
                       const int64_t &vocab_size,
                       const std::string &model_type,
                       const std::string &model_prefix);

c10::intrusive_ptr<SentencePiece> load_sp_model(const std::string &path);
c10::intrusive_ptr<SentencePiece>
load_sp_model_string(std::string content);

} // namespace torchtext
