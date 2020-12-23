#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex.h>
#include <regex_tokenizer.h>         // @manual
#include <sentencepiece.h>           // @manual
#include <torch/csrc/jit/python/pybind_utils.h> // @manual
#include <torch/csrc/utils/pybind.h> // @manual
#include <torch/script.h>
#include <vectors.h> // @manual
#include <vocab.h>   // @manual

namespace torchtext {

namespace py = pybind11;

namespace {
Vocab build_vocab_from_text_file(const std::string &file_path,
                                 const int64_t min_freq,
                                 const int64_t num_cpus,
                                 py::object fn) {
  torch::jit::script::Module module(*torch::jit::as_module(fn));
  return _build_vocab_from_text_file(file_path, min_freq, num_cpus, module);
}
} // namespace

// Registers our custom classes with pybind11.
PYBIND11_MODULE(_torchtext, m) {
  // Classes
  py::class_<Regex, c10::intrusive_ptr<Regex>>(m, "Regex")
      .def(py::init<std::string>())
      .def("Sub", &Regex::Sub)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Regex> &self) -> std::string {
            return _serialize_regex(self);
          },
          // __setstate__
          [](std::string state) -> c10::intrusive_ptr<Regex> {
            return _deserialize_regex(std::move(state));
          }));

  py::class_<RegexTokenizer, c10::intrusive_ptr<RegexTokenizer>>(m, "RegexTokenizer")
      .def_readonly("patterns_", &RegexTokenizer::patterns_)
      .def_readonly("replacements_", &RegexTokenizer::replacements_)
      .def_readonly("to_lower_", &RegexTokenizer::to_lower_)
      .def(py::init<std::vector<std::string>, std::vector<std::string>, bool>())
      .def("forward", &RegexTokenizer::forward)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<RegexTokenizer> &self) -> RegexTokenizerStates {
            return _serialize_regex_tokenizer(self);
          },
          // __setstate__
          [](RegexTokenizerStates states) -> c10::intrusive_ptr<RegexTokenizer> {
            return _deserialize_regex_tokenizer(std::move(states));
          }));

  py::class_<SentencePiece, c10::intrusive_ptr<SentencePiece>>(m, "SentencePiece")
      .def(py::init<std::string>())
      .def("_return_content",
           [](const SentencePiece &self) { return py::bytes(self.content_); })
      .def("Encode", &SentencePiece::Encode)
      .def("EncodeAsIds", &SentencePiece::EncodeAsIds)
      .def("DecodeIds", &SentencePiece::DecodeIds)
      .def("EncodeAsPieces", &SentencePiece::EncodeAsPieces)
      .def("DecodePieces", &SentencePiece::DecodePieces)
      .def("GetPieceSize", &SentencePiece::GetPieceSize)
      .def("unk_id", &SentencePiece::unk_id)
      .def("PieceToId", &SentencePiece::PieceToId)
      .def("IdToPiece", &SentencePiece::IdToPiece)
      .def(py::pickle(
           // __getstate__
           [](const c10::intrusive_ptr<SentencePiece> &self) -> py::bytes{
             return py::bytes(self->content_);
           },
           // __setstate__
           [](py::bytes state) -> c10::intrusive_ptr<SentencePiece> {
             return c10::make_intrusive<SentencePiece>(std::string(state));
           }));

  py::class_<Vectors, c10::intrusive_ptr<Vectors>>(m, "Vectors")
      .def(py::init<std::vector<std::string>, std::vector<int64_t>,
                    torch::Tensor, torch::Tensor>())
      .def_readonly("vectors_", &Vectors::vectors_)
      .def_readonly("unk_tensor_", &Vectors::unk_tensor_)
      .def("get_stoi", &Vectors::get_stoi)
      .def("__getitem__", &Vectors::__getitem__)
      .def("lookup_vectors", &Vectors::lookup_vectors)
      .def("__setitem__", &Vectors::__setitem__)
      .def("__len__", &Vectors::__len__)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Vectors> &self) -> VectorsStates {
            return _serialize_vectors(self);
          },
          // __setstate__
          [](VectorsStates states) -> c10::intrusive_ptr<Vectors> {
            return _deserialize_vectors(states);
          }));

  py::class_<Vocab, c10::intrusive_ptr<Vocab>>(m, "Vocab")
      .def(py::init<std::vector<std::string>>())
      .def_readonly("itos_", &Vocab::itos_)
      .def("__getitem__", &Vocab::__getitem__)
      .def("__len__", &Vocab::__len__)
      .def("insert_token", &Vocab::insert_token)
      .def("set_default_index", &Vocab::set_default_index)
      .def("get_default_index", &Vocab::get_default_index)
      .def("append_token", &Vocab::append_token)
      .def("lookup_token", &Vocab::lookup_token)
      .def("lookup_tokens", &Vocab::lookup_tokens)
      .def("lookup_indices", &Vocab::lookup_indices)
      .def("get_stoi", &Vocab::get_stoi)
      .def("get_itos", &Vocab::get_itos)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Vocab> &self) -> VocabStates {
            return _serialize_vocab(self);
          },
          // __setstate__
          [](VocabStates states) -> c10::intrusive_ptr<Vocab> {
            return _deserialize_vocab(states);
          }));

  // Functions
  m.def("_load_token_and_vectors_from_file",
        &_load_token_and_vectors_from_file);
  m.def("_load_vocab_from_file", &_load_vocab_from_file);
  m.def("_build_vocab_from_text_file", &build_vocab_from_text_file);
}

TORCH_LIBRARY_FRAGMENT(torchtext, m) {
  m.class_<Regex>("Regex")
    .def(torch::init<std::string>())
    .def("Sub", &Regex::Sub)
    .def_pickle(
        // __getstate__
        [](const c10::intrusive_ptr<Regex> &self) -> std::string {
          return _serialize_regex(self);
        },
        // __setstate__
        [](std::string state) -> c10::intrusive_ptr<Regex> {
          return _deserialize_regex(std::move(state));
        });

  m.class_<RegexTokenizer>("RegexTokenizer")
    .def(torch::init<std::vector<std::string>, std::vector<std::string>, bool>())
    .def("forward", &RegexTokenizer::forward)
    .def_pickle(
        // __getstate__
        [](const c10::intrusive_ptr<RegexTokenizer> &self) -> RegexTokenizerStates {
          return _serialize_regex_tokenizer(self);
        },
        // __setstate__
        [](RegexTokenizerStates states) -> c10::intrusive_ptr<RegexTokenizer> {
          return _deserialize_regex_tokenizer(std::move(states));
        });

  m.class_<SentencePiece>("SentencePiece")
    .def(torch::init<std::string>())
    .def("Encode", &SentencePiece::Encode)
    .def("EncodeAsIds", &SentencePiece::EncodeAsIds)
    .def("DecodeIds", &SentencePiece::DecodeIds)
    .def("EncodeAsPieces", &SentencePiece::EncodeAsPieces)
    .def("DecodePieces", &SentencePiece::DecodePieces)
    .def("GetPieceSize", &SentencePiece::GetPieceSize)
    .def("unk_id", &SentencePiece::unk_id)
    .def("PieceToId", &SentencePiece::PieceToId)
    .def("IdToPiece", &SentencePiece::IdToPiece)
    .def_pickle(
        // The underlying content of SentencePiece contains byte string,
        // and returing it as std::string cause UTF8 decoding error.
        // Since TorchScript does not support byte string, we use byte Tensor to
        // pass around the data.
        // __getstate__
        [](const c10::intrusive_ptr<SentencePiece> &self) -> torch::Tensor {
          auto *data = static_cast<void*>(const_cast<char*>(self->content_.data()));
          auto numel = static_cast<int64_t>(self->content_.size());
          return torch::from_blob(data, {numel}, {torch::kUInt8}).clone();
        },
        // __setstate__
        [](torch::Tensor state) -> c10::intrusive_ptr<SentencePiece> {
          auto *data = static_cast<char*>(state.data_ptr());
          auto numel = state.size(0);
          return c10::make_intrusive<SentencePiece>(std::string(data, numel));
        });

  m.class_<Vectors>("Vectors")
    .def(torch::init<std::vector<std::string>, std::vector<std::int64_t>, torch::Tensor, torch::Tensor>())
    .def("__getitem__", &Vectors::__getitem__)
    .def("lookup_vectors", &Vectors::lookup_vectors)
    .def("__setitem__", &Vectors::__setitem__)
    .def("__len__", &Vectors::__len__)
    .def_pickle(
        // __getstate__
        [](const c10::intrusive_ptr<Vectors> &self) -> VectorsStates {
          return _serialize_vectors(self);
        },
        // __setstate__
        [](VectorsStates states) -> c10::intrusive_ptr<Vectors> {
          return _deserialize_vectors(states);
        });

  m.class_<Vocab>("Vocab")
    .def(torch::init<StringList>())
    .def("__getitem__", &Vocab::__getitem__)
    .def("__len__", &Vocab::__len__)
    .def("insert_token", &Vocab::insert_token)
    .def("append_token", &Vocab::append_token)
    .def("lookup_token", &Vocab::lookup_token)
    .def("lookup_tokens", &Vocab::lookup_tokens)
    .def("lookup_indices", &Vocab::lookup_indices)
    .def("get_stoi", &Vocab::get_stoi)
    .def("get_itos", &Vocab::get_itos)
    .def_pickle(
        // __getstate__
        [](const c10::intrusive_ptr<Vocab> &self) -> VocabStates {
          return _serialize_vocab(self);
        },
        // __setstate__
        [](VocabStates states) -> c10::intrusive_ptr<Vocab> {
          return _deserialize_vocab(states);
        });

  m.def("torchtext::generate_sp_model", &generate_sp_model);
  m.def("torchtext::load_sp_model", &load_sp_model);
  m.def("torchtext::load_sp_model_string", &load_sp_model_string);
}

} // namespace torchtext
