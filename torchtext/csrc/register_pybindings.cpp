#include <clip_tokenizer.h> // @manual
#include <gpt2_bpe_tokenizer.h> // @manual
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex.h>
#include <regex_tokenizer.h> // @manual
#include <sentencepiece.h> // @manual
#include <torch/csrc/jit/python/pybind_utils.h> // @manual
#include <torch/csrc/utils/pybind.h> // @manual
#include <torch/script.h>
#include <vectors.h> // @manual
#include <vocab.h> // @manual
#include <vocab_factory.h> // @manual

#include <iostream>

namespace torchtext {

namespace py = pybind11;

namespace {
Vocab build_vocab_from_text_file(
    const std::string& file_path,
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
      .def("FindAndConsume", &Regex::FindAndConsume)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Regex>& self) -> std::string {
            return _serialize_regex(self);
          },
          // __setstate__
          [](std::string state) -> c10::intrusive_ptr<Regex> {
            return _deserialize_regex(std::move(state));
          }));

  py::class_<RegexTokenizer, c10::intrusive_ptr<RegexTokenizer>>(
      m, "RegexTokenizer")
      .def_readonly("patterns_", &RegexTokenizer::patterns_)
      .def_readonly("replacements_", &RegexTokenizer::replacements_)
      .def_readonly("to_lower_", &RegexTokenizer::to_lower_)
      .def(py::init<std::vector<std::string>, std::vector<std::string>, bool>())
      .def("forward", &RegexTokenizer::forward)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<RegexTokenizer>& self)
              -> RegexTokenizerStates {
            return _serialize_regex_tokenizer(self);
          },
          // __setstate__
          [](RegexTokenizerStates states)
              -> c10::intrusive_ptr<RegexTokenizer> {
            return _deserialize_regex_tokenizer(std::move(states));
          }));

  py::class_<SentencePiece, c10::intrusive_ptr<SentencePiece>>(
      m, "SentencePiece")
      .def(py::init<std::string>())
      .def(
          "_return_content",
          [](const SentencePiece& self) { return py::bytes(self.content_); })
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
          [](const c10::intrusive_ptr<SentencePiece>& self) -> py::bytes {
            return py::bytes(self->content_);
          },
          // __setstate__
          [](py::bytes state) -> c10::intrusive_ptr<SentencePiece> {
            return c10::make_intrusive<SentencePiece>(std::string(state));
          }));

  py::class_<Vectors, c10::intrusive_ptr<Vectors>>(m, "Vectors")
      .def(py::init<
           std::vector<std::string>,
           std::vector<int64_t>,
           torch::Tensor,
           torch::Tensor>())
      .def_readonly("vectors_", &Vectors::vectors_)
      .def_readonly("unk_tensor_", &Vectors::unk_tensor_)
      .def("get_stoi", &Vectors::get_stoi)
      .def("__getitem__", &Vectors::__getitem__)
      .def("lookup_vectors", &Vectors::lookup_vectors)
      .def("__setitem__", &Vectors::__setitem__)
      .def("__len__", &Vectors::__len__)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Vectors>& self) -> VectorsStates {
            return _serialize_vectors(self);
          },
          // __setstate__
          [](VectorsStates states) -> c10::intrusive_ptr<Vectors> {
            return _deserialize_vectors(states);
          }));

  py::class_<Vocab, c10::intrusive_ptr<Vocab>>(m, "Vocab")
      .def(py::init<StringList, c10::optional<int64_t>>())
      .def_readonly("itos_", &Vocab::itos_)
      .def_readonly("default_index_", &Vocab::default_index_)
      .def(
          "__contains__",
          [](c10::intrusive_ptr<Vocab>& self, const py::str& item) -> bool {
            Py_ssize_t length;
            const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
            return self->__contains__(c10::string_view{buffer, (size_t)length});
          })
      .def(
          "__getitem__",
          [](c10::intrusive_ptr<Vocab>& self, const py::str& item) -> int64_t {
            Py_ssize_t length;
            const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
            return self->__getitem__(c10::string_view{buffer, (size_t)length});
          })
      .def("insert_token", &Vocab::insert_token)
      .def("set_default_index", &Vocab::set_default_index)
      .def("get_default_index", &Vocab::get_default_index)
      .def("__len__", &Vocab::__len__)
      .def("append_token", &Vocab::append_token)
      .def("lookup_token", &Vocab::lookup_token)
      .def("lookup_tokens", &Vocab::lookup_tokens)
      .def(
          "lookup_indices",
          [](const c10::intrusive_ptr<Vocab>& self, const py::list& items) {
            std::vector<int64_t> indices(items.size());
            int64_t counter = 0;
            for (const auto& item : items) {
              Py_ssize_t length;
              const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
              indices[counter++] =
                  self->__getitem__(c10::string_view{buffer, (size_t)length});
            }
            return indices;
          })
      .def("get_stoi", &Vocab::get_stoi)
      .def("get_itos", &Vocab::get_itos)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Vocab>& self) -> VocabStates {
            return _serialize_vocab(self);
          },
          // __setstate__
          [](VocabStates states) -> c10::intrusive_ptr<Vocab> {
            return _deserialize_vocab(states);
          }));

  py::class_<GPT2BPEEncoder, c10::intrusive_ptr<GPT2BPEEncoder>>(
      m, "GPT2BPEEncoder")
      .def(py::init<
           std::unordered_map<std::string, int64_t>,
           std::unordered_map<std::string, int64_t>,
           std::string,
           std::unordered_map<int64_t, std::string>,
           bool>())
      .def_property_readonly("bpe_encoder_", &GPT2BPEEncoder::GetBPEEncoder)
      .def_property_readonly(
          "bpe_merge_ranks_", &GPT2BPEEncoder::GetBPEMergeRanks)
      .def_readonly("seperator_", &GPT2BPEEncoder::seperator_)
      .def_property_readonly("byte_encoder_", &GPT2BPEEncoder::GetByteEncoder)
      .def("encode", &GPT2BPEEncoder::Encode)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<GPT2BPEEncoder>& self)
              -> GPT2BPEEncoderStatesPybind {
            return _serialize_gpt2_bpe_encoder_pybind(self);
          },
          // __setstate__
          [](GPT2BPEEncoderStatesPybind states)
              -> c10::intrusive_ptr<GPT2BPEEncoder> {
            return _deserialize_gpt2_bpe_encoder_pybind(states);
          }));

  py::class_<CLIPEncoder, c10::intrusive_ptr<CLIPEncoder>>(m, "CLIPEncoder")
      .def(py::init<
           std::unordered_map<std::string, int64_t>,
           std::unordered_map<std::string, int64_t>,
           std::string,
           std::unordered_map<int64_t, std::string>,
           bool>())
      .def_property_readonly("bpe_encoder_", &CLIPEncoder::GetBPEEncoder)
      .def_property_readonly("bpe_merge_ranks_", &CLIPEncoder::GetBPEMergeRanks)
      .def_readonly("seperator_", &CLIPEncoder::seperator_)
      .def_property_readonly("byte_encoder_", &CLIPEncoder::GetByteEncoder)
      .def("encode", &CLIPEncoder::Encode)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<CLIPEncoder>& self)
              -> CLIPEncoderStatesPybind {
            return _serialize_clip_encoder_pybind(self);
          },
          // __setstate__
          [](CLIPEncoderStatesPybind states)
              -> c10::intrusive_ptr<CLIPEncoder> {
            return _deserialize_clip_encoder_pybind(states);
          }));

  // Functions
  m.def(
      "_load_token_and_vectors_from_file", &_load_token_and_vectors_from_file);
  m.def("_load_vocab_from_file", &_load_vocab_from_file);
  m.def("_build_vocab_from_text_file", &build_vocab_from_text_file);
  m.def(
      "_build_vocab_from_text_file_using_python_tokenizer",
      &_build_vocab_from_text_file_using_python_tokenizer);
}

} // namespace torchtext
