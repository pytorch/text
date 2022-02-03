#include <clip_tokenizer.h> // @manual
#include <gpt2_bpe_tokenizer.h> // @manual
#include <regex.h>
#include <regex_tokenizer.h> // @manual
#include <sentencepiece.h> // @manual
#include <torch/script.h>
#include <vectors.h> // @manual
#include <vocab.h> // @manual

#include <iostream>
namespace torchtext {

TORCH_LIBRARY_FRAGMENT(torchtext, m) {
  m.class_<Regex>("Regex")
      .def(torch::init<std::string>())
      .def("Sub", &Regex::Sub)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Regex>& self) -> std::string {
            return _serialize_regex(self);
          },
          // __setstate__
          [](std::string state) -> c10::intrusive_ptr<Regex> {
            return _deserialize_regex(std::move(state));
          });

  m.class_<RegexTokenizer>("RegexTokenizer")
      .def(torch::
               init<std::vector<std::string>, std::vector<std::string>, bool>())
      .def("forward", &RegexTokenizer::forward)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<RegexTokenizer>& self)
              -> RegexTokenizerStates {
            return _serialize_regex_tokenizer(self);
          },
          // __setstate__
          [](RegexTokenizerStates states)
              -> c10::intrusive_ptr<RegexTokenizer> {
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
          // Since TorchScript does not support byte string, we use byte Tensor
          // to pass around the data.
          // __getstate__
          [](const c10::intrusive_ptr<SentencePiece>& self) -> torch::Tensor {
            auto* data =
                static_cast<void*>(const_cast<char*>(self->content_.data()));
            auto numel = static_cast<int64_t>(self->content_.size());
            return torch::from_blob(data, {numel}, {torch::kUInt8}).clone();
          },
          // __setstate__
          [](torch::Tensor state) -> c10::intrusive_ptr<SentencePiece> {
            auto* data = static_cast<char*>(state.data_ptr());
            auto numel = state.size(0);
            return c10::make_intrusive<SentencePiece>(std::string(data, numel));
          });

  m.class_<Vectors>("Vectors")
      .def(torch::init<
           std::vector<std::string>,
           std::vector<std::int64_t>,
           torch::Tensor,
           torch::Tensor>())
      .def("__getitem__", &Vectors::__getitem__)
      .def("lookup_vectors", &Vectors::lookup_vectors)
      .def("__setitem__", &Vectors::__setitem__)
      .def("__len__", &Vectors::__len__)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Vectors>& self) -> VectorsStates {
            return _serialize_vectors(self);
          },
          // __setstate__
          [](VectorsStates states) -> c10::intrusive_ptr<Vectors> {
            return _deserialize_vectors(states);
          });

  m.class_<Vocab>("Vocab")
      .def(torch::init<StringList, c10::optional<int64_t>>())
      .def(
          "__contains__",
          [](const c10::intrusive_ptr<Vocab>& self, const std::string& item)
              -> bool { return self->__contains__(c10::string_view{item}); })
      .def(
          "__getitem__",
          [](const c10::intrusive_ptr<Vocab>& self, const std::string& item)
              -> int64_t { return self->__getitem__(c10::string_view{item}); })
      .def("insert_token", &Vocab::insert_token)
      .def("__len__", &Vocab::__len__)
      .def("set_default_index", &Vocab::set_default_index)
      .def("get_default_index", &Vocab::get_default_index)
      .def("append_token", &Vocab::append_token)
      .def("lookup_token", &Vocab::lookup_token)
      .def("lookup_tokens", &Vocab::lookup_tokens)
      .def(
          "lookup_indices",
          [](const c10::intrusive_ptr<Vocab>& self,
             const std::vector<std::string>& items) {
            std::vector<int64_t> indices(items.size());
            int64_t counter = 0;
            for (const auto& item : items) {
              indices[counter++] = self->__getitem__(c10::string_view{item});
            }
            return indices;
          })
      .def("get_stoi", &Vocab::get_stoi)
      .def("get_itos", &Vocab::get_itos)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Vocab>& self) -> VocabStates {
            return _serialize_vocab(self);
          },
          // __setstate__
          [](VocabStates states) -> c10::intrusive_ptr<Vocab> {
            return _deserialize_vocab(states);
          });

  m.class_<GPT2BPEEncoder>("GPT2BPEEncoder")
      .def(torch::init<
           c10::Dict<std::string, int64_t>,
           c10::Dict<std::string, int64_t>,
           std::string,
           c10::Dict<int64_t, std::string>,
           bool>())
      .def("encode", &GPT2BPEEncoder::Encode)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<GPT2BPEEncoder>& self)
              -> GPT2BPEEncoderStatesTorchbind {
            return _serialize_gpt2_bpe_encoder_torchbind(self);
          },
          // __setstate__
          [](GPT2BPEEncoderStatesTorchbind states)
              -> c10::intrusive_ptr<GPT2BPEEncoder> {
            return _deserialize_gpt2_bpe_encoder_torchbind(states);
          });

  m.class_<CLIPEncoder>("CLIPEncoder")
      .def(torch::init<
           c10::Dict<std::string, int64_t>,
           c10::Dict<std::string, int64_t>,
           std::string,
           c10::Dict<int64_t, std::string>,
           bool>())
      .def("encode", &CLIPEncoder::Encode)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<CLIPEncoder>& self)
              -> CLIPEncoderStatesTorchbind {
            return _serialize_clip_encoder_torchbind(self);
          },
          // __setstate__
          [](CLIPEncoderStatesTorchbind states)
              -> c10::intrusive_ptr<CLIPEncoder> {
            return _deserialize_clip_encoder_torchbind(states);
          });

  m.def("torchtext::generate_sp_model", &generate_sp_model);
  m.def("torchtext::load_sp_model", &load_sp_model);
  m.def("torchtext::load_sp_model_string", &load_sp_model_string);
  m.def("torchtext::gpt2_bpe_pre_tokenizer", &gpt2_bpe_pre_tokenizer);
}

} // namespace torchtext
