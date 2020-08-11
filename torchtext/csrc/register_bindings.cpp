#include <regex_tokenizer.h>
#include <sentencepiece.h>
#include <torch/script.h>
#include <vectors.h>
#include <vocab.h>

namespace torchtext {

// Registers our custom classes with torch.
static auto regex_tokenizer =
    torch::class_<RegexTokenizer>("torchtext", "RegexTokenizer")
        .def(torch::init<std::vector<std::string>, std::vector<std::string>,
                         bool>())
        .def("forward", &RegexTokenizer::forward)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<RegexTokenizer> &self)
                -> std::tuple<std::vector<std::string>,
                              std::vector<std::string>, bool> {
              return std::make_tuple(self->patterns_, self->replacements_,
                                     self->to_lower_);
            },
            // __getstate__
            [](std::tuple<std::vector<std::string>, std::vector<std::string>,
                          bool>
                   states) -> c10::intrusive_ptr<RegexTokenizer> {
              auto patterns = std::get<0>(states);
              auto replacements = std::get<1>(states);
              auto to_lower = std::get<2>(states);

              return c10::make_intrusive<RegexTokenizer>(
                  std::move(patterns), std::move(replacements), to_lower);
            });

static auto sentencepiece =
    torch::class_<SentencePiece>("torchtext", "SentencePiece")
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
            // __setstate__
            [](const c10::intrusive_ptr<SentencePiece> &self) -> std::string {
              return self->content_;
            },
            // __getstate__
            [](std::string state) -> c10::intrusive_ptr<SentencePiece> {
              return c10::make_intrusive<SentencePiece>(std::move(state));
            });

static auto vocab =
    torch::class_<Vocab>("torchtext", "Vocab")
        .def(torch::init<StringList, std::string>())
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
            // __setstate__
            [](const c10::intrusive_ptr<Vocab> &self) -> VocabStates {
              return _set_vocab_states(self);
            },
            // __getstate__
            [](VocabStates states) -> c10::intrusive_ptr<Vocab> {
              return _get_vocab_from_states(states);
            });

static auto vectors =
    torch::class_<Vectors>("torchtext", "Vectors")
        .def(torch::init<std::vector<std::string>, torch::Tensor,
                         torch::Tensor>())
        .def("__getitem__", &Vectors::__getitem__)
        .def("lookup_vectors", &Vectors::lookup_vectors)
        .def("__setitem__", &Vectors::__setitem__)
        .def("__len__", &Vectors::__len__)
        .def_pickle(
            // __setstate__
            [](const c10::intrusive_ptr<Vectors> &self) -> VectorsStates {
              return _set_vectors_states(self);
            },
            // __getstate__
            [](VectorsStates states) -> c10::intrusive_ptr<Vectors> {
              return _get_vectors_from_states(states);
            });

// Registers our custom op with torch.
static auto registry =
    torch::RegisterOperators()
        .op("torchtext::generate_sp_model", &generate_sp_model)
        .op("torchtext::_load_token_and_vectors_from_file",
            &_load_token_and_vectors_from_file)
        .op(torch::RegisterOperators::options()
                .schema("torchtext::load_sp_model(str path) -> "
                        "__torch__.torch.classes.torchtext.SentencePiece model")
                .catchAllKernel<decltype(load_sp_model), &load_sp_model>());

} // namespace torchtext
