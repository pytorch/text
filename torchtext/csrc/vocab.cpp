#include <torch/script.h>

namespace torchtext {
namespace {

struct TextVocab : torch::CustomClassHolder {

private:
  std::unordered_map<std::string, int64_t> _vocab;
  std::vector<std::string> _vocab_index_vec;
  std::string _unk_token;

public:
  TextVocab(std::vector<std::string> init, std::string unk_token) {
    _vocab_index_vec = std::vector<std::string>(init.begin(), init.end());
    _unk_token = unk_token;
    if (find(_vocab_index_vec.begin(), _vocab_index_vec.end(), _unk_token) ==
        _vocab_index_vec.end()) {
      _vocab_index_vec.insert(_vocab_index_vec.begin(), _unk_token);
    }
    for (size_t _idx = 0; _idx < _vocab_index_vec.size(); _idx++) {
      _vocab[_vocab_index_vec[_idx]] = _idx;
    }
  }

  int64_t stoi(const std::string &input) const {
    auto it = _vocab.find(input);
    if (it != _vocab.end()) {
      return it->second;
    } else {
      auto _it = _vocab.find(_unk_token);
      return _it->second;
    }
  }

  std::string itos(const int64_t &input) const {
    return _vocab_index_vec[input];
  }
};

static auto textvocab =
    torch::class_<TextVocab>("torchtext", "TextVocab")
        .def(torch::init<std::vector<std::string>, std::string>())
        .def("stoi", &TextVocab::stoi)
        .def("itos", &TextVocab::itos);

} // namespace
} // namespace torchtext
