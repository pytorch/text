#include <c10/util/string_view.h>
#include <torch/script.h>
namespace torchtext {

typedef std::vector<std::string> StringList;
typedef std::tuple<std::string, std::vector<int64_t>, std::vector<std::string>,
                   std::vector<torch::Tensor>>
    VocabStates;

struct Vocab : torch::CustomClassHolder {
  static const int32_t MAX_VOCAB_SIZE = 30000000;
  int64_t unk_index_;
  std::vector<int32_t> stoi_;
  const std::string version_str_ = "0.0.1";
  StringList itos_;
  std::string unk_token_;

  explicit Vocab(const std::vector<std::string> &tokens,
                 const std::string &unk_token);
  int64_t __len__() const;
  int64_t __getitem__(const c10::string_view &token) const;
  void append_token(const std::string &token);
  void insert_token(const std::string &token, const int64_t &index);
  std::string lookup_token(const int64_t &index);
  std::vector<std::string> lookup_tokens(const std::vector<int64_t> &indices);
  std::vector<int64_t>
  lookup_indices(const std::vector<c10::string_view> &tokens);
  std::unordered_map<std::string, int64_t> get_stoi() const;
  std::vector<std::string> get_itos() const;

protected:
  uint32_t _hash(const c10::string_view &str) const {
    uint32_t h = 2166136261;
    for (size_t i = 0; i < str.size(); i++) {
      h = h ^ uint32_t(uint8_t(str[i]));
      h = h * 16777619;
    }
    return h;
  }

  int32_t _find(const c10::string_view &w) const {
    int32_t stoi_size = stoi_.size();
    int32_t id = _hash(w) % stoi_size;
    while (stoi_[id] != -1 && c10::string_view{itos_[stoi_[id]].data(),
                                               itos_[stoi_[id]].size()} != w) {
      id = (id + 1) % stoi_size;
    }
    return id;
  }

  void _add(const std::string &w) {
    int32_t h = _find(c10::string_view{w.data(), w.size()});
    if (stoi_[h] == -1) {
      itos_.push_back(w);
      stoi_[h] = itos_.size() - 1;
    }
  }
};

VocabStates _serialize_vocab(const c10::intrusive_ptr<Vocab> &self);
c10::intrusive_ptr<Vocab> _deserialize_vocab(VocabStates states);

Vocab _load_vocab_from_file(const std::string &file_path,
                            const std::string &unk_token,
                            const uint32_t min_freq, const uint32_t num_cpus);
Vocab _build_vocab_from_text_file(const std::string &file_path,
                                  const std::string &unk_token,
                                  const uint32_t min_freq,
                                  const uint32_t num_cpus,
                                  torch::jit::script::Module tokenizer);

} // namespace torchtext
