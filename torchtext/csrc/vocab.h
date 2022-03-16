#pragma once
#include <c10/util/string_view.h>
#include <torch/script.h>
#include <algorithm>

namespace torchtext {

typedef std::vector<std::string> StringList;
typedef ska_ordered::order_preserving_flat_hash_map<std::string, int64_t>
    IndexDict;
typedef std::tuple<
    std::string,
    std::vector<int64_t>,
    std::vector<std::string>,
    std::vector<torch::Tensor>>
    VocabStates;

// sorting using a custom object
struct CompareTokens {
  bool operator()(
      const std::pair<std::string, int64_t>& a,
      const std::pair<std::string, int64_t>& b) {
    if (a.second == b.second) {
      return a.first < b.first;
    }
    return a.second > b.second;
  }
};

int64_t _infer_lines(const std::string& file_path);

struct Vocab : torch::CustomClassHolder {
  static const int32_t MAX_VOCAB_SIZE = 30000000;
  int64_t unk_index_{};
  std::vector<int32_t> stoi_;
  const std::string version_str_ = "0.0.2";
  StringList itos_;
  c10::optional<int64_t> default_index_ = {};

  // TODO: [can we remove this?] we need to keep this constructor, otherwise
  // torch binding gets compilation error: no matching constructor for
  // initialization of 'torchtext::Vocab'
  explicit Vocab(StringList tokens);
  explicit Vocab(
      StringList tokens,
      const c10::optional<int64_t>& default_index);
  int64_t __len__() const;
  int64_t __getitem__(const c10::string_view& token) const;
  bool __contains__(const c10::string_view& token) const;
  void set_default_index(c10::optional<int64_t> index);
  c10::optional<int64_t> get_default_index() const;
  void insert_token(std::string token, const int64_t& index);
  void append_token(std::string token);
  std::string lookup_token(const int64_t& index);
  std::vector<std::string> lookup_tokens(const std::vector<int64_t>& indices);
  std::vector<int64_t> lookup_indices(
      const std::vector<c10::string_view>& tokens);
  std::unordered_map<std::string, int64_t> get_stoi() const;
  std::vector<std::string> get_itos() const;

 protected:
  uint32_t _hash(const c10::string_view& str) const {
    uint32_t h = 2166136261;
    for (size_t i = 0; i < str.size(); i++) {
      h = h ^ uint32_t(uint8_t(str[i]));
      h = h * 16777619;
    }
    return h;
  }

  uint32_t _find(const c10::string_view& w) const {
    uint32_t stoi_size = stoi_.size();
    uint32_t id = _hash(w) % stoi_size;
    while (stoi_[id] != -1 && itos_[stoi_[id]] != w) {
      id = (id + 1) % stoi_size;
    }
    return id;
  }

  void _add(std::string w) {
    uint32_t h = _find(c10::string_view{w.data(), w.size()});
    if (stoi_[h] == -1) {
      itos_.emplace_back(std::move(w));
      stoi_[h] = itos_.size() - 1;
    }
  }
};

VocabStates _serialize_vocab(const c10::intrusive_ptr<Vocab>& self);
c10::intrusive_ptr<Vocab> _deserialize_vocab(VocabStates states);

Vocab _load_vocab_from_file(
    const std::string& file_path,
    const int64_t min_freq,
    const int64_t num_cpus);
Vocab _build_vocab_from_text_file(
    const std::string& file_path,
    const int64_t min_freq,
    const int64_t num_cpus,
    torch::jit::script::Module tokenizer);
} // namespace torchtext
