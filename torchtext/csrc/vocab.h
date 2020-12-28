#include <torch/script.h>

namespace torchtext {

typedef std::vector<std::string> StringList;
typedef ska_ordered::order_preserving_flat_hash_map<std::string, int64_t>
    IndexDict;
typedef std::tuple<std::string, std::vector<int64_t>, std::vector<std::string>,
                   c10::optional<int64_t>, std::vector<torch::Tensor>>
    VocabStates;

struct Vocab : torch::CustomClassHolder {
private:
  IndexDict stoi_;

public:
  const std::string version_str_ = "0.0.1";
  StringList itos_;
  c10::optional<int64_t> default_index_ = {};

  explicit Vocab(const std::vector<std::string> &tokens);
  explicit Vocab(const StringList &tokens, const IndexDict &stoi);
  int64_t __len__() const;
  int64_t __getitem__(const std::string &token) const;
  void __setitem__(const std::string &token, const int64_t &index);
  void append_token(const std::string &token);
  void insert_token(const std::string &token, const int64_t &index);
  void _delete_token(const std::string &token);
  void set_default_index(const int64_t index);
  int64_t get_default_index() const;
  std::string lookup_token(const int64_t &index);
  std::vector<std::string> lookup_tokens(const std::vector<int64_t> &indices);
  std::vector<int64_t> lookup_indices(const std::vector<std::string> &tokens);
  std::unordered_map<std::string, int64_t> get_stoi() const;
  std::vector<std::string> get_itos() const;
};

VocabStates _serialize_vocab(const c10::intrusive_ptr<Vocab> &self);
c10::intrusive_ptr<Vocab> _deserialize_vocab(VocabStates states);

Vocab _load_vocab_from_file(const std::string &file_path,
                            const int64_t min_freq, const int64_t num_cpus);
Vocab _build_vocab_from_text_file(const std::string &file_path,
                                  const int64_t min_freq,
                                  const int64_t num_cpus,
                                  torch::jit::script::Module tokenizer);

} // namespace torchtext
