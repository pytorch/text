#include <pybind11/pybind11.h>
#include <torch/script.h>

namespace torchtext {

typedef std::vector<std::string> StringList;
typedef ska_ordered::order_preserving_flat_hash_map<std::string, int64_t>
    IndexDict;
typedef std::tuple<std::string, std::vector<int64_t>, std::vector<std::string>,
                   std::vector<torch::Tensor>>
    VocabStates;

struct Vocab : torch::CustomClassHolder {
private:
  int64_t unk_index_;
  IndexDict stoi_;

public:
  const std::string version_str_ = "0.0.1";
  StringList itos_;
  std::string unk_token_;

  explicit Vocab(const std::vector<std::string> &tokens,
                 const std::string &unk_token);
  explicit Vocab(const StringList &tokens, const IndexDict &stoi,

                 const std::string &unk_token, const int64_t unk_index);
  int64_t __len__() const;
  int64_t __getitem__(const std::string &token) const;
  void append_token(const std::string &token);
  void insert_token(const std::string &token, const int64_t &index);
  std::string lookup_token(const int64_t &index);
  std::vector<std::string> lookup_tokens(const std::vector<int64_t> &indices);
  std::vector<int64_t> lookup_indices(const std::vector<std::string> &tokens);
  std::unordered_map<std::string, int64_t> get_stoi() const;
  std::vector<std::string> get_itos() const;
};

c10::intrusive_ptr<Vocab> _get_vocab_from_states(VocabStates states);
VocabStates _set_vocab_states(const c10::intrusive_ptr<Vocab> &self);
Vocab _load_vocab_from_file(const std::string &file_path,
                            const std::string &unk_token,
                            const int64_t min_freq, const int64_t num_cpus);
Vocab _load_vocab_from_raw_text_file(const std::string &file_path,
                                     const std::string &unk_token,
                                     const int64_t min_freq,
                                     const int64_t num_cpus,
                                     py::object tokenizer);

} // namespace torchtext
