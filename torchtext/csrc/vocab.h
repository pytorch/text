#include <common.h>
#include <torch/script.h>

namespace torchtext {

typedef c10::Dict<std::string, int64_t> IndexDict;
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
  explicit Vocab(const StringList &tokens, const IndexDict &stoindex,
                 const std::string &unk_token, const int64_t unk_index);
  int64_t __len__() const;
  int64_t __getitem__(const std::string &token) const;
  void append_token(const std::string &token);
  void insert_token(const std::string &token, const int64_t &index);
  std::string lookup_token(const int64_t &index);
  std::vector<std::string> lookup_tokens(const std::vector<int64_t> &indices);
  std::vector<int64_t> lookup_indices(const std::vector<std::string> &tokens);
  c10::Dict<std::string, int64_t> get_stoi() const;
  std::vector<std::string> get_itos() const;
};

c10::intrusive_ptr<Vocab> _get_vocab_from_states(VocabStates states);
VocabStates _set_vocab_states(const c10::intrusive_ptr<Vocab> &self);
// c10::intrusive_ptr<Vocab> _load_vocab_from_file(const std::string &file_path,
//                                                 const std::string &unk_token,
//                                                 const int64_t min_freq,
//                                                 const int64_t num_cpus);
void register_vocab_ops(torch::Library &m);

} // namespace torchtext
