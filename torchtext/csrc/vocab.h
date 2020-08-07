#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/script.h>

namespace torchtext {

typedef std::tuple<std::string, std::vector<int64_t>, std::vector<std::string>,
                   std::vector<torch::Tensor>>
    VocabStates;

struct Vocab : torch::CustomClassHolder {
private:
  int64_t unk_index_;
  c10::Dict<std::string, int64_t> stoi_;

public:
  const std::string version_str_ = "0.0.1";
  std::vector<std::string> itos_;
  std::string unk_token_;

  explicit Vocab(const std::vector<std::string> &tokens,
                 const std::string &unk_token);
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

VocabStates _set_vocab_states(const c10::intrusive_ptr<Vocab> &self);
c10::intrusive_ptr<Vocab> _get_vocab_from_states(VocabStates states);

void register_vocab_pybind(pybind11::module m);
} // namespace torchtext
